"""
WhisperX Speech-to-Text Agent
Supports two modes:
- Remote: Calls Modal GPU service (when MODAL_SERVICE_URL is set)
- Local:  Runs WhisperX locally (dev mode, when MODAL_SERVICE_URL is not set)
"""

import os
import base64
import logging
import requests as http_requests
from typing import Dict, Any, Optional
from io import BytesIO
from pydub import AudioSegment
from .base_agent import MCPAgent

logger = logging.getLogger(__name__)

MODAL_TRANSCRIBE_URL = os.environ.get("MODAL_TRANSCRIBE_URL", "")

# Lazy-loaded globals to avoid import overhead on startup
_whisperx = None
_torch = None


def _patch_torch_load():
    """
    PyTorch 2.6 changed torch.load default to weights_only=True,
    breaking pyannote/whisperx model loading. Patch it back to False.
    """
    import torch
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


def _get_whisperx():
    global _whisperx
    if _whisperx is None:
        _patch_torch_load()
        import whisperx as wx
        _whisperx = wx
    return _whisperx


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class WhisperXAgent(MCPAgent):
    """
    Local speech-to-text with speaker diarization and word-level timestamps.

    Replaces AssemblyAI entirely — zero API cost.
    """

    def __init__(self, hf_token: str = "", model_size: str = "large-v2",
                 compute_type: str = "float16", device: str = None):
        super().__init__("whisperx_stt", ["transcribe_audio", "get_status"])

        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.model_size = model_size
        self.compute_type = compute_type

        # Avoid loading torch locally if using Modal GPU service
        if MODAL_TRANSCRIBE_URL:
            self.device = "remote"
            self.compute_type = "remote"
        else:
            torch = _get_torch()
            if device:
                self.device = device
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                self.compute_type = "int8"  # CPU needs int8

        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_pipeline = None

        logger.info(f"WhisperXAgent initialized: model={model_size}, device={self.device}, compute={self.compute_type}")

    def _load_model(self):
        """Lazy-load WhisperX model"""
        if self._model is not None:
            return

        whisperx = _get_whisperx()
        logger.info(f"Loading WhisperX model: {self.model_size}...")
        self._model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type
        )
        logger.info("✓ WhisperX model loaded")

    def _load_align_model(self, language_code: str = "en"):
        """Lazy-load alignment model for word-level timestamps"""
        if self._align_model is not None:
            return

        whisperx = _get_whisperx()
        logger.info("Loading alignment model...")
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=self.device
        )
        logger.info("✓ Alignment model loaded")

    def _load_diarize_pipeline(self):
        """Lazy-load pyannote diarization pipeline"""
        if self._diarize_pipeline is not None:
            return

        if not self.hf_token:
            logger.warning("No HF_TOKEN set — diarization disabled. Set HF_TOKEN env var.")
            return

        logger.info("Loading diarization pipeline...")
        from whisperx.diarize import DiarizationPipeline
        self._diarize_pipeline = DiarizationPipeline(
            token=self.hf_token,
            device=self.device
        )
        logger.info("✓ Diarization pipeline loaded")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")

        if action == "transcribe_audio":
            return self._transcribe(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")

    def transcribe_file(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        """
        Transcribe audio file with speaker diarization and word-level timestamps.

        Routes to Modal GPU service if MODAL_TRANSCRIBE_URL is set,
        otherwise falls back to local WhisperX inference.
        """
        if MODAL_TRANSCRIBE_URL:
            return self._transcribe_remote(audio_path, num_speakers)
        return self._transcribe_local(audio_path, num_speakers)

    def _transcribe_remote(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        """Send audio to Modal GPU service for transcription."""
        try:
            logger.info(f"Transcribing via Modal: {audio_path}")

            # Read audio file and encode to base64
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {"audio": audio_b64}
            if num_speakers:
                payload["num_speakers"] = num_speakers

            response = http_requests.post(
                MODAL_TRANSCRIBE_URL,
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "completed":
                logger.info(f"✓ Modal transcription complete: {len(result.get('conversations', []))} segments")
            else:
                logger.error(f"Modal transcription error: {result.get('error', 'unknown')}")

            return result

        except Exception as e:
            logger.error(f"Modal transcription failed: {e}", exc_info=True)
            return {"status": "error", "error": f"Modal service error: {str(e)}"}

    def _transcribe_local(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        """Run WhisperX locally (dev mode)."""
        whisperx = _get_whisperx()

        # Load models on demand
        self._load_model()

        try:
            logger.info(f"Transcribing: {audio_path}")

            # Step 1: Load and convert audio
            audio = whisperx.load_audio(audio_path)

            # Step 2: Transcribe with Whisper
            logger.info("Step 1/3: Transcribing...")
            result = self._model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", "en")
            logger.info(f"  Detected language: {detected_language}")
            logger.info(f"  Segments: {len(result['segments'])}")

            # Step 3: Align for word-level timestamps
            logger.info("Step 2/3: Aligning words...")
            self._load_align_model(detected_language)
            result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            logger.info(f"  Aligned {sum(len(s.get('words', [])) for s in result['segments'])} words")

            # Step 4: Speaker diarization
            logger.info("Step 3/3: Diarizing speakers...")
            self._load_diarize_pipeline()

            if self._diarize_pipeline is not None:
                diarize_args = {}
                if num_speakers:
                    diarize_args["num_speakers"] = num_speakers

                diarize_segments = self._diarize_pipeline(audio_path, **diarize_args)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            else:
                # No diarization — assign all to SPEAKER_00
                for segment in result["segments"]:
                    segment["speaker"] = "SPEAKER_00"

            # Step 5: Build conversation structure
            conversations = self._build_conversations(result, audio_path)

            # Step 6: Create full audio base64
            full_audio_b64 = self._audio_to_base64(audio_path)

            logger.info(f"✓ Transcription complete: {len(conversations)} conversation segments")

            return {
                "status": "completed",
                "conversations": conversations,
                "full_audio": full_audio_b64,
                "language": detected_language
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    def _transcribe(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Internal handler for process_request"""
        audio_path = request.get("audio_path")
        if not audio_path:
            return self.create_response(False, error="Missing audio_path")

        num_speakers = request.get("num_speakers")
        result = self.transcribe_file(audio_path, num_speakers)

        if result["status"] == "completed":
            return self.create_response(True, result)
        else:
            return self.create_response(False, error=result.get("error", "Unknown error"))

    def _build_conversations(self, result: Dict, audio_path: str) -> list:
        """
        Convert WhisperX output to WaveCrafter conversation structure.

        Each conversation entry:
        {
            "speaker": "A",
            "original": {
                "text": "transcribed text",
                "speaker_audio": "<base64>",
                "start": 0.0,
                "end": 2.5
            },
            "modified": {
                "text": "transcribed text",
                "emotions": []
            }
        }
        """
        conversations = []

        # Load full audio for segment extraction
        full_audio = AudioSegment.from_file(audio_path)

        # Map speaker IDs to letters
        speaker_map = {}
        speaker_counter = 0

        for segment in result.get("segments", []):
            speaker_id = segment.get("speaker", "SPEAKER_00")

            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = chr(ord('A') + speaker_counter)
                speaker_counter += 1

            speaker_label = speaker_map[speaker_id]
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)

            # Extract this segment's audio
            segment_audio = full_audio[start_ms:end_ms]
            segment_b64 = self._segment_to_base64(segment_audio)

            # Build word list with timestamps
            words_with_timestamps = []
            for word_info in segment.get("words", []):
                words_with_timestamps.append({
                    "word": word_info.get("word", ""),
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "score": word_info.get("score", 1.0)
                })

            conv_key = f"conversation_{len(conversations)}"
            conversations.append({
                conv_key: {
                    "speaker": speaker_label,
                    "original": {
                        "text": segment["text"].strip(),
                        "speaker_audio": segment_b64,
                        "start": segment["start"],
                        "end": segment["end"],
                        "words": words_with_timestamps
                    },
                    "modified": {
                        "text": segment["text"].strip(),
                        "emotions": []
                    }
                }
            })

        return conversations

    def _audio_to_base64(self, audio_path: str) -> str:
        """Convert audio file to base64 WAV string"""
        audio = AudioSegment.from_file(audio_path)
        buffer = BytesIO()
        audio.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _segment_to_base64(self, audio_segment: AudioSegment) -> str:
        """Convert AudioSegment to base64 WAV string"""
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
