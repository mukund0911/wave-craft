"""
Speech Processing Agent (Revamped)
Uses WhisperX for transcription and Chatterbox for voice cloning.

Replaces the old AssemblyAI + VoiceCraft pipeline entirely.
"""

import os
import base64
import logging
import asyncio
import tempfile
from io import BytesIO
from typing import Dict, Any, List, Optional
from pydub import AudioSegment
from .base_agent import MCPAgent
from .whisperx_agent import WhisperXAgent
from .chatterbox_agent import ChatterboxAgent

logger = logging.getLogger(__name__)


class SpeechProcessingAgent(MCPAgent):
    """
    Orchestrates the full speech processing pipeline:
    1. WhisperX transcription with speaker diarization
    2. User edits transcript (frontend)
    3. Chatterbox generates modified audio with voice cloning + emotion

    This agent is the bridge between routes.py and the specialized agents.
    """

    def __init__(self, hf_token: str = None, device: str = None):
        super().__init__("speech_processing", [
            "transcribe", "process_modifications", "get_status"
        ])

        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.device = device

        # Initialize sub-agents
        self.whisperx_agent = WhisperXAgent(
            hf_token=self.hf_token,
            device=self.device
        )
        self.chatterbox_agent = ChatterboxAgent(device=self.device)

        logger.info("SpeechProcessingAgent initialized")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main dispatch for speech processing requests.

        Actions:
        - transcribe: Transcribe audio file
        - process_modifications: Generate modified audio from edited transcripts
        """
        action = request.get("action")

        if action == "transcribe":
            return self._handle_transcription(request)
        elif action == "process_modifications":
            return await self._handle_modifications(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")

    # ──────────────────────────────────────────────────
    # Transcription (WhisperX)
    # ──────────────────────────────────────────────────

    def transcribe_audio(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        """
        Transcribe audio file with speaker diarization.

        This is the synchronous entry point called by routes.py.
        WhisperX is fast enough to not need async polling.

        Args:
            audio_path: Path to uploaded audio file
            num_speakers: Optional expected speaker count

        Returns:
            Dict with conversations list and full_audio base64
        """
        logger.info(f"Starting transcription: {audio_path}")

        # Only optimize audio locally — Modal handles its own format conversion
        if self.whisperx_agent.device == "remote":
            optimized_path = audio_path
        else:
            optimized_path = self._optimize_audio(audio_path)

        try:
            # Run WhisperX
            result = self.whisperx_agent.transcribe_file(
                optimized_path,
                num_speakers=num_speakers
            )

            if result["status"] == "completed":
                logger.info(f"✓ Transcription complete: {len(result['conversations'])} segments")
            else:
                logger.error(f"Transcription failed: {result.get('error')}")

            return result

        finally:
            # Clean up optimized file if different from original
            if optimized_path != audio_path and os.path.exists(optimized_path):
                os.remove(optimized_path)

    def _handle_transcription(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Internal handler for process_request"""
        audio_path = request.get("audio_path")
        if not audio_path:
            return self.create_response(False, error="Missing audio_path")

        result = self.transcribe_audio(
            audio_path,
            num_speakers=request.get("num_speakers")
        )

        if result["status"] == "completed":
            return self.create_response(True, result)
        else:
            return self.create_response(False, error=result.get("error", "Unknown"))

    # ──────────────────────────────────────────────────
    # Modification Processing (Chatterbox)
    # ──────────────────────────────────────────────────

    async def _handle_modifications(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process modified transcripts and generate new audio.

        For each conversation segment:
        1. Compare original vs modified text
        2. If changed: use Chatterbox to re-synthesize with cloned voice
        3. If unchanged: keep original audio
        4. Assemble all segments into final audio

        Args (in request):
            conversations: List of conversation dicts with original + modified
            full_audio: Base64 of original full audio

        Returns:
            response with final_audio_base64
        """
        conversations = request.get("conversations", [])
        full_audio_b64 = request.get("full_audio", "")

        if not conversations:
            return self.create_response(False, error="No conversations provided")

        try:
            # Process each segment
            processed_segments = []

            for conv_item in conversations:
                # Extract the conversation data (might be nested)
                if isinstance(conv_item, dict):
                    for key, conv in conv_item.items():
                        result = await self._process_single_segment(conv)
                        processed_segments.append(result)
                else:
                    continue

            # Assemble final audio
            final_audio_b64 = self._assemble_audio(processed_segments)

            logger.info(f"✓ All {len(processed_segments)} segments processed")

            return self.create_response(True, {
                "modified_audio": final_audio_b64,
                "segments_processed": len(processed_segments),
                "segments_changed": sum(1 for s in processed_segments if s.get("changed")),
                "stats": self.chatterbox_agent.get_stats()
            })

        except Exception as e:
            logger.error(f"Modification processing failed: {e}", exc_info=True)
            return self.create_response(False, error=str(e))

    async def _process_single_segment(self, conv: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single conversation segment.

        Returns dict with audio_base64 and metadata.
        """
        original = conv.get("original", {})
        modified = conv.get("modified", {})
        speaker = conv.get("speaker", "A")

        original_text = original.get("text", "")
        modified_text = modified.get("text", "")
        emotions = modified.get("emotions", [])
        exaggeration = modified.get("exaggeration", 0.5)
        reference_audio_b64 = original.get("speaker_audio", "")

        # Determine if anything changed
        text_changed = original_text.strip() != modified_text.strip()
        has_emotions = len(emotions) > 0

        if not text_changed and not has_emotions:
            # No changes — passthrough original audio
            logger.info(f"Speaker {speaker}: No changes")
            return {
                "audio_base64": reference_audio_b64,
                "changed": False,
                "speaker": speaker
            }

        # Use Chatterbox to re-synthesize
        logger.info(f"Speaker {speaker}: Processing changes (text_changed={text_changed}, emotions={len(emotions)})")

        result = self.chatterbox_agent._modify_speech({
            "reference_audio_b64": reference_audio_b64,
            "original_text": original_text,
            "modified_text": modified_text,
            "emotions": emotions,
            "exaggeration": exaggeration
        })

        if result["success"] and result.get("data"):
            return {
                "audio_base64": result["data"].get("audio_base64", reference_audio_b64),
                "changed": result["data"].get("changed", False),
                "speaker": speaker,
                "method": result["data"].get("method", "unknown")
            }
        else:
            # Fallback to original
            logger.warning(f"Speaker {speaker}: Chatterbox failed, using original")
            return {
                "audio_base64": reference_audio_b64,
                "changed": False,
                "speaker": speaker,
                "method": "fallback"
            }

    def _assemble_audio(self, segments: List[Dict[str, Any]],
                         crossfade_ms: int = 50) -> str:
        """
        Assemble processed segments into final audio with crossfades.

        Args:
            segments: List of dicts with audio_base64
            crossfade_ms: Crossfade duration between segments

        Returns:
            Base64 string of assembled WAV audio
        """
        if not segments:
            return ""

        combined = AudioSegment.empty()

        for i, segment in enumerate(segments):
            audio_b64 = segment.get("audio_base64", "")
            if not audio_b64:
                continue

            try:
                audio_bytes = base64.b64decode(audio_b64)
                seg_audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")

                if len(combined) > 0 and len(seg_audio) > crossfade_ms:
                    combined = combined.append(seg_audio, crossfade=crossfade_ms)
                else:
                    combined += seg_audio

            except Exception as e:
                logger.warning(f"Failed to process segment {i}: {e}")
                continue

        if len(combined) == 0:
            return ""

        # Export as WAV
        buffer = BytesIO()
        combined.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # ──────────────────────────────────────────────────
    # Audio Optimization
    # ──────────────────────────────────────────────────

    def _optimize_audio(self, audio_path: str) -> str:
        """
        Optimize audio for transcription:
        - Convert to WAV
        - Mono channel
        - 16kHz sample rate (WhisperX default)

        Returns path to optimized file (may be same as input for WAVs).
        """
        try:
            audio = AudioSegment.from_file(audio_path)

            # Check if optimization needed
            needs_optimization = (
                audio.channels > 1 or
                audio.frame_rate != 16000 or
                not audio_path.lower().endswith('.wav')
            )

            if not needs_optimization:
                return audio_path

            # Optimize
            audio = audio.set_channels(1).set_frame_rate(16000)

            optimized_path = tempfile.mktemp(suffix=".wav", prefix="optimized_")
            audio.export(optimized_path, format="wav")

            logger.info(f"Audio optimized: {audio_path} → {optimized_path}")
            return optimized_path

        except Exception as e:
            logger.warning(f"Audio optimization failed, using original: {e}")
            return audio_path