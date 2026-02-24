"""
Chatterbox Voice Cloning & TTS Agent
Supports two modes:
- Remote: Calls Modal GPU service (when MODAL_SERVICE_URL is set)
- Local:  Runs Chatterbox locally (dev mode)
"""

import os
import base64
import logging
import tempfile
import requests as http_requests
from io import BytesIO
from typing import Dict, Any, Optional, List
from pydub import AudioSegment
from .base_agent import MCPAgent

logger = logging.getLogger(__name__)

MODAL_TTS_URL = os.environ.get("MODAL_TTS_URL", "")

# Lazy-loaded globals
_chatterbox_model = None
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _load_chatterbox(device: str):
    """Load Chatterbox model (lazy, cached)"""
    global _chatterbox_model
    if _chatterbox_model is not None:
        return _chatterbox_model

    logger.info("Loading Chatterbox TTS model...")
    from chatterbox.tts import ChatterboxTTS

    _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
    logger.info(f"✓ Chatterbox loaded on {device}")
    return _chatterbox_model


# Supported emotion types
EMOTION_TYPES = {
    "happy", "sad", "angry", "excited", "calm",
    "fearful", "surprised", "neutral"
}

# Paralinguistic tags that Chatterbox supports natively
PARALINGUISTIC_TAGS = {
    "[laugh]", "[sigh]", "[gasp]", "[cough]", "[chuckle]"
}


class ChatterboxAgent(MCPAgent):
    """
    Voice cloning and TTS agent using Chatterbox.

    Solves the VoiceCraft word-insertion problem:
    VoiceCraft tries to edit existing audio (fails on new words).
    Chatterbox does full zero-shot synthesis from reference audio (works for any text).
    """

    def __init__(self, device: str = None):
        super().__init__("chatterbox_tts", [
            "clone_voice", "generate_speech", "modify_speech"
        ])

        # Avoid loading torch locally if using Modal GPU service
        if MODAL_TTS_URL:
            self.device = "remote"
        else:
            torch = _get_torch()
            if device:
                self.device = device
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        self._model = None
        self._stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "total_inference_time": 0.0
        }

        logger.info(f"ChatterboxAgent initialized: device={self.device}")

    def _ensure_model(self):
        """Lazy-load model on first use"""
        if self._model is None:
            self._model = _load_chatterbox(self.device)

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")

        if action == "modify_speech":
            return self._modify_speech(request)
        elif action == "clone_voice":
            return self._clone_voice(request)
        elif action == "generate_speech":
            return self._generate_speech(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")

    def _modify_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate speech for modified text using reference speaker audio.
        Routes to Modal if MODAL_TTS_URL is set, otherwise runs locally.
        """
        if MODAL_TTS_URL:
            return self._modify_speech_remote(request)
        return self._modify_speech_local(request)

    def _modify_speech_remote(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send TTS request to Modal GPU service."""
        try:
            reference_audio_b64 = request.get("reference_audio_b64", "")
            original_text = request.get("original_text", "")
            modified_text = request.get("modified_text", "")
            emotions = request.get("emotions", [])
            exaggeration = request.get("exaggeration", 0.5)

            # Skip if no changes
            text_changed = original_text.strip() != modified_text.strip()
            if not text_changed and not emotions:
                return self.create_response(True, {
                    "audio_base64": reference_audio_b64,
                    "changed": False,
                    "method": "passthrough"
                })

            processed_text = self._apply_emotion_tags(modified_text, emotions)

            payload = {
                "text": processed_text,
                "reference_audio": reference_audio_b64,
                "exaggeration": max(0.0, min(1.0, float(exaggeration))),
            }

            logger.info(f"Sending TTS to Modal: '{processed_text[:60]}...'")
            response = http_requests.post(
                MODAL_TTS_URL,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()

            if result.get("status") == "completed":
                return self.create_response(True, {
                    "audio_base64": result["audio"],
                    "changed": True,
                    "method": "modal_chatterbox",
                })
            else:
                raise Exception(result.get("error", "Modal TTS failed"))

        except Exception as e:
            logger.error(f"Modal TTS failed: {e}", exc_info=True)
            return self.create_response(True, {
                "audio_base64": request.get("reference_audio_b64", ""),
                "changed": False,
                "method": "fallback",
                "error": str(e)
            })

    def _modify_speech_local(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Run Chatterbox locally (dev mode)."""
        import time

        self._stats["total_requests"] += 1
        start_time = time.time()

        try:
            self._ensure_model()

            reference_audio_b64 = request.get("reference_audio_b64")
            original_text = request.get("original_text", "")
            modified_text = request.get("modified_text", "")
            emotions = request.get("emotions", [])
            exaggeration = request.get("exaggeration", 0.5)

            if not reference_audio_b64:
                return self.create_response(False, error="Missing reference_audio_b64")
            if not modified_text:
                return self.create_response(False, error="Missing modified_text")

            # Check if text actually changed
            text_changed = original_text.strip() != modified_text.strip()

            if not text_changed and not emotions:
                # No changes — return original audio
                logger.info("No text or emotion changes — returning original audio")
                return self.create_response(True, {
                    "audio_base64": reference_audio_b64,
                    "changed": False,
                    "method": "passthrough"
                })

            # Process text with emotion tags
            processed_text = self._apply_emotion_tags(modified_text, emotions)

            # Save reference audio to temp file (Chatterbox needs file path)
            ref_audio_path = self._save_temp_audio(reference_audio_b64)

            try:
                # Clamp exaggeration to valid range
                exaggeration = max(0.0, min(1.0, float(exaggeration)))

                logger.info(f"Generating speech:")
                logger.info(f"  Text: '{processed_text[:80]}...'")
                logger.info(f"  Exaggeration: {exaggeration}")
                logger.info(f"  Text changed: {text_changed}")

                # Generate with Chatterbox
                wav = self._model.generate(
                    text=processed_text,
                    audio_prompt_path=ref_audio_path,
                    exaggeration=exaggeration,
                )

                # Convert to base64
                output_b64 = self._tensor_to_base64(wav)

                inference_time = time.time() - start_time
                self._stats["successful"] += 1
                self._stats["total_inference_time"] += inference_time

                logger.info(f"✓ Generated in {inference_time:.2f}s")

                return self.create_response(True, {
                    "audio_base64": output_b64,
                    "changed": True,
                    "method": "chatterbox",
                    "inference_time": inference_time,
                    "exaggeration": exaggeration,
                    "text_length": len(processed_text)
                })

            finally:
                # Clean up temp file
                if os.path.exists(ref_audio_path):
                    os.remove(ref_audio_path)

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Voice cloning failed: {e}", exc_info=True)

            # Fallback: return original audio
            return self.create_response(True, {
                "audio_base64": request.get("reference_audio_b64", ""),
                "changed": False,
                "method": "fallback",
                "error": str(e)
            })

    def _clone_voice(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for _modify_speech"""
        return self._modify_speech(request)

    def _generate_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate speech for artificial speakers (no reference audio from conversation).
        Uses a default neutral voice or specified characteristics.
        """
        import time

        self._stats["total_requests"] += 1
        start_time = time.time()

        try:
            self._ensure_model()

            text = request.get("text", "")
            speaker_characteristics = request.get("speaker_characteristics", "")
            exaggeration = request.get("exaggeration", 0.5)

            if not text:
                return self.create_response(False, error="Missing text")

            # For artificial speakers without reference audio,
            # use Chatterbox's default voice
            logger.info(f"Generating artificial speech: '{text[:60]}...'")

            wav = self._model.generate(
                text=text,
                exaggeration=max(0.0, min(1.0, float(exaggeration))),
            )

            output_b64 = self._tensor_to_base64(wav)
            inference_time = time.time() - start_time
            self._stats["successful"] += 1

            return self.create_response(True, {
                "audio_base64": output_b64,
                "voice_used": "chatterbox_default",
                "inference_time": inference_time,
                "text_length": len(text)
            })

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Speech generation failed: {e}", exc_info=True)
            return self.create_response(False, error=f"Failed to generate speech: {str(e)}")

    def _apply_emotion_tags(self, text: str, emotions: List[Dict]) -> str:
        """
        Apply emotion tags to text for Chatterbox processing.

        Chatterbox supports native paralinguistic tags like [laugh], [sigh], etc.
        For tone emotions (happy, sad, etc.), we rely on the exaggeration parameter
        and text-level styling.

        Args:
            text: Original text
            emotions: List of emotion dicts:
                [{"type": "happy", "range": [0, 5], "intensity": 0.8}]
                [{"type": "[laugh]", "position": 5}]

        Returns:
            Processed text with tags inserted
        """
        if not emotions:
            return text

        # Sort by position (reverse order for safe insertion)
        insertions = []
        for emotion in emotions:
            etype = emotion.get("type", "")

            # Paralinguistic tags get inserted into text
            if etype in PARALINGUISTIC_TAGS:
                position = emotion.get("position", len(text))
                insertions.append((position, etype))

        # Insert tags in reverse order to preserve positions
        result = text
        for position, tag in sorted(insertions, key=lambda x: x[0], reverse=True):
            # Insert tag at position with space padding
            result = result[:position] + f" {tag} " + result[position:]

        return result.strip()

    def _save_temp_audio(self, audio_b64: str) -> str:
        """Save base64 audio to temp WAV file, return path"""
        audio_bytes = base64.b64decode(audio_b64)
        audio_seg = AudioSegment.from_file(BytesIO(audio_bytes))

        # Ensure WAV format, mono, 22050 Hz
        audio_seg = audio_seg.set_channels(1).set_frame_rate(22050)

        temp_path = tempfile.mktemp(suffix=".wav", prefix="chatterbox_ref_")
        audio_seg.export(temp_path, format="wav")
        return temp_path

    def _tensor_to_base64(self, wav_tensor) -> str:
        """Convert PyTorch tensor to base64 WAV string"""
        import torchaudio

        buffer = BytesIO()

        # Chatterbox returns tensor at model sample rate (typically 24000)
        sample_rate = self._model.sr if hasattr(self._model, 'sr') else 24000

        # Ensure correct shape [channels, samples]
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        torchaudio.save(buffer, wav_tensor.cpu(), sample_rate, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self._stats["total_requests"]
        return {
            **self._stats,
            "success_rate": (self._stats["successful"] / total * 100) if total > 0 else 0,
            "avg_inference_time": (
                self._stats["total_inference_time"] / self._stats["successful"]
                if self._stats["successful"] > 0 else 0
            ),
            "device": self.device
        }
