"""
OpenAI Text-to-Speech Agent (LangGraph refactor)
Fast TTS fallback when latency matters more than voice cloning quality.
"""
import os
import base64
import logging
from io import BytesIO
from typing import Dict, Any
from .base import create_response

logger = logging.getLogger(__name__)


class TextToSpeechAgent:
    """OpenAI TTS for low-latency artificial speaker generation."""

    VOICE_MAPPING = {
        "male": {"calm": "onyx", "energetic": "echo", "warm": "nova"},
        "female": {"calm": "shimmer", "energetic": "alloy", "warm": "nova"}
    }

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set — OpenAI TTS will fail")
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def generate_speech(self, text: str, speaker_characteristics: str = "", speed: float = 1.0) -> Dict[str, Any]:
        if not self.client:
            return create_response("openai_tts", False, error="OpenAI client not initialized")
        if not text:
            return create_response("openai_tts", False, error="Missing text")

        voice = self._select_voice(speaker_characteristics)
        effective_speed = speed if speed else self._determine_speed(speaker_characteristics)

        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=effective_speed
            )
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return create_response("openai_tts", True, {
                "audio_base64": audio_base64,
                "voice_used": voice,
                "speed_used": effective_speed,
                "text_length": len(text)
            })
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return create_response("openai_tts", False, error=f"Failed to generate speech: {str(e)}")

    def _select_voice(self, speaker_characteristics: str) -> str:
        default_voice = "alloy"
        lower_chars = speaker_characteristics.lower()
        if "male" in lower_chars or "man" in lower_chars:
            if "calm" in lower_chars or "relaxed" in lower_chars:
                return self.VOICE_MAPPING["male"]["calm"]
            elif "energetic" in lower_chars or "excited" in lower_chars:
                return self.VOICE_MAPPING["male"]["energetic"]
            else:
                return self.VOICE_MAPPING["male"]["warm"]
        elif "female" in lower_chars or "woman" in lower_chars:
            if "calm" in lower_chars or "relaxed" in lower_chars:
                return self.VOICE_MAPPING["female"]["calm"]
            elif "energetic" in lower_chars or "excited" in lower_chars:
                return self.VOICE_MAPPING["female"]["energetic"]
            else:
                return self.VOICE_MAPPING["female"]["warm"]
        return default_voice

    def _determine_speed(self, speaker_characteristics: str) -> float:
        lower_chars = speaker_characteristics.lower()
        if "fast" in lower_chars or "quick" in lower_chars or "energetic" in lower_chars:
            return 1.25
        elif "slow" in lower_chars or "calm" in lower_chars or "relaxed" in lower_chars:
            return 0.85
        return 1.0
