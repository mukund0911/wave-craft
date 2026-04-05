"""
Chatterbox Voice Cloning & TTS Agent (LangGraph refactor)
Supports remote (Modal GPU) or local inference.
"""
import os
import base64
import logging
import tempfile
import time
import threading
import requests as http_requests
from io import BytesIO
from typing import Dict, Any, List
from pydub import AudioSegment
from .base import create_response

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
    global _chatterbox_model
    if _chatterbox_model is not None:
        return _chatterbox_model
    logger.info("Loading Chatterbox TTS model...")
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        _chatterbox_model = ChatterboxTurboTTS.from_pretrained(device=device)
        logger.info(f"Chatterbox Turbo loaded on {device}")
    except (ImportError, Exception) as e:
        logger.warning(f"Turbo unavailable ({e}), falling back to base model")
        from chatterbox.tts import ChatterboxTTS
        _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
        logger.info(f"Chatterbox base loaded on {device}")
    return _chatterbox_model


EMOTION_TYPES = {"happy", "sad", "angry", "excited", "calm", "fearful", "surprised", "neutral"}
PARALINGUISTIC_TAGS = {
    "[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]",
    "[groan]", "[sniff]", "[clear throat]", "[shush]"
}


def _request_with_retry(url, payload, timeout, max_retries=2):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            response = http_requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response
        except http_requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries:
                delay = (attempt + 1)
                logger.warning(f"Request attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                time.sleep(delay)
    raise last_err


class ChatterboxAgent:
    """Voice cloning and TTS using Chatterbox."""

    def __init__(self, device: str = None):
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
        self._stats_lock = threading.Lock()
        logger.info(f"ChatterboxAgent: device={self.device}")

    def _ensure_model(self):
        if self._model is None:
            self._model = _load_chatterbox(self.device)

    def modify_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if MODAL_TTS_URL:
            return self._modify_speech_remote(request)
        return self._modify_speech_local(request)

    def _modify_speech_remote(self, request: Dict[str, Any]) -> Dict[str, Any]:
        reference_audio_b64 = request.get("reference_audio_b64", "")
        original_text = request.get("original_text", "")
        modified_text = request.get("modified_text", "")
        emotions = request.get("emotions", [])
        exaggeration = request.get("exaggeration", 0.5)

        text_changed = original_text.strip() != modified_text.strip()
        if not text_changed and not emotions:
            return create_response("chatterbox", True, {
                "audio_base64": reference_audio_b64,
                "changed": False,
                "method": "passthrough"
            })

        processed_text = self._apply_emotion_tags(modified_text, emotions)
        effective_exaggeration = float(exaggeration)
        for emotion in emotions:
            if emotion.get("type") in EMOTION_TYPES:
                effective_exaggeration = emotion.get("intensity", 0.5) * 2.0
                break
        effective_exaggeration = max(0.0, min(2.0, effective_exaggeration))

        cfg_weight = 0.5
        if effective_exaggeration > 0.7:
            cfg_weight = max(0.1, 0.5 - (effective_exaggeration - 0.7) * 0.3)

        payload = {
            "text": processed_text,
            "reference_audio": reference_audio_b64,
            "exaggeration": effective_exaggeration,
            "cfg_weight": cfg_weight,
        }

        with self._stats_lock:
            self._stats["total_requests"] += 1

        try:
            t_start = time.time()
            response = _request_with_retry(MODAL_TTS_URL, payload, timeout=120)
            elapsed = time.time() - t_start
            result = response.json()
            if result.get("status") == "completed":
                with self._stats_lock:
                    self._stats["successful"] += 1
                    self._stats["total_inference_time"] += elapsed
                return create_response("chatterbox", True, {
                    "audio_base64": result["audio"],
                    "changed": True,
                    "method": "modal_chatterbox",
                })
            else:
                raise Exception(result.get("error", "Modal TTS failed"))
        except Exception as e:
            logger.error(f"Modal TTS failed: {e}")
            with self._stats_lock:
                self._stats["failed"] += 1
            return create_response("chatterbox", True, {
                "audio_base64": reference_audio_b64,
                "changed": False,
                "method": "fallback",
                "error": str(e)
            })

    def _modify_speech_local(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._stats_lock:
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
                return create_response("chatterbox", False, error="Missing reference_audio_b64")
            if not modified_text:
                return create_response("chatterbox", False, error="Missing modified_text")

            text_changed = original_text.strip() != modified_text.strip()
            if not text_changed and not emotions:
                return create_response("chatterbox", True, {
                    "audio_base64": reference_audio_b64,
                    "changed": False,
                    "method": "passthrough"
                })

            processed_text = self._apply_emotion_tags(modified_text, emotions)
            ref_audio_path = self._save_temp_audio(reference_audio_b64)
            try:
                effective_exaggeration = float(exaggeration)
                for emotion in emotions:
                    if emotion.get("type") in EMOTION_TYPES:
                        effective_exaggeration = emotion.get("intensity", 0.5) * 2.0
                        break
                effective_exaggeration = max(0.0, min(2.0, effective_exaggeration))
                cfg_weight = 0.5
                if effective_exaggeration > 0.7:
                    cfg_weight = max(0.1, 0.5 - (effective_exaggeration - 0.7) * 0.3)

                wav = self._model.generate(
                    text=processed_text,
                    audio_prompt_path=ref_audio_path,
                    exaggeration=effective_exaggeration,
                    cfg_weight=cfg_weight,
                )
                output_b64 = self._tensor_to_base64(wav)
                inference_time = time.time() - start_time
                with self._stats_lock:
                    self._stats["successful"] += 1
                    self._stats["total_inference_time"] += inference_time
                return create_response("chatterbox", True, {
                    "audio_base64": output_b64,
                    "changed": True,
                    "method": "chatterbox",
                    "inference_time": inference_time,
                    "exaggeration": exaggeration,
                    "text_length": len(processed_text)
                })
            finally:
                if os.path.exists(ref_audio_path):
                    os.remove(ref_audio_path)
        except Exception as e:
            with self._stats_lock:
                self._stats["failed"] += 1
            logger.error(f"Voice cloning failed: {e}", exc_info=True)
            return create_response("chatterbox", True, {
                "audio_base64": request.get("reference_audio_b64", ""),
                "changed": False,
                "method": "fallback",
                "error": str(e)
            })

    def generate_speech(self, text: str, exaggeration: float = 0.5) -> Dict[str, Any]:
        with self._stats_lock:
            self._stats["total_requests"] += 1
        start_time = time.time()
        try:
            self._ensure_model()
            wav = self._model.generate(
                text=text,
                exaggeration=max(0.0, min(1.0, float(exaggeration))),
            )
            output_b64 = self._tensor_to_base64(wav)
            inference_time = time.time() - start_time
            with self._stats_lock:
                self._stats["successful"] += 1
            return create_response("chatterbox", True, {
                "audio_base64": output_b64,
                "voice_used": "chatterbox_default",
                "inference_time": inference_time,
                "text_length": len(text)
            })
        except Exception as e:
            with self._stats_lock:
                self._stats["failed"] += 1
            logger.error(f"Speech generation failed: {e}", exc_info=True)
            return create_response("chatterbox", False, error=f"Failed to generate speech: {str(e)}")

    def _apply_emotion_tags(self, text: str, emotions: List[Dict]) -> str:
        if not emotions:
            return text
        insertions = []
        for emotion in emotions:
            etype = emotion.get("type", "")
            if etype in PARALINGUISTIC_TAGS:
                position = emotion.get("position", len(text))
                insertions.append((position, etype))
        result = text
        for position, tag in sorted(insertions, key=lambda x: x[0], reverse=True):
            result = result[:position] + f" {tag} " + result[position:]
        return result.strip()

    def _save_temp_audio(self, audio_b64: str) -> str:
        audio_bytes = base64.b64decode(audio_b64)
        audio_seg = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
        audio_seg = audio_seg.set_channels(1).set_frame_rate(22050)
        temp_path = tempfile.mktemp(suffix=".wav", prefix="chatterbox_ref_")
        audio_seg.export(temp_path, format="wav")
        return temp_path

    def _tensor_to_base64(self, wav_tensor) -> str:
        import torchaudio
        buffer = BytesIO()
        sample_rate = self._model.sr if hasattr(self._model, 'sr') else 24000
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        torchaudio.save(buffer, wav_tensor.cpu(), sample_rate, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
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
