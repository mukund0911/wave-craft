"""
WhisperX Speech-to-Text Agent (LangGraph refactor)
Supports remote (Modal GPU) or local inference.
"""
import os
import base64
import logging
import time
import requests as http_requests
from typing import Dict, Any, Optional
from io import BytesIO
from pydub import AudioSegment
from .base import create_response

logger = logging.getLogger(__name__)

MODAL_TRANSCRIBE_URL = os.environ.get("MODAL_TRANSCRIBE_URL", "")

# Lazy-loaded globals
_whisperx = None
_torch = None


def _patch_torch_load():
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


class WhisperXAgent:
    """Local or remote speech-to-text with speaker diarization."""

    def __init__(self, hf_token: str = "", model_size: str = "large-v2",
                 compute_type: str = "float16", device: str = None):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.model_size = model_size
        self.compute_type = compute_type

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
                self.compute_type = "int8"

        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_pipeline = None
        self._align_cache: Dict[str, Any] = {}

        logger.info(f"WhisperXAgent: model={model_size}, device={self.device}, compute={self.compute_type}")

    def _load_model(self):
        if self._model is not None:
            return
        whisperx = _get_whisperx()
        logger.info(f"Loading WhisperX model: {self.model_size}...")
        self._model = whisperx.load_model(self.model_size, self.device, compute_type=self.compute_type)
        logger.info("WhisperX model loaded")

    def _load_align_model(self, language_code: str = "en"):
        if language_code in self._align_cache:
            return self._align_cache[language_code]
        whisperx = _get_whisperx()
        try:
            model, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            self._align_cache[language_code] = (model, metadata)
            logger.info(f"Alignment model loaded ({language_code})")
            return model, metadata
        except Exception as e:
            logger.warning(f"No alignment model for '{language_code}', falling back to English: {e}")
            return self._align_cache.get("en")

    def _load_diarize_pipeline(self):
        if self._diarize_pipeline is not None:
            return
        if not self.hf_token:
            logger.warning("No HF_TOKEN set — diarization disabled.")
            return
        from whisperx.diarize import DiarizationPipeline
        self._diarize_pipeline = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-community-1",
            token=self.hf_token,
            device=self.device
        )
        logger.info("Diarization pipeline loaded")

    def transcribe_file(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        if MODAL_TRANSCRIBE_URL:
            return self._transcribe_remote(audio_path, num_speakers)
        return self._transcribe_local(audio_path, num_speakers)

    def _transcribe_remote(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            payload = {"audio": audio_b64}
            if num_speakers:
                payload["num_speakers"] = num_speakers
            response = _request_with_retry(MODAL_TRANSCRIBE_URL, payload, timeout=1800)
            return response.json()
        except Exception as e:
            logger.error(f"Modal transcription failed: {e}", exc_info=True)
            return {"status": "error", "error": f"Modal service error: {str(e)}"}

    def _transcribe_local(self, audio_path: str, num_speakers: int = None) -> Dict[str, Any]:
        whisperx = _get_whisperx()
        self._load_model()
        try:
            audio = whisperx.load_audio(audio_path)
            result = self._model.transcribe(audio, batch_size=16)
            detected_language = result.get("language", "en")

            self._load_align_model(detected_language)
            align_entry = self._align_cache.get(detected_language) or self._align_cache.get("en")
            if align_entry:
                result = whisperx.align(
                    result["segments"], align_entry[0], align_entry[1],
                    audio, self.device, return_char_alignments=False
                )

            self._load_diarize_pipeline()
            if self._diarize_pipeline is not None:
                diarize_args = {"return_embeddings": True}
                if num_speakers:
                    diarize_args["num_speakers"] = num_speakers
                else:
                    diarize_args["min_speakers"] = 2
                    diarize_args["max_speakers"] = 10
                diarize_result = self._diarize_pipeline(audio_path, **diarize_args)
                if isinstance(diarize_result, tuple):
                    diarize_segments, speaker_embeddings = diarize_result
                else:
                    diarize_segments = diarize_result
                    speaker_embeddings = None
                result = whisperx.assign_word_speakers(diarize_segments, result)
                if speaker_embeddings and not num_speakers:
                    result = self._merge_similar_speakers(result, speaker_embeddings)
            else:
                for segment in result["segments"]:
                    segment["speaker"] = "SPEAKER_00"

            conversations = self._build_conversations(result, audio_path)
            full_audio_b64 = self._audio_to_base64(audio_path)
            return {
                "status": "completed",
                "conversations": conversations,
                "full_audio": full_audio_b64,
                "language": detected_language
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _build_conversations(self, result: Dict, audio_path: str) -> list:
        full_audio = AudioSegment.from_file(audio_path)
        speaker_map = {}
        speaker_counter = 0
        conversations = []

        for segment in result.get("segments", []):
            speaker_id = segment.get("speaker", "SPEAKER_00")
            if speaker_id not in speaker_map:
                if speaker_counter < 26:
                    speaker_map[speaker_id] = chr(ord('A') + speaker_counter)
                else:
                    speaker_map[speaker_id] = f"Speaker {speaker_counter + 1}"
                speaker_counter += 1

            speaker_label = speaker_map[speaker_id]
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            segment_audio = full_audio[start_ms:end_ms]
            segment_b64 = self._segment_to_base64(segment_audio)

            words_with_timestamps = []
            for word_info in segment.get("words", []):
                words_with_timestamps.append({
                    "word": word_info.get("word", ""),
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "score": word_info.get("score", 1.0),
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
                        "words": words_with_timestamps,
                    },
                    "modified": {
                        "text": segment["text"].strip(),
                        "emotions": [],
                    },
                }
            })
        return conversations

    @staticmethod
    def _merge_similar_speakers(result, speaker_embeddings, threshold=0.65):
        import numpy as np
        from collections import defaultdict
        speakers = list(speaker_embeddings.keys())
        if len(speakers) <= 1:
            return result
        embeddings = {s: np.array(e) for s, e in speaker_embeddings.items()}
        parent = {s: s for s in speakers}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                s1, s2 = speakers[i], speakers[j]
                e1, e2 = embeddings[s1], embeddings[s2]
                norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if norm1 == 0 or norm2 == 0:
                    continue
                similarity = np.dot(e1, e2) / (norm1 * norm2)
                if similarity >= threshold:
                    union(s1, s2)

        groups = defaultdict(list)
        for s in speakers:
            groups[find(s)].append(s)

        merge_map = {}
        for members in groups.values():
            if len(members) <= 1:
                continue
            counts = {s: sum(1 for seg in result["segments"] if seg.get("speaker") == s) for s in members}
            canonical = max(members, key=lambda s: counts[s])
            for s in members:
                if s != canonical:
                    merge_map[s] = canonical

        if not merge_map:
            return result

        for segment in result["segments"]:
            speaker = segment.get("speaker", "")
            if speaker in merge_map:
                segment["speaker"] = merge_map[speaker]
            for word in segment.get("words", []):
                if word.get("speaker", "") in merge_map:
                    word["speaker"] = merge_map[word["speaker"]]
        return result

    def _audio_to_base64(self, audio_path: str) -> str:
        audio = AudioSegment.from_file(audio_path)
        buffer = BytesIO()
        audio.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _segment_to_base64(self, audio_segment: AudioSegment) -> str:
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
