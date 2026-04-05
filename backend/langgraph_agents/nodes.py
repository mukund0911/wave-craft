"""
LangGraph nodes for WaveCraft agent workflows.
Each node is a pure function: state -> state_updates
"""
import os
import base64
import wave
import tempfile
import logging
import threading
from io import BytesIO
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydub import AudioSegment

from .state import TranscriptionState, ModificationState, ArtificialSpeakerState
from .agents.whisperx import WhisperXAgent
from .agents.chatterbox import ChatterboxAgent
from .agents.dialogue import DialogueGeneratorAgent
from .agents.tts import TextToSpeechAgent

from ..utils.audio_quality import score_audio_quality

logger = logging.getLogger(__name__)

# Shared agent instances (lazy singletons per worker)
_whisperx_agent = None
_chatterbox_agent = None
_dialogue_agent = None
_tts_agent = None
_lock = threading.Lock()


def _get_whisperx_agent() -> WhisperXAgent:
    global _whisperx_agent
    if _whisperx_agent is None:
        with _lock:
            if _whisperx_agent is None:
                _whisperx_agent = WhisperXAgent()
    return _whisperx_agent


def _get_chatterbox_agent() -> ChatterboxAgent:
    global _chatterbox_agent
    if _chatterbox_agent is None:
        with _lock:
            if _chatterbox_agent is None:
                _chatterbox_agent = ChatterboxAgent()
    return _chatterbox_agent


def _get_dialogue_agent() -> DialogueGeneratorAgent:
    global _dialogue_agent
    if _dialogue_agent is None:
        with _lock:
            if _dialogue_agent is None:
                _dialogue_agent = DialogueGeneratorAgent()
    return _dialogue_agent


def _get_tts_agent() -> TextToSpeechAgent:
    global _tts_agent
    if _tts_agent is None:
        with _lock:
            if _tts_agent is None:
                _tts_agent = TextToSpeechAgent()
    return _tts_agent


# ──────────────────────────────────────────────────
# Transcription Graph Nodes
# ──────────────────────────────────────────────────

def optimize_audio_node(state: TranscriptionState) -> TranscriptionState:
    """Optimize audio for WhisperX (mono, 16kHz). Keeps original if already optimal."""
    audio_path = state.get("audio_path", "")
    if not audio_path or not audio_path.lower().endswith(".wav"):
        return state
    try:
        with wave.open(audio_path, 'rb') as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if channels == 1 and framerate == 16000:
            return state
        import audioop
        if channels == 2:
            frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        if framerate != 16000:
            frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 16000, None)
        optimized_path = tempfile.mktemp(suffix=".wav", prefix="optimized_")
        with wave.open(optimized_path, 'wb') as out:
            out.setnchannels(1)
            out.setsampwidth(sampwidth)
            out.setframerate(16000)
            out.writeframes(frames)
        logger.info(f"Audio optimized: {audio_path} -> {optimized_path}")
        return {**state, "audio_path": optimized_path, "original_audio_path": audio_path}
    except Exception as e:
        logger.warning(f"Audio optimization failed, using original: {e}")
        return state


def transcribe_node(state: TranscriptionState) -> TranscriptionState:
    """Run WhisperX transcription with speaker diarization."""
    agent = _get_whisperx_agent()
    result = agent.transcribe_file(
        state["audio_path"],
        num_speakers=state.get("num_speakers")
    )
    if result.get("status") == "error":
        return {**state, "error": result.get("error", "Transcription failed")}
    return {
        **state,
        "conversations": result.get("conversations", []),
        "full_audio": result.get("full_audio", ""),
        "language": result.get("language", "en"),
    }


# ──────────────────────────────────────────────────
# Modification Graph Nodes
# ──────────────────────────────────────────────────

def parse_changes_node(state: ModificationState) -> ModificationState:
    """Compare original vs modified text; mark segments for synthesis."""
    conversations = state.get("conversations", [])
    segments = []
    for conv_item in conversations:
        if not isinstance(conv_item, dict):
            continue
        for key, conv in conv_item.items():
            original = conv.get("original", {})
            modified = conv.get("modified", {})
            original_text = original.get("text", "")
            modified_text = modified.get("text", "")
            emotions = modified.get("emotions", [])
            exaggeration = modified.get("exaggeration", 0.5)
            text_changed = original_text.strip() != modified_text.strip()
            has_emotions = len(emotions) > 0
            segments.append({
                "key": key,
                "speaker": conv.get("speaker", "A"),
                "original_text": original_text,
                "modified_text": modified_text,
                "emotions": emotions,
                "exaggeration": exaggeration,
                "reference_audio_b64": original.get("speaker_audio", ""),
                "needs_synthesis": text_changed or has_emotions,
            })
    return {**state, "parsed_segments": segments, "segments": segments, "retry_params": {}}


def _process_single_segment(seg: Dict[str, Any], retry_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Worker for parallel synthesis."""
    retry_params = retry_params or {}
    if not seg["needs_synthesis"]:
        return {
            "key": seg["key"],
            "audio_base64": seg["reference_audio_b64"],
            "changed": False,
            "speaker": seg["speaker"],
            "method": "passthrough",
        }
    if not seg["modified_text"].strip():
        # All words deleted -> silence
        silence = _generate_silence(duration_ms=100)
        return {
            "key": seg["key"],
            "audio_base64": silence,
            "changed": True,
            "speaker": seg["speaker"],
            "method": "silence",
        }
    # Guard against empty reference audio
    if not seg.get("reference_audio_b64", "").strip():
        logger.warning(f"Speaker {seg['speaker']}: empty reference audio, using silence fallback")
        silence = _generate_silence(duration_ms=100)
        return {
            "key": seg["key"],
            "audio_base64": silence,
            "changed": True,
            "speaker": seg["speaker"],
            "method": "silence",
            "error": "empty_reference_audio",
        }
    # Apply retry params if present (reduce exaggeration for conservative retry)
    exaggeration = retry_params.get("exaggeration", seg["exaggeration"])
    agent = _get_chatterbox_agent()
    result = agent.modify_speech({
        "reference_audio_b64": seg["reference_audio_b64"],
        "original_text": seg["original_text"],
        "modified_text": seg["modified_text"],
        "emotions": seg["emotions"],
        "exaggeration": exaggeration,
    })
    if result.get("success") and result.get("data"):
        data = result["data"]
        return {
            "key": seg["key"],
            "audio_base64": data.get("audio_base64", seg["reference_audio_b64"]),
            "changed": data.get("changed", False),
            "speaker": seg["speaker"],
            "method": data.get("method", "unknown"),
            "error": data.get("error"),
        }
    else:
        err = result.get("error", "unknown")
        logger.warning(f"Speaker {seg['speaker']}: Chatterbox failed ({err}), using original")
        return {
            "key": seg["key"],
            "audio_base64": seg["reference_audio_b64"],
            "changed": False,
            "speaker": seg["speaker"],
            "method": "fallback",
            "error": err,
        }


def synthesize_segments_node(state: ModificationState) -> ModificationState:
    """Parallel synthesis of changed segments. Unchanged segments passthrough."""
    # Always read from parsed_segments so retries get original unprocessed data
    raw_segments = state.get("parsed_segments", [])
    if not raw_segments:
        return {**state, "error": "No segments to process"}

    max_workers = min(len(raw_segments), 3)
    processed = [None] * len(raw_segments)

    retry_params = state.get("retry_params", {})
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_segment, seg, retry_params): idx
            for idx, seg in enumerate(raw_segments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            processed[idx] = future.result()

    return {**state, "segments": processed}


def assemble_audio_node(state: ModificationState) -> ModificationState:
    """Assemble processed segments into final WAV using stdlib wave + audioop."""
    segments = state.get("segments", [])
    if not segments:
        return {**state, "error": "No segments to assemble"}

    pcm_chunks = []
    output_params = None

    for i, segment in enumerate(segments):
        audio_b64 = segment.get("audio_base64", "")
        if not audio_b64:
            continue
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with wave.open(BytesIO(audio_bytes), 'rb') as wf:
                params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
                frames = wf.readframes(wf.getnframes())
                if output_params is None:
                    output_params = params
                if params != output_params:
                    frames = _convert_wav_frames(frames, params, output_params)
                pcm_chunks.append(frames)
        except Exception as e:
            logger.warning(f"Failed to process segment {i}: {e}")
            continue

    if not pcm_chunks or output_params is None:
        return {**state, "error": "No valid audio segments to assemble"}

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setnchannels(output_params[0])
        out.setsampwidth(output_params[1])
        out.setframerate(output_params[2])
        for chunk in pcm_chunks:
            out.writeframes(chunk)

    final_audio = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {**state, "final_audio": final_audio}


def quality_check_node(state: ModificationState) -> ModificationState:
    """Validate assembled audio. If silent/clipped, flag for retry (once)."""
    final_audio = state.get("final_audio", "")
    retry_count = state.get("retry_count", 0)
    if not final_audio:
        return state
    try:
        audio_bytes = base64.b64decode(final_audio)
        audio_seg = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
        score = score_audio_quality(audio_seg)
        total = score.get("total", 0)
        energy = score.get("energy", 0)
        clipping = score.get("clipping", 1)

        logger.info(f"Assembled audio quality score: {total:.3f} (energy={energy:.3f}, clipping={clipping:.3f})")

        # If audio is essentially silent or heavily clipped, and we haven't retried
        if (energy < 0.02 or clipping < 0.90) and retry_count < 1:
            logger.warning("Assembled audio failed quality check — will retry with relaxed params")
            # Reduce exaggeration for a more conservative synthesis on retry
            retry_params = {"exaggeration": max(0.3, 0.5)}
            return {
                **state,
                "retry_count": retry_count + 1,
                "needs_retry": True,
                "retry_params": retry_params,
            }

        return {**state, "needs_retry": False, "quality_score": score}
    except Exception as e:
        logger.warning(f"Quality check failed (non-fatal): {e}")
        return {**state, "needs_retry": False}


def build_stats_node(state: ModificationState) -> ModificationState:
    """Compute per-job segment stats and collect failures."""
    segments = state.get("segments", [])
    failures = []
    failed_count = 0
    changed_count = 0
    total_inference_time = 0.0
    for idx, seg in enumerate(segments):
        if seg.get("method") == "fallback":
            failed_count += 1
            failures.append({
                "index": idx,
                "speaker": seg.get("speaker"),
                "error": seg.get("error", "unknown"),
            })
        if seg.get("changed"):
            changed_count += 1
        total_inference_time += seg.get("inference_time", 0.0)

    successful = changed_count - failed_count
    stats = {
        "total_requests": len(segments),
        "successful": successful,
        "failed": failed_count,
        "total_inference_time": total_inference_time,
        "success_rate": (successful / changed_count * 100) if changed_count > 0 else 0,
        "avg_inference_time": (total_inference_time / successful) if successful > 0 else 0,
    }
    return {
        **state,
        "segments_processed": len(segments),
        "segments_changed": changed_count,
        "segments_failed": failed_count,
        "failures": failures[:20],
        "stats": stats,
    }


def s3_upload_node(state: ModificationState) -> ModificationState:
    """Optional S3 upload for final audio."""
    if os.environ.get("S3_ENABLED", "false").lower() != "true":
        return state
    final_audio = state.get("final_audio", "")
    if not final_audio:
        return state
    try:
        from ..utils.s3_storage import S3AudioStorage
        storage = S3AudioStorage()
        audio_bytes = base64.b64decode(final_audio)
        url = storage.upload_and_get_url(audio_bytes)
        if url:
            logger.info(f"Uploaded to S3: {url}")
            return {**state, "s3_url": url}
    except Exception as e:
        logger.warning(f"S3 upload failed (using base64 fallback): {e}")
    return state


# ──────────────────────────────────────────────────
# Artificial Speaker Graph Nodes
# ──────────────────────────────────────────────────

def generate_dialogue_node(state: ArtificialSpeakerState) -> ArtificialSpeakerState:
    """Generate dialogue via GPT."""
    agent = _get_dialogue_agent()
    result = agent.generate_dialogue(
        prompt=state.get("prompt", ""),
        conversation_history=state.get("conversation_history", []),
        speaker_characteristics=state.get("speaker_characteristics", "")
    )
    if not result.get("success"):
        return {**state, "error": result.get("error", "Dialogue generation failed")}
    return {**state, "generated_dialogue": result["data"]["dialogue"]}


def select_tts_engine_node(state: ArtificialSpeakerState) -> ArtificialSpeakerState:
    """Choose TTS engine (OpenAI for speed, Chatterbox for quality)."""
    use_openai = state.get("use_openai_tts", False)
    engine = "openai" if use_openai else "chatterbox"
    logger.info(f"Selected TTS engine: {engine}")
    return {**state, "tts_engine": engine}


def synthesize_speech_node(state: ArtificialSpeakerState) -> ArtificialSpeakerState:
    """Execute chosen TTS engine."""
    text = state.get("generated_dialogue", "")
    engine = state.get("tts_engine", "chatterbox")
    exaggeration = state.get("exaggeration", 0.5)
    characteristics = state.get("speaker_characteristics", "")

    if not text:
        return {**state, "error": "No dialogue text to synthesize"}

    if engine == "openai":
        agent = _get_tts_agent()
        result = agent.generate_speech(text, speaker_characteristics=characteristics)
        if not result.get("success"):
            # Fallback to Chatterbox if OpenAI fails
            logger.warning("OpenAI TTS failed, falling back to Chatterbox")
            agent = _get_chatterbox_agent()
            result = agent.generate_speech(text, exaggeration=exaggeration)
    else:
        agent = _get_chatterbox_agent()
        result = agent.generate_speech(text, exaggeration=exaggeration)

    if not result.get("success"):
        return {**state, "error": result.get("error", "TTS failed")}

    data = result.get("data", {})
    return {
        **state,
        "generated_audio": data.get("audio_base64", ""),
        "voice_used": data.get("voice_used", engine),
    }


# ──────────────────────────────────────────────────
# Error handler (shared)
# ──────────────────────────────────────────────────

def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Format error into state. Idempotent — preserves existing error."""
    error = state.get("error")
    if error:
        logger.error(f"Graph error: {error}")
    return state


# ──────────────────────────────────────────────────
# Conditional edge helpers
# ──────────────────────────────────────────────────

def has_error(state: Dict[str, Any]) -> str:
    return "error" if state.get("error") else "continue"


def needs_retry(state: ModificationState) -> str:
    if state.get("needs_retry") and state.get("retry_count", 0) <= 1:
        return "retry"
    return "continue"


# ──────────────────────────────────────────────────
# Internal utilities
# ──────────────────────────────────────────────────

def _generate_silence(duration_ms: int = 100, sample_rate: int = 24000) -> str:
    num_samples = int(sample_rate * duration_ms / 1000)
    silence_frames = b'\x00\x00' * num_samples
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence_frames)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _convert_wav_frames(frames: bytes, src_params: tuple, dst_params: tuple) -> bytes:
    src_ch, src_sw, src_fr = src_params
    dst_ch, dst_sw, dst_fr = dst_params
    if src_sw != dst_sw:
        import audioop
        frames = audioop.lin2lin(frames, src_sw, dst_sw)
        src_sw = dst_sw
    if src_ch != dst_ch:
        import audioop
        if src_ch == 1 and dst_ch == 2:
            frames = audioop.tostereo(frames, dst_sw, 1, 1)
        elif src_ch == 2 and dst_ch == 1:
            frames = audioop.tomono(frames, dst_sw, 0.5, 0.5)
    if src_fr != dst_fr:
        import audioop
        frames, _ = audioop.ratecv(frames, dst_sw, dst_ch, src_fr, dst_fr, None)
    return frames
