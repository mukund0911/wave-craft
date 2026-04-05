"""
WaveCrafter Backend Routes (LangGraph refactor)

API Endpoints:
  POST /upload                 - Upload audio file for transcription
  GET  /status/<job_id>        - Check transcription status
  POST /conversations_modified - Process modified transcripts -> generate audio
  POST /artificial_speaker     - Generate AI speaker dialogue + voice
  GET  /generation_status/<job_id> - Poll generation progress
  GET  /health                 - Health check
"""

import os
import json
import uuid
import hashlib
import logging
import base64
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from flask import Blueprint, request, jsonify, current_app

logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)


# ──────────────────────────────────────────────────
# TTL Cache
# ──────────────────────────────────────────────────

class TTLCache:
    """Simple thread-safe TTL cache with max size eviction."""

    def __init__(self, max_size=20, ttl_seconds=1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return default
            value, ts = entry
            if time.time() - ts > self.ttl_seconds:
                del self._store[key]
                return default
            return value

    def set(self, key, value):
        with self._lock:
            self._store[key] = (value, time.time())
            self._evict()

    def __contains__(self, key):
        return self.get(key) is not None

    def __getitem__(self, key):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        self.set(key, value)

    def _evict(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self._store[k]
        while len(self._store) > self.max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]


audio_cache = TTLCache(max_size=20, ttl_seconds=1800)
job_store = TTLCache(max_size=100, ttl_seconds=3600)
job_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=3)


# ──────────────────────────────────────────────────
# Graph accessors (lazy singleton per app context)
# ──────────────────────────────────────────────────

def get_transcription_graph():
    from .langgraph_agents.graphs import build_transcription_graph
    if not hasattr(current_app, '_transcription_graph'):
        current_app._transcription_graph = build_transcription_graph()
    return current_app._transcription_graph


def get_modification_graph():
    from .langgraph_agents.graphs import build_modification_graph
    if not hasattr(current_app, '_modification_graph'):
        current_app._modification_graph = build_modification_graph()
    return current_app._modification_graph


def get_artificial_speaker_graph():
    from .langgraph_agents.graphs import build_artificial_speaker_graph
    if not hasattr(current_app, '_artificial_speaker_graph'):
        current_app._artificial_speaker_graph = build_artificial_speaker_graph()
    return current_app._artificial_speaker_graph


# ──────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────

def file_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def save_upload(file) -> str:
    from werkzeug.utils import secure_filename
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    safe_name = secure_filename(file.filename) or 'upload.wav'
    name, ext = os.path.splitext(safe_name)
    unique_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(upload_dir, unique_name)
    file.save(filepath)
    return filepath


# ──────────────────────────────────────────────────
# Upload endpoint
# ──────────────────────────────────────────────────

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        filepath = save_upload(file)
        logger.info(f"Upload received: {file.filename}")

        num_speakers = request.form.get('num_speakers')
        if num_speakers:
            try:
                num_speakers = int(num_speakers)
                if num_speakers < 1 or num_speakers > 20:
                    num_speakers = None
            except (ValueError, TypeError):
                num_speakers = None

        fhash = file_hash(filepath)
        if fhash in audio_cache:
            logger.info(f"Cache hit: {fhash}")
            cached = audio_cache[fhash]
            try:
                os.remove(filepath)
            except OSError:
                pass
            return jsonify({
                "status": "completed",
                "conversations": cached["conversations"],
                "full_audio": cached["full_audio"],
                "cached": True
            }), 200

        job_id = str(uuid.uuid4())
        with job_lock:
            job_store[job_id] = {
                "status": "processing",
                "result": None,
                "error": None,
                "filepath": filepath,
                "fhash": fhash
            }

        app = current_app._get_current_object()
        _executor.submit(_run_transcription, app, job_id, filepath, fhash, num_speakers)

        return jsonify({
            "status": "processing",
            "job_id": job_id,
            "message": "Transcription started"
        }), 202

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_transcription(app, job_id: str, filepath: str, fhash: str, num_speakers: int = None):
    """Background transcription task using LangGraph."""
    optimized_path = None
    try:
        with app.app_context():
            graph = get_transcription_graph()
            result = graph.invoke({
                "audio_path": filepath,
                "num_speakers": num_speakers,
            })

            optimized_path = result.get("audio_path")

            error = result.get("error")
            if error:
                with job_lock:
                    job_store[job_id] = {
                        "status": "error",
                        "result": None,
                        "error": error,
                        "filepath": filepath,
                        "fhash": fhash
                    }
                return

            conversations = result.get("conversations", [])
            full_audio = result.get("full_audio", "")
            language = result.get("language", "en")

            with job_lock:
                job_store[job_id] = {
                    "status": "completed",
                    "result": {
                        "conversations": conversations,
                        "full_audio": full_audio,
                        "language": language,
                    },
                    "error": None,
                    "filepath": filepath,
                    "fhash": fhash
                }
                audio_cache[fhash] = {
                    "conversations": conversations,
                    "full_audio": full_audio
                }

    except Exception as e:
        logger.error(f"Background transcription failed: {e}", exc_info=True)
        with job_lock:
            job_store[job_id] = {
                "status": "error",
                "result": None,
                "error": str(e),
                "filepath": filepath,
                "fhash": fhash
            }
    finally:
        try:
            # Clean up optimized temp file if it was created and differs from original
            if optimized_path and optimized_path != filepath and os.path.exists(optimized_path):
                os.remove(optimized_path)
                logger.info(f"Cleaned up optimized audio: {optimized_path}")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up upload: {filepath}")
        except OSError as e:
            logger.warning(f"Failed to clean up upload {filepath}: {e}")


# ──────────────────────────────────────────────────
# Status polling endpoint
# ──────────────────────────────────────────────────

@main.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    with job_lock:
        job = job_store.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        status = job["status"]
        result = job.get("result")
        error = job.get("error")

    if status == "completed" and result:
        return jsonify({
            "status": "completed",
            "conversations": result["conversations"],
            "full_audio": result["full_audio"],
            "language": result.get("language", "en"),
        }), 200
    elif status == "error":
        return jsonify({
            "status": "error",
            "error": error or "Unknown error"
        }), 500
    else:
        return jsonify({
            "status": "processing",
            "message": "Transcription in progress..."
        }), 200


# ──────────────────────────────────────────────────
# Modified transcript -> audio generation
# ──────────────────────────────────────────────────

MAX_SEGMENTS = 2000


@main.route('/conversations_modified', methods=['POST'])
def modified_transcript():
    try:
        if request.content_type and 'multipart' in request.content_type:
            conversations_str = request.form.get('conversations_updated', '[]')
            full_audio = request.form.get('full_audio', '')
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid or missing JSON body"}), 400
            conversations_str = json.dumps(data.get('conversations_updated', []))
            full_audio = data.get('full_audio', '')

        conversations = json.loads(conversations_str) if isinstance(conversations_str, str) else conversations_str

        if not conversations:
            return jsonify({"error": "No conversations provided"}), 400
        if not isinstance(conversations, list):
            return jsonify({"error": "conversations must be a list"}), 400
        if len(conversations) > MAX_SEGMENTS:
            return jsonify({"error": f"Too many segments (max {MAX_SEGMENTS})"}), 400

        logger.info(f"Queued {len(conversations)} modified conversations for generation")

        job_id = str(uuid.uuid4())
        with job_lock:
            job_store[job_id] = {
                "kind": "generation",
                "status": "processing",
                "result": None,
                "error": None,
            }

        app = current_app._get_current_object()
        _executor.submit(_run_generation, app, job_id, conversations, full_audio)

        return jsonify({
            "status": "processing",
            "job_id": job_id,
            "message": f"Generation started for {len(conversations)} segments"
        }), 202

    except Exception as e:
        logger.error(f"Modified transcript endpoint failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_generation(app, job_id: str, conversations: list, full_audio: str):
    """Background generation task using LangGraph."""
    try:
        with app.app_context():
            graph = get_modification_graph()
            result = graph.invoke({
                "conversations": conversations,
                "full_audio": full_audio,
                "retry_count": 0,
            })

            error = result.get("error")
            if error:
                with job_lock:
                    job_store[job_id] = {
                        "kind": "generation",
                        "status": "error",
                        "result": None,
                        "error": error,
                    }
                return

            response_data = {
                "modified_audio": result.get("final_audio", ""),
                "segments_processed": result.get("segments_processed", 0),
                "segments_changed": result.get("segments_changed", 0),
                "segments_failed": result.get("segments_failed", 0),
                "failures": result.get("failures", []),
                "stats": result.get("stats", {}),
            }

            if result.get("s3_url"):
                response_data["audio_url"] = result["s3_url"]

            with job_lock:
                job_store[job_id] = {
                    "kind": "generation",
                    "status": "completed",
                    "result": response_data,
                    "error": None,
                }

    except Exception as e:
        logger.error(f"Background generation failed: {e}", exc_info=True)
        with job_lock:
            job_store[job_id] = {
                "kind": "generation",
                "status": "error",
                "result": None,
                "error": str(e),
            }


@main.route('/generation_status/<job_id>', methods=['GET'])
def generation_status(job_id):
    with job_lock:
        job = job_store.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        status = job["status"]
        result = job.get("result")
        error = job.get("error")

    if status == "completed" and result:
        return jsonify({"status": "completed", **result}), 200
    if status == "error":
        return jsonify({"status": "error", "error": error or "Unknown error"}), 500
    return jsonify({"status": "processing", "message": "Generation in progress..."}), 200


# ──────────────────────────────────────────────────
# Artificial Speaker endpoint
# ──────────────────────────────────────────────────

@main.route('/artificial_speaker', methods=['POST'])
def artificial_speaker():
    """
    Queue an AI speaker generation job.

    Request JSON:
        {
            "prompt": "Explain quantum computing like I'm five",
            "speaker_characteristics": "warm female voice, calm",
            "speaker_name": "AI",
            "exaggeration": 0.5,
            "use_openai_tts": false,
            "conversation_history": []
        }

    Returns 202:
        {
            "status": "processing",
            "job_id": "...",
            "message": "Artificial speaker generation started"
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        job_id = str(uuid.uuid4())
        with job_lock:
            job_store[job_id] = {
                "kind": "artificial_speaker",
                "status": "processing",
                "result": None,
                "error": None,
            }

        app = current_app._get_current_object()
        _executor.submit(
            _run_artificial_speaker,
            app, job_id, prompt,
            data.get("speaker_characteristics", ""),
            data.get("speaker_name", "AI"),
            data.get("exaggeration", 0.5),
            data.get("use_openai_tts", False),
            data.get("conversation_history", []),
        )

        return jsonify({
            "status": "processing",
            "job_id": job_id,
            "message": "Artificial speaker generation started"
        }), 202

    except Exception as e:
        logger.error(f"Artificial speaker endpoint failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_artificial_speaker(app, job_id, prompt, speaker_characteristics, speaker_name, exaggeration, use_openai_tts, conversation_history):
    """Background artificial speaker generation task using LangGraph."""
    try:
        with app.app_context():
            graph = get_artificial_speaker_graph()
            result = graph.invoke({
                "prompt": prompt,
                "speaker_characteristics": speaker_characteristics,
                "speaker_name": speaker_name,
                "exaggeration": exaggeration,
                "use_openai_tts": use_openai_tts,
                "conversation_history": conversation_history,
            })

            error = result.get("error")
            if error:
                with job_lock:
                    job_store[job_id] = {
                        "kind": "artificial_speaker",
                        "status": "error",
                        "result": None,
                        "error": error,
                    }
                return

            with job_lock:
                job_store[job_id] = {
                    "kind": "artificial_speaker",
                    "status": "completed",
                    "result": {
                        "dialogue": result.get("generated_dialogue", ""),
                        "audio": result.get("generated_audio", ""),
                        "voice_used": result.get("voice_used", "unknown"),
                    },
                    "error": None,
                }

    except Exception as e:
        logger.error(f"Background artificial speaker generation failed: {e}", exc_info=True)
        with job_lock:
            job_store[job_id] = {
                "kind": "artificial_speaker",
                "status": "error",
                "result": None,
                "error": str(e),
            }


@main.route('/artificial_speaker_status/<job_id>', methods=['GET'])
def artificial_speaker_status(job_id):
    """Poll artificial speaker generation job status."""
    with job_lock:
        job = job_store.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        status = job["status"]
        result = job.get("result")
        error = job.get("error")

    if status == "completed" and result:
        return jsonify({"status": "completed", **result}), 200
    if status == "error":
        return jsonify({"status": "error", "error": error or "Unknown error"}), 500
    return jsonify({"status": "processing", "message": "Generation in progress..."}), 200


# ──────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────

@main.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "service": "wavecraft-api",
        "engine": "langgraph+whisperx+chatterbox"
    }), 200
