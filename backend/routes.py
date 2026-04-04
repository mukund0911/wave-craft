"""
WaveCrafter Backend Routes (Revamped)

API Endpoints:
  POST /upload          - Upload audio file for transcription
  GET  /status/<job_id> - Check transcription status (kept for compatibility)
  POST /conversations_modified - Process modified transcripts → generate audio

Removed:
  - Speech/music classification (all uploads are speech now)
  - AssemblyAI (replaced by WhisperX)
  - VoiceCraft/Modal (replaced by Chatterbox)
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
        self._store = {}  # key -> (value, timestamp)
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
        """Remove expired entries, then oldest if over capacity."""
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self._store[k]
        while len(self._store) > self.max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]


# In-memory caches with TTL
audio_cache = TTLCache(max_size=20, ttl_seconds=1800)
job_store = TTLCache(max_size=100, ttl_seconds=3600)

# Lock for thread-safe job_store access
job_lock = threading.Lock()

# Thread pool for background transcription
_executor = ThreadPoolExecutor(max_workers=3)


def get_speech_agent():
    """Get speech processing agent from app context"""
    from .mcp_agents.speech_processing_agent import SpeechProcessingAgent
    if not hasattr(current_app, '_speech_agent'):
        current_app._speech_agent = SpeechProcessingAgent()
    return current_app._speech_agent


def file_hash(filepath: str) -> str:
    """Calculate MD5 hash for cache key"""
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def save_upload(file) -> str:
    """Save uploaded file to temp directory with UUID suffix to prevent collisions, return path"""
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
    """
    Upload audio file for transcription.

    WhisperX is fast enough to run synchronously for small files.
    For larger files we still use async with polling.

    Returns:
        - If fast mode (< 30s audio): immediate transcription result
        - If async mode: job_id for polling via /status/<job_id>
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save upload
        filepath = save_upload(file)
        logger.info(f"Upload received: {file.filename}")

        # Parse optional num_speakers
        num_speakers = request.form.get('num_speakers')
        if num_speakers:
            try:
                num_speakers = int(num_speakers)
                if num_speakers < 1 or num_speakers > 20:
                    num_speakers = None
            except (ValueError, TypeError):
                num_speakers = None

        # Check cache
        fhash = file_hash(filepath)
        if fhash in audio_cache:
            logger.info(f"Cache hit: {fhash}")
            cached = audio_cache[fhash]
            # Clean up the uploaded file since we have a cache hit
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

        # Start async transcription in background thread
        job_id = str(uuid.uuid4())

        with job_lock:
            job_store[job_id] = {
                "status": "processing",
                "result": None,
                "error": None,
                "filepath": filepath,
                "fhash": fhash
            }

        # Run transcription in thread pool
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
    """Background transcription task"""
    try:
        with app.app_context():
            speech_agent = get_speech_agent()
            result = speech_agent.transcribe_audio(filepath, num_speakers=num_speakers)

            with job_lock:
                if result["status"] == "completed":
                    job_store[job_id] = {
                        "status": "completed",
                        "result": result,
                        "error": None,
                        "filepath": filepath,
                        "fhash": fhash
                    }

                    # Cache the result
                    audio_cache[fhash] = {
                        "conversations": result["conversations"],
                        "full_audio": result["full_audio"]
                    }
                else:
                    job_store[job_id] = {
                        "status": "error",
                        "result": None,
                        "error": result.get("error", "Unknown error"),
                        "filepath": filepath,
                        "fhash": fhash
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
        # Always clean up uploaded file
        try:
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
    """Check transcription job status"""
    with job_lock:
        job = job_store.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        # Copy status and relevant data while holding the lock
        status = job["status"]
        result = job.get("result")
        error = job.get("error")

    if status == "completed" and result:
        return jsonify({
            "status": "completed",
            "conversations": result["conversations"],
            "full_audio": result["full_audio"]
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
# Modified transcript → audio generation
# ──────────────────────────────────────────────────

MAX_SEGMENTS = 2000  # realistic upper bound for long-form podcasts


@main.route('/conversations_modified', methods=['POST'])
def modified_transcript():
    """
    Process modified transcripts and generate final audio (async).

    Returns job_id immediately; poll /generation_status/<job_id> for result.
    """
    try:
        # Parse request data (handle both multipart and JSON)
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

        # Start async generation
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
    """Background generation task."""
    try:
        with app.app_context():
            speech_agent = get_speech_agent()
            result = speech_agent.handle_modifications({
                "action": "process_modifications",
                "conversations": conversations,
                "full_audio": full_audio,
            })

            if not result.get("success"):
                with job_lock:
                    job_store[job_id] = {
                        "kind": "generation",
                        "status": "error",
                        "result": None,
                        "error": result.get("error", "Processing failed"),
                    }
                return

            result_data = result.get("data", {})
            response_data = {
                "modified_audio": result_data.get("modified_audio", ""),
                "segments_processed": result_data.get("segments_processed", 0),
                "segments_changed": result_data.get("segments_changed", 0),
                "segments_failed": result_data.get("segments_failed", 0),
                "failures": result_data.get("failures", []),
                "stats": result_data.get("stats", {}),
            }

            # Optional S3 upload
            if os.environ.get("S3_ENABLED", "false").lower() == "true":
                try:
                    from .utils.s3_storage import S3AudioStorage
                    storage = S3AudioStorage()
                    audio_bytes = base64.b64decode(response_data["modified_audio"])
                    url = storage.upload_and_get_url(audio_bytes)
                    if url:
                        response_data["audio_url"] = url
                        logger.info(f"Uploaded to S3: {url}")
                except Exception as s3_err:
                    logger.warning(f"S3 upload failed (using base64 fallback): {s3_err}")

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
    """Poll generation job status."""
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
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "wavecraft-api",
        "engine": "whisperx+chatterbox"
    }), 200
