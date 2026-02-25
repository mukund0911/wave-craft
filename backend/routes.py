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
import hashlib
import asyncio
import logging
import threading
import base64
from io import BytesIO
from flask import Blueprint, request, jsonify, current_app

logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# In-memory caches
audio_cache = {}
job_store = {}  # {job_id: {status, result, error, filepath}}

# Lock for thread-safe job_store access
job_lock = threading.Lock()


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
    """Save uploaded file to temp directory, return path"""
    from werkzeug.utils import secure_filename
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = secure_filename(file.filename) or 'upload.wav'
    filepath = os.path.join(upload_dir, safe_name)
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

        # Check cache
        fhash = file_hash(filepath)
        if fhash in audio_cache:
            logger.info(f"Cache hit: {fhash}")
            cached = audio_cache[fhash]
            return jsonify({
                "status": "completed",
                "conversations": cached["conversations"],
                "full_audio": cached["full_audio"],
                "cached": True
            }), 200

        # Start async transcription in background thread
        import uuid
        job_id = str(uuid.uuid4())

        with job_lock:
            job_store[job_id] = {
                "status": "processing",
                "result": None,
                "error": None,
                "filepath": filepath,
                "fhash": fhash
            }

        # Run transcription in background thread
        # Pass the actual Flask app so the thread can use its context
        app = current_app._get_current_object()
        thread = threading.Thread(
            target=_run_transcription,
            args=(app, job_id, filepath, fhash),
            daemon=True
        )
        thread.start()

        return jsonify({
            "status": "processing",
            "job_id": job_id,
            "message": "Transcription started"
        }), 202

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _run_transcription(app, job_id: str, filepath: str, fhash: str):
    """Background transcription task"""
    try:
        with app.app_context():
            speech_agent = get_speech_agent()
            result = speech_agent.transcribe_audio(filepath)

            with job_lock:
                if result["status"] == "completed":
                    job_store[job_id]["status"] = "completed"
                    job_store[job_id]["result"] = result

                    # Cache the result
                    audio_cache[fhash] = {
                        "conversations": result["conversations"],
                        "full_audio": result["full_audio"]
                    }
                else:
                    job_store[job_id]["status"] = "error"
                    job_store[job_id]["error"] = result.get("error", "Unknown error")

    except Exception as e:
        logger.error(f"Background transcription failed: {e}", exc_info=True)
        with job_lock:
            job_store[job_id]["status"] = "error"
            job_store[job_id]["error"] = str(e)


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

@main.route('/conversations_modified', methods=['POST'])
def modified_transcript():
    """
    Process modified transcripts and generate final audio.

    Expects multipart form data or JSON with:
    - conversations_updated: JSON list of modified conversations
    - full_audio: base64 of original audio (optional)
    - Each conversation can have emotion tags
    """
    try:
        # Parse request data (handle both multipart and JSON)
        if request.content_type and 'multipart' in request.content_type:
            conversations_str = request.form.get('conversations_updated', '[]')
            full_audio = request.form.get('full_audio', '')
        else:
            data = request.get_json()
            conversations_str = json.dumps(data.get('conversations_updated', []))
            full_audio = data.get('full_audio', '')

        conversations = json.loads(conversations_str) if isinstance(conversations_str, str) else conversations_str

        if not conversations:
            return jsonify({"error": "No conversations provided"}), 400

        logger.info(f"Processing {len(conversations)} modified conversations")

        # Use speech agent to process modifications
        speech_agent = get_speech_agent()

        request_data = {
            "action": "process_modifications",
            "conversations": conversations,
            "full_audio": full_audio
        }

        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(speech_agent.process_request(request_data))
        finally:
            loop.close()

        if not result.get("success"):
            error_msg = result.get("error", "Processing failed")
            logger.error(f"Modification processing failed: {error_msg}")
            return jsonify({"error": error_msg}), 500

        result_data = result.get("data", {})

        # Build response
        response_data = {
            "modified_audio": result_data.get("modified_audio", ""),
            "segments_processed": result_data.get("segments_processed", 0),
            "segments_changed": result_data.get("segments_changed", 0),
            "stats": result_data.get("stats", {})
        }

        # Try S3 upload if enabled
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

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Modified transcript endpoint failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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