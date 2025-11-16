# app/routes.py
import os
import json
import hashlib
from flask_socketio import SocketIO
from flask import Blueprint, request, jsonify

from backend.models.speech_music_classifier.sm_inference import sm_inference
from backend.models.speech_edit.speech_to_textv2 import SpeechModel

main = Blueprint('main', __name__)
socketio = SocketIO()

# Folder to store uploaded files
UPLOAD_FOLDER = './uploads'
SEPARATED_FOLDER = './separated'
CACHE_FOLDER = './cache'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEPARATED_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Simple in-memory store for job tracking (use Redis in production)
job_store = {}

def get_file_hash(filepath):
    """Generate hash of file for caching"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cached_result(file_hash):
    """Check if result exists in cache"""
    cache_path = os.path.join(CACHE_FOLDER, f"{file_hash}.json")
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

def save_to_cache(file_hash, result):
    """Save result to cache"""
    cache_path = os.path.join(CACHE_FOLDER, f"{file_hash}.json")
    with open(cache_path, 'w') as f:
        json.dump(result, f)

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Check cache first
        file_hash = get_file_hash(filepath)
        cached_result = get_cached_result(file_hash)

        if cached_result:
            return jsonify({
                "message": "File processed (from cache)",
                "cached": True,
                **cached_result
            }), 200

        # Predict whether the uploaded file is "speech" or "music"
        prediction = sm_inference(filepath)

        if prediction == "Speech":
            _speech_model = SpeechModel(filepath)

            # Submit async job
            print(f"[DEBUG] Submitting async transcription for: {filepath}")
            transcript_id = _speech_model.speech_to_text_async()
            print(f"[DEBUG] Got transcript_id: {transcript_id}")

            # Store job info
            job_id = f"{file_hash}_{transcript_id}"
            job_store[job_id] = {
                'filepath': filepath,
                'transcript_id': transcript_id,
                'file_hash': file_hash,
                'status': 'processing'
            }

            print(f"[DEBUG] Created job with id: {job_id}")
            print(f"[DEBUG] Job store now contains: {list(job_store.keys())}")

            return jsonify({
                "message": "File uploaded. Processing started.",
                "job_id": job_id,
                "prediction": prediction,
                "status": "processing"
            }), 202  # 202 = Accepted, processing
        else:
            return jsonify({
                "message": "Music separation not yet implemented",
                "prediction": prediction
            }), 501  # 501 = Not Implemented

@main.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Poll endpoint for checking transcription status"""
    try:
        print(f"[DEBUG] Checking status for job_id: {job_id}")
        print(f"[DEBUG] Current job_store keys: {list(job_store.keys())}")

        if job_id not in job_store:
            print(f"[ERROR] Job {job_id} not found in job_store")
            return jsonify({"error": "Job not found"}), 404

        job_info = job_store[job_id]
        filepath = job_info['filepath']
        transcript_id = job_info['transcript_id']
        file_hash = job_info['file_hash']

        print(f"[DEBUG] filepath: {filepath}, transcript_id: {transcript_id}")

        _speech_model = SpeechModel(filepath)
        result = _speech_model.get_transcript_status(transcript_id)

        print(f"[DEBUG] Result from get_transcript_status: {result}")

        if result['status'] == 'completed':
            # Cache the result
            cache_data = {
                'prediction': 'Speech',
                'full_audio': result['full_audio'],
                'conversations': result['conversations']
            }
            save_to_cache(file_hash, cache_data)

            # Update job store
            job_store[job_id]['status'] = 'completed'

            return jsonify({
                "status": "completed",
                "prediction": "Speech",
                "full_audio": result['full_audio'],
                "conversations": result['conversations']
            }), 200

        elif result['status'] == 'error':
            job_store[job_id]['status'] = 'error'
            return jsonify({
                "status": "error",
                "error": result.get('error', 'Unknown error')
            }), 500

        else:
            # Still processing
            return jsonify({
                "status": result['status'],  # 'queued' or 'processing'
                "message": f"Transcription {result['status']}..."
            }), 202

    except Exception as e:
        print(f"Error in check_status: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": f"Server error: {str(e)}"
        }), 500

@main.route('/conversations_modified', methods=['POST'])
def modified_transcript():
    """Generate final audio from modified conversations"""
    try:
        # Get the conversations JSON from the request
        conversation_mod = request.files.get('conversationsUpdated')
        full_audio_file = request.files.get('full_audio')

        if not conversation_mod:
            return jsonify({"error": "No conversations data provided"}), 400

        # Parse the conversations JSON
        import json
        import asyncio
        final_structure_data = conversation_mod.read().decode('utf-8')
        conversations_list = json.loads(final_structure_data)

        print(f"[DEBUG] Received {len(conversations_list)} conversations for processing")

        # Import the speech processing agent
        from backend.mcp_agents.speech_processing_agent import SpeechProcessingAgent
        from backend.models.config import ASSEMBLY_AI_KEY

        # Initialize the agent
        speech_agent = SpeechProcessingAgent(ASSEMBLY_AI_KEY)

        # Prepare request for final audio generation
        request_data = {
            "action": "generate_final_audio",
            "conversations": conversations_list
        }

        # Generate final audio (run async function in sync context)
        print("[DEBUG] Generating final audio...")
        result = asyncio.run(speech_agent.process_request(request_data))

        if not result["success"]:
            print(f"[ERROR] Failed to generate final audio: {result.get('error')}")
            return jsonify({
                "error": result.get('error', 'Failed to generate final audio')
            }), 500

        print("[DEBUG] Final audio generated successfully")

        # Optional: Upload to S3 and return presigned URL
        final_audio_base64 = result["data"]["final_audio_base64"]
        s3_url = None

        try:
            from backend.utils.s3_storage import S3AudioStorage
            import base64

            s3_storage = S3AudioStorage()

            if s3_storage.is_enabled():
                print("[DEBUG] Uploading final audio to S3...")

                # Decode base64 audio to bytes
                audio_bytes = base64.b64decode(final_audio_base64)

                # Upload and get presigned URL (valid for 1 hour)
                s3_url = s3_storage.upload_and_get_url(
                    audio_data=audio_bytes,
                    filename=None,  # Auto-generate filename
                    expiration=3600  # 1 hour
                )

                if s3_url:
                    print(f"[DEBUG] ✓ Audio uploaded to S3: {s3_url[:50]}...")
                else:
                    print("[WARNING] S3 upload failed, using base64 fallback")

        except Exception as e:
            print(f"[WARNING] S3 upload error: {e}, using base64 fallback")

        # Return the final audio data
        response_data = {
            "message": "Final audio generated successfully!",
            "modified_audio": final_audio_base64,  # Base64 for backward compatibility
            "duration_seconds": result["data"]["duration_seconds"],
            "sample_rate": result["data"]["sample_rate"],
            "segments_processed": result["data"]["segments_processed"],
            "segments_cloned": result["data"].get("segments_cloned", 0),
            "processing_time_seconds": result["data"].get("processing_time_seconds", 0)
        }

        # Add S3 URL if available
        if s3_url:
            response_data["audio_url"] = s3_url
            response_data["storage_method"] = "s3"
        else:
            response_data["storage_method"] = "base64"

        return jsonify(response_data), 200

    except Exception as e:
        print(f"[ERROR] Error in modified_transcript: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Failed to generate final audio: {str(e)}"
        }), 500
        

# def separate_audio(filepath):
#     """ This function separates the uploaded file into its components using Demucs. """
#     # Get model
#     model = get_model('htdemucs')  # Load Demucs model (htdemucs is a good general choice)

#     output_folder = Path(SEPARATED_FOLDER) / Path(filepath).stem
#     os.makedirs(output_folder, exist_ok=True)

#     # Use subprocess to safely call Demucs
#     command = f"demucs {filepath} -o {output_folder}"
#     subprocess.run(command.split(), check=True)

#     # Check for separated files
#     result_folder = output_folder / Path(filepath).stem
#     if not result_folder.exists():
#         return {"error": "Separation failed"}

#     # Collect all result files
#     result_files = [str(result_folder / f) for f in os.listdir(result_folder) if os.path.isfile(result_folder / f)]

#     return {"separated_files": result_files}
