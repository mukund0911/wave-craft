# app/routes.py
import os
# import subprocess
# from pathlib import Path
from flask_socketio import SocketIO
# from demucs.pretrained import get_model
from flask import Blueprint, request, jsonify

from backend.models.speech_music_classifier.sm_inference import sm_inference
from backend.models.speech_edit.speech_to_textv2 import SpeechModel

main = Blueprint('main', __name__)
socketio = SocketIO()

# Folder to store uploaded files
UPLOAD_FOLDER = './uploads'
SEPARATED_FOLDER = './separated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEPARATED_FOLDER, exist_ok=True)

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

        # Predict whether the uploaded is is "speech" or "music"
        prediction = sm_inference(filepath)

        if prediction == "Speech":
            _speech_model = SpeechModel(filepath)
            conversations, full_audio = _speech_model.speech_to_text()
        else:
            separated_convs = "TODO"

        return jsonify({"message": "File processed. Processing started.", 
                        "prediction": prediction, 
                        "full_audio" : full_audio,
                        "conversations" : conversations}), 200

@main.route('/conversations_modified', methods=['POST'])  
def modified_transcript():
    # Get the finalStructure JSON from the request
    conversation_mod = request.files.get('conversationsUpdated')
    full_audio_file = request.files.get('full_audio')

    if conversation_mod:
        # Read the finalStructure JSON
        final_structure_data = conversation_mod.read().decode('utf-8')
        print("Received finalStructure:", final_structure_data)

    if full_audio_file:
        # Save the full_audio file
        audio_path = os.path.join("uploads", full_audio_file.filename)
        full_audio_file.save(audio_path)
        print(f"Audio file received")

    _speech_model = SpeechModel(f"./uploads/{full_audio_file.filename}")
    modified_audio = _speech_model.text_to_speech(conversation_mod)


    return jsonify({"message": "Data and audio received successfully!", 
                    "modified_audio":modified_audio}), 200
        

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
