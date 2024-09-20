# app/routes.py
from flask import Blueprint, request, jsonify
from demucs.apply import apply_model
from demucs.pretrained import get_model
from pathlib import Path
import os
import shutil
import subprocess

main = Blueprint('main', __name__)

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

        # Call Demucs sound separation
        separated_audio = separate_audio(filepath)

        return jsonify({"message": "File processed", "separated_audio": separated_audio}), 200


def separate_audio(filepath):
    """ This function separates the uploaded file into its components using Demucs. """
    # Get model
    model = get_model('htdemucs')  # Load Demucs model (htdemucs is a good general choice)

    output_folder = Path(SEPARATED_FOLDER) / Path(filepath).stem
    os.makedirs(output_folder, exist_ok=True)

    # Use subprocess to safely call Demucs
    command = f"demucs {filepath} -o {output_folder}"
    subprocess.run(command.split(), check=True)

    # Check for separated files
    result_folder = output_folder / Path(filepath).stem
    if not result_folder.exists():
        return {"error": "Separation failed"}

    # Collect all result files
    result_files = [str(result_folder / f) for f in os.listdir(result_folder) if os.path.isfile(result_folder / f)]

    return {"separated_files": result_files}
