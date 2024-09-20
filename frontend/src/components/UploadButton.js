import React from 'react';
import '../styles/UploadButton.css';

function UploadButton() {
    return (
        <div className="upload-container">
            <div class ="inner">
                <button className="upload-button">Choose audio file to upload</button>
            </div>
            <div class ="inner">
                <p className="file-formats">Formats: .wav, .mp3</p>
            </div>
        </div>
        
    );
}

export default UploadButton;
