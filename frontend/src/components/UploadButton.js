import React, { useState, useEffect } from 'react';
import '../styles/UploadButton.css';
import axios from 'axios';
import io from 'socket.io-client';

function UploadButton() {

    const handleFileInput = (event) => {
        const file = event.target.files[0];
        event.preventDefault()
        const url = 'http://127.0.0.1:5000/upload';
        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileName', file.name);
        const config = {
            headers: {
                'content-type': 'multipart/form-data',
            },
        };
        axios.post(url, formData, config).then((response) => {
            console.log(response.data);
        });
    };
    
    const handleButtonClick = () => {
        document.getElementById('file-input').click();
    };

    return (
        <div className="upload-container">
            <div class ="inner">
                <button className="upload-button" onClick={handleButtonClick}>Choose audio file to upload</button>
                <input hidden type="file" id="file-input" className="file-input" onChange={handleFileInput} accept=".mp3, .wav" />
            </div>
            <div class ="inner">
                <p className="file-formats">Formats: .wav, .mp3</p>
            </div>
        </div>
        
    );
}

export default UploadButton;
