import React from 'react';
import '../styles/UploadButton.css';
import axios from 'axios';
import { useNavigate  } from 'react-router-dom';

function UploadButton() {

    const navigate = useNavigate();

    const handleFileInput = (event) => {
        const file = event.target.files[0];
        event.preventDefault()
        const url = `${process.env.REACT_APP_API_URL}/upload` || "http://127.0.0.1:5000/upload";
        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileName', file.name);
        const config = {
            headers: {
                'content-type': 'multipart/form-data',
            },
            withCredentials: true
        };
        axios.post(url, formData, config)
            .then((response) => {
                const prediction = response.data.prediction
                const conversations = response.data.conversations
                const full_audio = response.data.full_audio
                // Navigate to the prediction page with the result
                navigate('/result', { state: { prediction: prediction, full_audio: full_audio, conversations: conversations } });
            })
            .catch((error) => {
                console.error('Error uploading file:', error);
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
            {/* <div class ="inner">
                <p className="file-formats">Formats: .wav, .mp3</p>
            </div> */}
        </div>
        
    );
}

export default UploadButton;
