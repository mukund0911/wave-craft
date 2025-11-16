import React, { useState } from 'react';
import '../styles/UploadButton.css';
import axios from 'axios';
import { useNavigate  } from 'react-router-dom';

function UploadButton() {
    const navigate = useNavigate();
    const [isProcessing, setIsProcessing] = useState(false);
    const [statusMessage, setStatusMessage] = useState('');
    const [progress, setProgress] = useState(0);

    const pollStatus = async (jobId, apiUrl) => {
        const maxAttempts = 120; // 2 minutes max (polling every 1 second)
        let attempts = 0;

        const poll = async () => {
            try {
                const response = await axios.get(`${apiUrl}/status/${jobId}`);

                if (response.status === 200 && response.data.status === 'completed') {
                    // Processing complete
                    setStatusMessage('Transcription complete! Loading results...');
                    setProgress(100);
                    const { prediction, conversations, full_audio } = response.data;

                    // Navigate to result page
                    setTimeout(() => {
                        navigate('/result', {
                            state: { prediction, full_audio, conversations }
                        });
                    }, 500);
                    return;
                }

                // Update status message
                const status = response.data.status || 'processing';
                if (status === 'queued') {
                    setStatusMessage('Queued for processing...');
                    setProgress(10);
                } else if (status === 'processing') {
                    setStatusMessage('Transcribing audio...');
                    setProgress(Math.min(30 + attempts * 2, 90)); // Simulate progress
                }

                attempts++;
                if (attempts >= maxAttempts) {
                    throw new Error('Transcription timed out. Please try again.');
                }

                // Poll again after 1 second
                setTimeout(poll, 1000);
            } catch (error) {
                console.error('Error polling status:', error);
                setStatusMessage('Error processing file. Please try again.');
                setIsProcessing(false);
                alert('Failed to process audio. Please try again.');
            }
        };

        poll();
    };

    const handleFileInput = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        event.preventDefault();
        setIsProcessing(true);
        setStatusMessage('Uploading file...');
        setProgress(5);

        const apiUrl = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileName', file.name);

        const config = {
            headers: {
                'content-type': 'multipart/form-data',
            },
            withCredentials: true
        };

        try {
            const response = await axios.post(`${apiUrl}/upload`, formData, config);

            // Check if result was cached
            if (response.data.cached) {
                setStatusMessage('Retrieved from cache!');
                setProgress(100);
                const { prediction, conversations, full_audio } = response.data;

                setTimeout(() => {
                    navigate('/result', {
                        state: { prediction, full_audio, conversations }
                    });
                }, 500);
            } else if (response.status === 202) {
                // Async processing started
                const jobId = response.data.job_id;
                setStatusMessage('File uploaded. Processing...');
                setProgress(20);

                // Start polling
                pollStatus(jobId, apiUrl);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            setStatusMessage('Upload failed. Please try again.');
            setIsProcessing(false);
            alert('Failed to upload file. Please try again.');
        }
    };
    
    const handleButtonClick = () => {
        document.getElementById('file-input').click();
    };

    return (
        <div className="upload-container">
            <div className="inner">
                <button
                    className="upload-button"
                    onClick={handleButtonClick}
                    disabled={isProcessing}
                    style={{
                        opacity: isProcessing ? 0.6 : 1,
                        cursor: isProcessing ? 'not-allowed' : 'pointer'
                    }}
                >
                    {isProcessing ? 'Processing...' : 'Choose audio file to upload'}
                </button>
                <input
                    hidden
                    type="file"
                    id="file-input"
                    className="file-input"
                    onChange={handleFileInput}
                    accept=".mp3, .wav"
                    disabled={isProcessing}
                />
            </div>

            {isProcessing && (
                <div className="processing-status" style={{
                    marginTop: '20px',
                    textAlign: 'center'
                }}>
                    <div style={{
                        marginBottom: '10px',
                        fontSize: '14px',
                        color: '#555'
                    }}>
                        {statusMessage}
                    </div>
                    <div style={{
                        width: '100%',
                        maxWidth: '400px',
                        height: '8px',
                        backgroundColor: '#e0e0e0',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        margin: '0 auto'
                    }}>
                        <div style={{
                            width: `${progress}%`,
                            height: '100%',
                            backgroundColor: '#4CAF50',
                            transition: 'width 0.3s ease',
                            borderRadius: '4px'
                        }} />
                    </div>
                    <div style={{
                        marginTop: '5px',
                        fontSize: '12px',
                        color: '#888'
                    }}>
                        {progress}%
                    </div>
                </div>
            )}
        </div>
    );
}

export default UploadButton;
