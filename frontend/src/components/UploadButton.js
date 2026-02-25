import React, { Component } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/LandingPage.css';

const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

/**
 * Premium Upload Button with drag-and-drop zone and animated progress.
 */
class UploadButtonClass extends Component {
    constructor(props) {
        super(props);
        this.state = {
            isDragging: false,
            isUploading: false,
            progress: 0,
            statusText: '',
            error: null,
        };
        this.fileInputRef = React.createRef();
    }

    handleDragOver = (e) => {
        e.preventDefault();
        this.setState({ isDragging: true });
    };

    handleDragLeave = (e) => {
        e.preventDefault();
        this.setState({ isDragging: false });
    };

    handleDrop = (e) => {
        e.preventDefault();
        this.setState({ isDragging: false });
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    };

    handleClick = () => {
        this.fileInputRef.current.click();
    };

    handleFileChange = (e) => {
        if (e.target.files.length > 0) {
            this.uploadFile(e.target.files[0]);
        }
    };

    uploadFile = async (file) => {
        this.setState({
            isUploading: true,
            progress: 10,
            statusText: 'Uploading audio...',
            error: null,
        });

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(`${apiUrl}/upload`, formData, {
                headers: { 'content-type': 'multipart/form-data' },
                onUploadProgress: (progressEvent) => {
                    const pct = Math.round((progressEvent.loaded * 40) / progressEvent.total) + 10;
                    this.setState({ progress: pct, statusText: 'Uploading audio...' });
                },
            });

            if (response.data.status === 'completed') {
                // Immediate result (cached)
                this.setState({ progress: 100, statusText: 'Complete!' });
                setTimeout(() => {
                    this.props.navigate('/result', {
                        state: {
                            conversations: response.data.conversations,
                            full_audio: response.data.full_audio,
                        },
                    });
                }, 400);
            } else if (response.data.job_id) {
                // Async — poll for status
                this.setState({ progress: 50, statusText: 'Transcribing audio...' });
                this.pollStatus(response.data.job_id);
            }
        } catch (err) {
            console.error('Upload error:', err);
            this.setState({
                isUploading: false,
                error: err.response?.data?.error || 'Upload failed. Please try again.',
                progress: 0,
                statusText: '',
            });
        }
    };

    pollStatus = async (jobId) => {
        let attempts = 0;
        const maxAttempts = 120;

        const poll = async () => {
            if (attempts >= maxAttempts) {
                this.setState({
                    isUploading: false,
                    error: 'Processing timed out. Please try again.',
                    progress: 0,
                });
                return;
            }

            try {
                const response = await axios.get(`${apiUrl}/status/${jobId}`);

                if (response.data.status === 'completed') {
                    this.setState({ progress: 100, statusText: 'Complete!' });
                    setTimeout(() => {
                        this.props.navigate('/result', {
                            state: {
                                conversations: response.data.conversations,
                                full_audio: response.data.full_audio,
                            },
                        });
                    }, 400);
                    return;
                }

                if (response.data.status === 'error') {
                    this.setState({
                        isUploading: false,
                        error: response.data.error || 'Processing failed.',
                        progress: 0,
                    });
                    return;
                }

                // Still processing
                attempts++;
                const progressVal = Math.min(50 + (attempts / maxAttempts) * 45, 95);
                this.setState({
                    progress: progressVal,
                    statusText: 'Transcribing audio...',
                });
                setTimeout(poll, 1000);
            } catch (err) {
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(poll, 2000);
                } else {
                    this.setState({
                        isUploading: false,
                        error: 'Connection lost. Please try again.',
                        progress: 0,
                    });
                }
            }
        };

        poll();
    };

    render() {
        const { isDragging, isUploading, progress, statusText, error } = this.state;

        return (
            <div>
                {!isUploading ? (
                    <div
                        className={`upload-dropzone ${isDragging ? 'dragover' : ''}`}
                        onDragOver={this.handleDragOver}
                        onDragLeave={this.handleDragLeave}
                        onDrop={this.handleDrop}
                        onClick={this.handleClick}
                    >
                        <span className="upload-icon">
                            {isDragging ? '📥' : '🎧'}
                        </span>
                        <div className="upload-text">
                            {isDragging
                                ? 'Drop your audio here'
                                : 'Drop audio file or click to browse'}
                        </div>
                        <div className="upload-subtext">
                            Supports WAV, MP3, M4A, FLAC, OGG
                        </div>
                        <input
                            type="file"
                            ref={this.fileInputRef}
                            onChange={this.handleFileChange}
                            accept="audio/*"
                            style={{ display: 'none' }}
                        />
                    </div>
                ) : (
                    <div className="upload-progress">
                        <div className="progress-bar-wrapper">
                            <div
                                className="progress-bar-fill"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <div className="progress-text">{statusText}</div>
                    </div>
                )}

                {error && (
                    <div style={{
                        marginTop: '16px',
                        color: '#ef4444',
                        fontSize: '0.875rem',
                        textAlign: 'center',
                    }}>
                        {error}
                    </div>
                )}
            </div>
        );
    }
}

// Wrapper to inject useNavigate hook into class component
function UploadButton() {
    const navigate = useNavigate();
    return <UploadButtonClass navigate={navigate} />;
}

export default UploadButton;
