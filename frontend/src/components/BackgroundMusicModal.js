import React, { useState } from 'react';
import '../styles/BackgroundMusicModal.css';

function BackgroundMusicModal({ isOpen, onClose, onApply, audioData, speakerInfo }) {
    const [musicType, setMusicType] = useState('calm');
    const [volumeLevel, setVolumeLevel] = useState(0.3);
    const [targetScope, setTargetScope] = useState('conversation'); // 'conversation' or 'all'
    const [isLoading, setIsLoading] = useState(false);

    const musicTypes = [
        { value: 'calm', label: 'Calm & Ambient', description: 'Relaxing background tones' },
        { value: 'upbeat', label: 'Upbeat & Energetic', description: 'Positive, motivating music' },
        { value: 'dramatic', label: 'Dramatic & Orchestral', description: 'Cinematic and intense' },
        { value: 'corporate', label: 'Corporate & Professional', description: 'Business-appropriate background' },
        { value: 'nature', label: 'Nature Sounds', description: 'Natural ambient sounds' }
    ];

    const handleApply = async (e) => {
        e.preventDefault();
        
        if (!audioData) {
            alert('No audio data available');
            return;
        }

        setIsLoading(true);

        const requestData = {
            audio_base64: audioData.audio_base64 || audioData,
            music_type: musicType,
            volume_level: volumeLevel,
            target_speaker: targetScope === 'conversation' ? speakerInfo?.speaker : null
        };

        try {
            await onApply(requestData);
            onClose();
        } catch (error) {
            console.error('Error applying background music:', error);
            alert('Failed to apply background music. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h2>Add Background Music</h2>
                    <button className="close-button" onClick={onClose} disabled={isLoading}>
                        ×
                    </button>
                </div>

                <form onSubmit={handleApply} className="music-form">
                    <div className="form-group">
                        <label>Apply music to:</label>
                        <div className="radio-group">
                            <label className="radio-option">
                                <input
                                    type="radio"
                                    value="conversation"
                                    checked={targetScope === 'conversation'}
                                    onChange={(e) => setTargetScope(e.target.value)}
                                    disabled={isLoading}
                                />
                                This conversation only
                                {speakerInfo && (
                                    <span className="speaker-info">
                                        (Speaker {speakerInfo.speaker})
                                    </span>
                                )}
                            </label>
                            <label className="radio-option">
                                <input
                                    type="radio"
                                    value="all"
                                    checked={targetScope === 'all'}
                                    onChange={(e) => setTargetScope(e.target.value)}
                                    disabled={isLoading}
                                />
                                Entire conversation
                            </label>
                        </div>
                    </div>

                    <div className="form-group">
                        <label htmlFor="musicType">Music Type</label>
                        <div className="music-type-grid">
                            {musicTypes.map(type => (
                                <div 
                                    key={type.value}
                                    className={`music-type-option ${musicType === type.value ? 'selected' : ''}`}
                                    onClick={() => !isLoading && setMusicType(type.value)}
                                >
                                    <div className="music-type-label">{type.label}</div>
                                    <div className="music-type-description">{type.description}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="form-group">
                        <label htmlFor="volumeLevel">
                            Music Volume: {Math.round(volumeLevel * 100)}%
                        </label>
                        <input
                            type="range"
                            id="volumeLevel"
                            min="0.1"
                            max="0.8"
                            step="0.1"
                            value={volumeLevel}
                            onChange={(e) => setVolumeLevel(parseFloat(e.target.value))}
                            disabled={isLoading}
                        />
                        <div className="volume-labels">
                            <span>10%</span>
                            <span>80%</span>
                        </div>
                    </div>

                    <div className="preview-section">
                        <h4>Preview Settings</h4>
                        <div className="preview-info">
                            <div className="preview-item">
                                <strong>Music:</strong> {musicTypes.find(t => t.value === musicType)?.label}
                            </div>
                            <div className="preview-item">
                                <strong>Volume:</strong> {Math.round(volumeLevel * 100)}%
                            </div>
                            <div className="preview-item">
                                <strong>Scope:</strong> {targetScope === 'conversation' ? 'Single conversation' : 'Entire audio'}
                            </div>
                        </div>
                    </div>

                    <div className="form-actions">
                        <button 
                            type="button" 
                            onClick={onClose} 
                            className="cancel-button"
                            disabled={isLoading}
                        >
                            Cancel
                        </button>
                        <button 
                            type="submit" 
                            className="apply-button"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Applying...' : 'Apply Music'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default BackgroundMusicModal;