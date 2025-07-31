import React, { useState } from 'react';
import '../styles/ArtificialSpeakerModal.css';

function ArtificialSpeakerModal({ isOpen, onClose, onSubmit, conversationHistory }) {
    const [speakerPrompt, setSpeakerPrompt] = useState('');
    const [addBackgroundMusic, setAddBackgroundMusic] = useState(false);
    const [musicType, setMusicType] = useState('calm');
    const [musicVolume, setMusicVolume] = useState(0.3);
    const [isLoading, setIsLoading] = useState(false);

    const musicTypes = [
        { value: 'calm', label: 'Calm & Ambient' },
        { value: 'upbeat', label: 'Upbeat & Energetic' },
        { value: 'dramatic', label: 'Dramatic & Orchestral' },
        { value: 'corporate', label: 'Corporate & Professional' },
        { value: 'nature', label: 'Nature Sounds' }
    ];

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!speakerPrompt.trim()) {
            alert('Please provide speaker characteristics');
            return;
        }

        setIsLoading(true);

        const requestData = {
            conversation_history: conversationHistory,
            speaker_prompt: speakerPrompt,
            add_background_music: addBackgroundMusic,
            music_type: musicType,
            music_volume: musicVolume
        };

        try {
            await onSubmit(requestData);
            setSpeakerPrompt('');
            setAddBackgroundMusic(false);
            setMusicType('calm');
            setMusicVolume(0.3);
            onClose();
        } catch (error) {
            console.error('Error creating artificial speaker:', error);
            alert('Failed to create artificial speaker. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal-content">
                <div className="modal-header">
                    <h2>Add Artificial Speaker</h2>
                    <button className="close-button" onClick={onClose} disabled={isLoading}>
                        ×
                    </button>
                </div>
                
                <form onSubmit={handleSubmit} className="speaker-form">
                    <div className="form-group">
                        <label htmlFor="speakerPrompt">
                            Speaker Characteristics
                            <span className="tooltip">
                                ℹ️
                                <span className="tooltip-text">
                                    Describe the speaker's gender, personality, emotion, and topic of interest. 
                                    Example: "A friendly female speaker with a calm demeanor discussing technology trends"
                                </span>
                            </span>
                        </label>
                        <textarea
                            id="speakerPrompt"
                            value={speakerPrompt}
                            onChange={(e) => setSpeakerPrompt(e.target.value)}
                            placeholder="Describe the artificial speaker (e.g., 'A confident male speaker with an energetic tone discussing business strategies')"
                            rows={4}
                            disabled={isLoading}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label>
                            <input
                                type="checkbox"
                                checked={addBackgroundMusic}
                                onChange={(e) => setAddBackgroundMusic(e.target.checked)}
                                disabled={isLoading}
                            />
                            Add Background Music
                        </label>
                    </div>

                    {addBackgroundMusic && (
                        <div className="music-options">
                            <div className="form-group">
                                <label htmlFor="musicType">Music Type</label>
                                <select
                                    id="musicType"
                                    value={musicType}
                                    onChange={(e) => setMusicType(e.target.value)}
                                    disabled={isLoading}
                                >
                                    {musicTypes.map(type => (
                                        <option key={type.value} value={type.value}>
                                            {type.label}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="form-group">
                                <label htmlFor="musicVolume">
                                    Music Volume: {Math.round(musicVolume * 100)}%
                                </label>
                                <input
                                    type="range"
                                    id="musicVolume"
                                    min="0"
                                    max="1"
                                    step="0.1"
                                    value={musicVolume}
                                    onChange={(e) => setMusicVolume(parseFloat(e.target.value))}
                                    disabled={isLoading}
                                />
                            </div>
                        </div>
                    )}

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
                            className="submit-button"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Creating...' : 'Create Speaker'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ArtificialSpeakerModal;