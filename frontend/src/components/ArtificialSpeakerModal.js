import React, { useState } from 'react';
import '../styles/ArtificialSpeakerModal.css';

function ArtificialSpeakerModal({ isOpen, onClose, onSubmit, conversationHistory }) {
    const [speakerPrompt, setSpeakerPrompt] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!speakerPrompt.trim()) {
            alert('Please provide speaker characteristics');
            return;
        }

        setIsLoading(true);

        const requestData = {
            conversation_history: conversationHistory,
            speaker_prompt: speakerPrompt
        };

        try {
            await onSubmit(requestData);
            setSpeakerPrompt('');
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