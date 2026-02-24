import React, { useState } from 'react';
import '../styles/ArtificialSpeakerModal.css';

/**
 * Artificial Speaker Modal (Redesigned)
 *
 * Premium glassmorphic modal for creating AI-generated speakers.
 * Now includes emotion exaggeration control for Chatterbox TTS.
 */
function ArtificialSpeakerModal({ isOpen, onClose, onSubmit }) {
    const [prompt, setPrompt] = useState('');
    const [characteristics, setCharacteristics] = useState('');
    const [exaggeration, setExaggeration] = useState(0.5);
    const [isSubmitting, setIsSubmitting] = useState(false);

    if (!isOpen) return null;

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!prompt.trim()) return;

        setIsSubmitting(true);
        try {
            await onSubmit({
                prompt: prompt.trim(),
                speaker_characteristics: characteristics.trim(),
                exaggeration,
            });
            setPrompt('');
            setCharacteristics('');
            setExaggeration(0.5);
        } catch (err) {
            console.error('Submit failed:', err);
        }
        setIsSubmitting(false);
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-card" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <h3>🤖 Create AI Speaker</h3>
                    <button className="modal-close" onClick={onClose}>✕</button>
                </div>

                <form onSubmit={handleSubmit}>
                    <div className="modal-field">
                        <label>What should they say?</label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            placeholder="Enter dialogue or prompt for the AI speaker..."
                            rows={3}
                        />
                    </div>

                    <div className="modal-field">
                        <label>Voice Characteristics (optional)</label>
                        <input
                            type="text"
                            value={characteristics}
                            onChange={(e) => setCharacteristics(e.target.value)}
                            placeholder="e.g., deep male voice, young female, British accent..."
                        />
                    </div>

                    <div className="modal-field">
                        <label>
                            Emotion Exaggeration
                            <span className="field-value">{Math.round(exaggeration * 100)}%</span>
                        </label>
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={exaggeration}
                            onChange={(e) => setExaggeration(parseFloat(e.target.value))}
                            className="modal-slider"
                        />
                        <div className="slider-labels">
                            <span>Monotone</span>
                            <span>Natural</span>
                            <span>Expressive</span>
                        </div>
                    </div>

                    <div className="modal-actions">
                        <button type="button" className="btn-secondary" onClick={onClose}>
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="btn-primary"
                            disabled={!prompt.trim() || isSubmitting}
                        >
                            {isSubmitting ? '⏳ Generating...' : '🎵 Generate'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}

export default ArtificialSpeakerModal;