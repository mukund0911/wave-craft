import React, { useState } from 'react';
import '../styles/EmotionPicker.css';

/**
 * Emotion Picker Component
 *
 * Floating popover for selecting emotions on transcript words/phrases.
 * Features:
 * - Emoji emotion grid (happy, sad, angry, excited, calm, etc.)
 * - Intensity slider (subtle → exaggerated)
 * - Paralinguistic tag buttons ([laugh], [sigh], [gasp], etc.)
 */

const EMOTIONS = [
    { id: 'happy', emoji: '😊', label: 'Happy' },
    { id: 'sad', emoji: '😢', label: 'Sad' },
    { id: 'angry', emoji: '😠', label: 'Angry' },
    { id: 'excited', emoji: '🤩', label: 'Excited' },
    { id: 'calm', emoji: '😌', label: 'Calm' },
    { id: 'fearful', emoji: '😨', label: 'Fearful' },
    { id: 'surprised', emoji: '😲', label: 'Surprised' },
    { id: 'neutral', emoji: '😐', label: 'Neutral' },
];

const PARALINGUISTIC = [
    { id: '[laugh]', emoji: '😂', label: 'Laugh' },
    { id: '[sigh]', emoji: '😮‍💨', label: 'Sigh' },
    { id: '[gasp]', emoji: '😱', label: 'Gasp' },
    { id: '[cough]', emoji: '🤧', label: 'Cough' },
    { id: '[chuckle]', emoji: '🤭', label: 'Chuckle' },
];

function EmotionPicker({ position, onSelect, onClose, currentEmotion }) {
    const [selectedEmotion, setSelectedEmotion] = useState(currentEmotion?.type || null);
    const [intensity, setIntensity] = useState(currentEmotion?.intensity || 0.5);
    const [activeTab, setActiveTab] = useState('emotions'); // 'emotions' | 'sounds'

    const handleEmotionClick = (emotionId) => {
        setSelectedEmotion(emotionId);
        onSelect({
            type: emotionId,
            intensity: intensity,
        });
    };

    const handleParalinguisticClick = (tagId) => {
        onSelect({
            type: tagId,
            intensity: 1.0,
            isParalinguistic: true,
        });
        onClose();
    };

    const handleIntensityChange = (e) => {
        const val = parseFloat(e.target.value);
        setIntensity(val);
        if (selectedEmotion) {
            onSelect({
                type: selectedEmotion,
                intensity: val,
            });
        }
    };

    const handleRemove = () => {
        onSelect(null);
        onClose();
    };

    return (
        <div
            className="emotion-picker"
            style={{
                top: position?.top ?? 0,
                left: position?.left ?? 0,
            }}
        >
            <div className="emotion-picker-header">
                <div className="emotion-tabs">
                    <button
                        className={`emotion-tab ${activeTab === 'emotions' ? 'active' : ''}`}
                        onClick={() => setActiveTab('emotions')}
                    >
                        Emotions
                    </button>
                    <button
                        className={`emotion-tab ${activeTab === 'sounds' ? 'active' : ''}`}
                        onClick={() => setActiveTab('sounds')}
                    >
                        Sounds
                    </button>
                </div>
                <button className="emotion-close" onClick={onClose}>✕</button>
            </div>

            {activeTab === 'emotions' && (
                <>
                    <div className="emotion-grid">
                        {EMOTIONS.map((emotion) => (
                            <button
                                key={emotion.id}
                                className={`emotion-item ${selectedEmotion === emotion.id ? 'selected' : ''}`}
                                onClick={() => handleEmotionClick(emotion.id)}
                                title={emotion.label}
                            >
                                <span className="emotion-emoji">{emotion.emoji}</span>
                                <span className="emotion-label">{emotion.label}</span>
                            </button>
                        ))}
                    </div>

                    {selectedEmotion && (
                        <div className="emotion-intensity">
                            <div className="intensity-header">
                                <span>Intensity</span>
                                <span className="intensity-value">{Math.round(intensity * 100)}%</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={intensity}
                                onChange={handleIntensityChange}
                                className="intensity-slider"
                            />
                            <div className="intensity-labels">
                                <span>Subtle</span>
                                <span>Exaggerated</span>
                            </div>
                        </div>
                    )}
                </>
            )}

            {activeTab === 'sounds' && (
                <div className="paralinguistic-grid">
                    {PARALINGUISTIC.map((tag) => (
                        <button
                            key={tag.id}
                            className="paralinguistic-item"
                            onClick={() => handleParalinguisticClick(tag.id)}
                        >
                            <span className="paralinguistic-emoji">{tag.emoji}</span>
                            <span className="paralinguistic-label">{tag.label}</span>
                            <span className="paralinguistic-tag">{tag.id}</span>
                        </button>
                    ))}
                </div>
            )}

            {currentEmotion && (
                <button className="emotion-remove" onClick={handleRemove}>
                    Remove Emotion
                </button>
            )}
        </div>
    );
}

export default EmotionPicker;
