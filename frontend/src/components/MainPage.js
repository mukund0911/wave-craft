import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import Header from './Header';
import EmotionPicker from './EmotionPicker';
import '../styles/MainPage.css';
import '../styles/EmotionPicker.css';

const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

/**
 * MainPage — Transcript Editor with Emotion Tagging
 *
 * Features:
 * - Speaker-wise transcript display
 * - Click word to toggle delete (strikethrough)
 * - Right-click word to open emotion picker
 * - Add new words inline
 * - Paralinguistic tags ([laugh], [sigh], etc.)
 * - Generate modified audio with voice cloning
 * - Audio playback per segment and full
 */
class MainPage extends Component {
    constructor(props) {
        super(props);

        // Get data from navigation state
        const locationState = props.location?.state || (typeof window !== 'undefined'
            ? window.history.state?.usr : null) || {};

        this.state = {
            conversations: locationState.conversations || [],
            fullAudio: locationState.full_audio || '',
            // Track modifications per conversation
            modifications: {}, // { convKey: { deletedWords: Set, insertedWords: [], emotions: {}, exaggeration: 0.5 } }
            // Emotion picker
            showEmotionPicker: false,
            emotionPickerTarget: null, // { convKey, wordIndex }
            emotionPickerPosition: { top: 0, left: 0 },
            // Insert mode
            insertMode: null, // { convKey, afterWordIndex }
            insertText: '',
            // Generation
            isGenerating: false,
            generatedAudio: null,
            generatedStats: null,
            // Playback
            currentlyPlaying: null,
        };

        this.audioRef = React.createRef();
    }

    // ──────────────────────────────────────────────────
    // Data helpers
    // ──────────────────────────────────────────────────

    getConvData(convItem) {
        const keys = Object.keys(convItem);
        return { key: keys[0], data: convItem[keys[0]] };
    }

    getMods(convKey) {
        return this.state.modifications[convKey] || {
            deletedWords: new Set(),
            insertedWords: [], // [{ afterIndex, text }]
            emotions: {},      // { wordIndex: { type, intensity } }
            exaggeration: 0.5,
        };
    }

    setMods(convKey, mods) {
        this.setState(prev => ({
            modifications: {
                ...prev.modifications,
                [convKey]: mods,
            }
        }));
    }

    // ──────────────────────────────────────────────────
    // Word interactions
    // ──────────────────────────────────────────────────

    handleWordClick = (convKey, wordIndex) => {
        const mods = { ...this.getMods(convKey) };
        const deletedWords = new Set(mods.deletedWords);

        if (deletedWords.has(wordIndex)) {
            deletedWords.delete(wordIndex);
        } else {
            deletedWords.add(wordIndex);
        }

        mods.deletedWords = deletedWords;
        this.setMods(convKey, mods);
    };

    handleWordRightClick = (e, convKey, wordIndex) => {
        e.preventDefault();
        const rect = e.target.getBoundingClientRect();
        const mods = this.getMods(convKey);

        this.setState({
            showEmotionPicker: true,
            emotionPickerTarget: { convKey, wordIndex },
            emotionPickerPosition: {
                top: rect.bottom + 8,
                left: Math.min(rect.left, window.innerWidth - 340),
            },
        });
    };

    handleEmotionSelect = (emotion) => {
        const { convKey, wordIndex } = this.state.emotionPickerTarget;
        const mods = { ...this.getMods(convKey) };
        const emotions = { ...mods.emotions };

        if (emotion === null) {
            delete emotions[wordIndex];
        } else {
            emotions[wordIndex] = emotion;
        }

        mods.emotions = emotions;
        this.setMods(convKey, mods);
    };

    handleCloseEmotionPicker = () => {
        this.setState({ showEmotionPicker: false, emotionPickerTarget: null });
    };

    // ──────────────────────────────────────────────────
    // Insert words
    // ──────────────────────────────────────────────────

    handleInsertClick = (convKey, afterWordIndex) => {
        this.setState({
            insertMode: { convKey, afterWordIndex },
            insertText: '',
        });
    };

    handleInsertSubmit = () => {
        const { insertMode, insertText } = this.state;
        if (!insertMode || !insertText.trim()) {
            this.setState({ insertMode: null, insertText: '' });
            return;
        }

        const mods = { ...this.getMods(insertMode.convKey) };
        const insertedWords = [...(mods.insertedWords || [])];
        insertedWords.push({
            afterIndex: insertMode.afterWordIndex,
            text: insertText.trim(),
        });

        mods.insertedWords = insertedWords;
        this.setMods(insertMode.convKey, mods);
        this.setState({ insertMode: null, insertText: '' });
    };

    handleInsertKeyDown = (e) => {
        if (e.key === 'Enter') {
            this.handleInsertSubmit();
        } else if (e.key === 'Escape') {
            this.setState({ insertMode: null, insertText: '' });
        }
    };

    // ──────────────────────────────────────────────────
    // Add paralinguistic tag
    // ──────────────────────────────────────────────────

    handleAddParalinguistic = (convKey, afterWordIndex, tag) => {
        const mods = { ...this.getMods(convKey) };
        const insertedWords = [...(mods.insertedWords || [])];
        insertedWords.push({
            afterIndex: afterWordIndex,
            text: tag,
            isParalinguistic: true,
        });
        mods.insertedWords = insertedWords;
        this.setMods(convKey, mods);
    };

    // ──────────────────────────────────────────────────
    // Exaggeration control
    // ──────────────────────────────────────────────────

    handleExaggerationChange = (convKey, value) => {
        const mods = { ...this.getMods(convKey) };
        mods.exaggeration = parseFloat(value);
        this.setMods(convKey, mods);
    };

    // ──────────────────────────────────────────────────
    // Build modified text
    // ──────────────────────────────────────────────────

    buildModifiedText(originalText, mods) {
        const words = originalText.split(/\s+/);
        const result = [];

        for (let i = 0; i < words.length; i++) {
            // Check for inserted words before this position
            const insertsBefore = (mods.insertedWords || [])
                .filter(ins => ins.afterIndex === i - 1);
            for (const ins of insertsBefore) {
                result.push(ins.text);
            }

            // Add word if not deleted
            if (!mods.deletedWords.has(i)) {
                result.push(words[i]);
            }
        }

        // Insertions at the end
        const insertsEnd = (mods.insertedWords || [])
            .filter(ins => ins.afterIndex >= words.length - 1 || ins.afterIndex === -1);
        for (const ins of insertsEnd) {
            if (!result.includes(ins.text)) {
                result.push(ins.text);
            }
        }

        return result.join(' ');
    }

    buildEmotionsList(mods) {
        const emotions = [];
        for (const [wordIndex, emotion] of Object.entries(mods.emotions || {})) {
            emotions.push({
                type: emotion.type,
                intensity: emotion.intensity || 0.5,
                wordIndex: parseInt(wordIndex),
            });
        }
        return emotions;
    }

    // ──────────────────────────────────────────────────
    // Playback
    // ──────────────────────────────────────────────────

    playSegmentAudio = (convKey, audioB64) => {
        const audio = this.audioRef.current;
        if (!audio) return;

        if (this.state.currentlyPlaying === convKey) {
            audio.pause();
            audio.currentTime = 0;
            this.setState({ currentlyPlaying: null });
            return;
        }

        audio.src = `data:audio/wav;base64,${audioB64}`;
        audio.onended = () => this.setState({ currentlyPlaying: null });
        audio.play();
        this.setState({ currentlyPlaying: convKey });
    };

    // ──────────────────────────────────────────────────
    // Generate modified audio
    // ──────────────────────────────────────────────────

    handleGenerate = async () => {
        this.setState({ isGenerating: true, generatedAudio: null });

        try {
            const conversationsUpdated = this.state.conversations.map(convItem => {
                const { key, data } = this.getConvData(convItem);
                const mods = this.getMods(key);
                const modifiedText = this.buildModifiedText(data.original.text, mods);
                const emotions = this.buildEmotionsList(mods);

                return {
                    [key]: {
                        speaker: data.speaker,
                        original: data.original,
                        modified: {
                            text: modifiedText,
                            emotions: emotions,
                            exaggeration: mods.exaggeration || 0.5,
                        }
                    }
                };
            });

            const formData = new FormData();
            formData.append('conversations_updated', JSON.stringify(conversationsUpdated));
            formData.append('full_audio', this.state.fullAudio);

            const response = await axios.post(
                `${apiUrl}/conversations_modified`,
                formData,
                {
                    headers: { 'content-type': 'multipart/form-data' },
                    withCredentials: true,
                }
            );

            this.setState({
                isGenerating: false,
                generatedAudio: response.data.modified_audio,
                generatedStats: response.data.stats,
            });

        } catch (err) {
            console.error('Generation error:', err);
            this.setState({
                isGenerating: false,
            });
            alert(err.response?.data?.error || 'Failed to generate audio. Please try again.');
        }
    };

    // ──────────────────────────────────────────────────
    // Download
    // ──────────────────────────────────────────────────

    handleDownload = () => {
        const { generatedAudio } = this.state;
        if (!generatedAudio) return;

        const bytes = atob(generatedAudio);
        const array = new Uint8Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) {
            array[i] = bytes.charCodeAt(i);
        }
        const blob = new Blob([array], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'wavecraft_modified.wav';
        a.click();
        URL.revokeObjectURL(url);
    };

    handlePlayGenerated = () => {
        const { generatedAudio } = this.state;
        if (!generatedAudio || !this.audioRef.current) return;

        const audio = this.audioRef.current;
        audio.src = `data:audio/wav;base64,${generatedAudio}`;
        audio.play();
    };

    // ──────────────────────────────────────────────────
    // Stats
    // ──────────────────────────────────────────────────

    getEditStats() {
        let modified = 0, deleted = 0, inserted = 0, emotions = 0;

        for (const [key, mods] of Object.entries(this.state.modifications)) {
            deleted += (mods.deletedWords?.size || 0);
            inserted += (mods.insertedWords?.length || 0);
            emotions += Object.keys(mods.emotions || {}).length;
            if (deleted > 0 || inserted > 0 || emotions > 0) modified++;
        }

        return { modified, deleted, inserted, emotions };
    }

    // ──────────────────────────────────────────────────
    // Render
    // ──────────────────────────────────────────────────

    renderWords(convKey, text, mods) {
        const words = text.split(/\s+/);
        const elements = [];
        const { insertMode } = this.state;

        // Insert input at beginning (afterWordIndex === -1)
        if (insertMode && insertMode.convKey === convKey && insertMode.afterWordIndex === -1) {
            elements.push(
                <input
                    key="input-start"
                    type="text"
                    className="insert-input"
                    placeholder="Type new words…"
                    value={this.state.insertText}
                    onChange={(e) => this.setState({ insertText: e.target.value })}
                    onKeyDown={this.handleInsertKeyDown}
                    onBlur={this.handleInsertSubmit}
                    autoFocus
                    style={{
                        background: 'rgba(5, 150, 105, 0.08)',
                        border: '1px solid rgba(5, 150, 105, 0.25)',
                        borderRadius: '6px',
                        padding: '3px 8px',
                        color: '#059669',
                        fontSize: '0.9rem',
                        outline: 'none',
                        minWidth: '120px',
                        fontFamily: 'inherit',
                    }}
                />
            );
            elements.push(<span key="sp-ins-start"> </span>);
        }

        for (let i = 0; i < words.length; i++) {
            // Inserted words before this position
            const insertsBefore = (mods.insertedWords || [])
                .filter(ins => ins.afterIndex === i - 1);
            for (const ins of insertsBefore) {
                if (ins.isParalinguistic) {
                    elements.push(
                        <span key={`para-${i}-${ins.text}`} className="paralinguistic-tag">
                            {ins.text}
                        </span>
                    );
                } else {
                    elements.push(
                        <span key={`ins-${i}-${ins.text}`} className="word inserted">
                            {ins.text}{' '}
                        </span>
                    );
                }
            }

            // The word itself
            const isDeleted = mods.deletedWords.has(i);
            const emotion = mods.emotions[i];
            let className = 'word';
            if (isDeleted) className += ' deleted';
            if (emotion) className += ` has-emotion ${emotion.type}`;

            elements.push(
                <span
                    key={`w-${i}`}
                    className={className}
                    onClick={() => this.handleWordClick(convKey, i)}
                    onContextMenu={(e) => this.handleWordRightClick(e, convKey, i)}
                    onDoubleClick={() => this.handleInsertClick(convKey, i)}
                    title={isDeleted ? 'Click to restore' : 'Click to delete • Right-click for emotion • Double-click to insert after'}
                >
                    {words[i]}
                    {emotion && !emotion.isParalinguistic && (
                        <span className={`emotion-chip ${emotion.type}`}>
                            {emotion.type}
                        </span>
                    )}
                </span>
            );

            elements.push(<span key={`sp-${i}`}> </span>);

            // Show insert input after this word
            if (insertMode && insertMode.convKey === convKey && insertMode.afterWordIndex === i) {
                elements.push(
                    <input
                        key={`input-${i}`}
                        type="text"
                        className="insert-input"
                        placeholder="Type new words…"
                        value={this.state.insertText}
                        onChange={(e) => this.setState({ insertText: e.target.value })}
                        onKeyDown={this.handleInsertKeyDown}
                        onBlur={this.handleInsertSubmit}
                        autoFocus
                        style={{
                            background: 'rgba(5, 150, 105, 0.08)',
                            border: '1px solid rgba(5, 150, 105, 0.25)',
                            borderRadius: '6px',
                            padding: '3px 8px',
                            color: '#059669',
                            fontSize: '0.9rem',
                            outline: 'none',
                            minWidth: '120px',
                            fontFamily: 'inherit',
                        }}
                    />
                );
                elements.push(<span key={`sp-ins-${i}`}> </span>);
            }
        }

        return elements;
    }

    getSpeakerClass(speaker) {
        const map = { A: 'a', B: 'b', C: 'c', D: 'd', E: 'e' };
        return `speaker-${map[speaker] || 'a'}`;
    }

    render() {
        const {
            conversations, showEmotionPicker, emotionPickerTarget,
            emotionPickerPosition, isGenerating, generatedAudio,
        } = this.state;

        const stats = this.getEditStats();
        const hasChanges = stats.deleted > 0 || stats.inserted > 0 || stats.emotions > 0;

        return (
            <div className="main-page">
                {/* Same Header as landing page */}
                <Header />

                {/* Conversations */}
                <div className="main-content">
                    {conversations.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '80px 0', color: 'var(--text-muted)' }}>
                            <p style={{ fontSize: '1.2rem', marginBottom: '8px' }}>No transcript loaded</p>
                            <p>Upload an audio file to get started.</p>
                            <Link to="/" className="btn-primary" style={{
                                display: 'inline-flex', marginTop: '20px',
                                textDecoration: 'none'
                            }}>
                                Go to Upload
                            </Link>
                        </div>
                    ) : conversations.map((convItem, idx) => {
                        const { key, data } = this.getConvData(convItem);
                        const mods = this.getMods(key);
                        const isModified = (mods.deletedWords?.size > 0) ||
                            (mods.insertedWords?.length > 0) ||
                            Object.keys(mods.emotions || {}).length > 0;

                        return (
                            <div
                                key={key}
                                className="speaker-section"
                                style={{ animationDelay: `${idx * 0.05}s` }}
                            >
                                <div className="speaker-header">
                                    <div className={`speaker-avatar ${this.getSpeakerClass(data.speaker)}`}>
                                        {data.speaker}
                                    </div>
                                    <span className="speaker-name">
                                        Speaker {data.speaker}
                                    </span>
                                    {data.original?.start !== undefined && (
                                        <span className="speaker-time">
                                            {formatTime(data.original.start)} — {formatTime(data.original.end)}
                                        </span>
                                    )}
                                </div>

                                <div className={`transcript-block ${isModified ? 'modified' : ''}`}>
                                    <div className="transcript-text">
                                        {this.renderWords(key, data.original.text, mods)}
                                    </div>

                                    <div className="transcript-actions">
                                        <button
                                            className="action-btn"
                                            onClick={() => this.handleInsertClick(key, -1)}
                                            title="Insert text at beginning"
                                        >
                                            ➕ Insert
                                        </button>
                                        <button
                                            className={`action-btn ${Object.keys(mods.emotions || {}).length > 0 ? 'active' : ''}`}
                                            onClick={(e) => this.handleWordRightClick(e, key, 0)}
                                            title="Add emotion to first word"
                                        >
                                            😊 Emotion
                                        </button>

                                        {data.original?.speaker_audio && (
                                            <button
                                                className="play-btn"
                                                onClick={() => this.playSegmentAudio(key, data.original.speaker_audio)}
                                            >
                                                {this.state.currentlyPlaying === key ? '⏹ Stop' : '▶ Play'}
                                            </button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>

                {/* Bottom Generate Bar */}
                {conversations.length > 0 && (
                    <div className="bottom-bar">
                        <div className="bottom-bar-info">
                            <span className="bottom-bar-stat">
                                <span className="dot modified"></span>
                                {stats.modified} segments
                            </span>
                            <span className="bottom-bar-stat">
                                <span className="dot deleted"></span>
                                {stats.deleted} deleted
                            </span>
                            <span className="bottom-bar-stat">
                                <span className="dot inserted"></span>
                                {stats.inserted} inserted
                            </span>
                            <span className="bottom-bar-stat">
                                <span className="dot emotion"></span>
                                {stats.emotions} emotions
                            </span>
                        </div>

                        <button
                            className="generate-btn"
                            onClick={this.handleGenerate}
                            disabled={isGenerating}
                        >
                            {isGenerating ? (
                                <>⏳ Generating...</>
                            ) : (
                                <>🎵 Generate Modified Audio</>
                            )}
                        </button>
                    </div>
                )}

                {/* Loading Overlay */}
                {isGenerating && (
                    <div className="loading-overlay">
                        <div className="loading-spinner"></div>
                        <div className="loading-text">
                            Generating modified audio with voice cloning...
                        </div>
                    </div>
                )}

                {/* Download Panel */}
                {generatedAudio && (
                    <div className="download-panel">
                        <div className="download-card">
                            <h3>✨ Audio Generated!</h3>
                            <p>
                                Modified audio is ready.
                                Play to preview or download the file.
                            </p>
                            <div className="download-actions">
                                <button className="btn-secondary" onClick={this.handlePlayGenerated}>
                                    ▶ Play
                                </button>
                                <button className="btn-primary" onClick={this.handleDownload}>
                                    ⬇ Download WAV
                                </button>
                            </div>
                            <button
                                style={{
                                    marginTop: '16px',
                                    background: 'none',
                                    border: 'none',
                                    color: 'var(--text-muted)',
                                    cursor: 'pointer',
                                    fontSize: '0.85rem',
                                }}
                                onClick={() => this.setState({ generatedAudio: null })}
                            >
                                ← Back to Editor
                            </button>
                        </div>
                    </div>
                )}

                {/* Emotion Picker Popover */}
                {showEmotionPicker && emotionPickerTarget && (
                    <>
                        <div
                            style={{
                                position: 'fixed', inset: 0, zIndex: 199,
                                background: 'transparent',
                            }}
                            onClick={this.handleCloseEmotionPicker}
                        />
                        <EmotionPicker
                            position={emotionPickerPosition}
                            onSelect={this.handleEmotionSelect}
                            onClose={this.handleCloseEmotionPicker}
                            currentEmotion={
                                this.getMods(emotionPickerTarget.convKey)
                                    .emotions[emotionPickerTarget.wordIndex]
                            }
                        />
                    </>
                )}

                {/* Hidden audio element */}
                <audio ref={this.audioRef} style={{ display: 'none' }} />
            </div>
        );
    }
}

// Helper: format seconds to MM:SS
function formatTime(seconds) {
    if (seconds == null) return '';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// Wrapper to pass location via hook (for navigation state)
function MainPageWrapper(props) {
    // Read navigation state from window.history for class component
    return <MainPage {...props} location={window.history.state ? { state: window.history.state.usr } : {}} />;
}

export default MainPageWrapper;