import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import Header from './Header';
import EmotionPicker from './EmotionPicker';
import '../styles/MainPage.css';
import '../styles/EmotionPicker.css';

const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const GENERATION_MESSAGES = [
    'Analyzing edits...',
    'Cloning voices...',
    'Synthesizing speech...',
    'Assembling audio...',
];

const SESSION_KEY = 'wavecraft_session';

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

        // Fall back to sessionStorage if navigation state is empty
        let conversations = locationState.conversations || [];
        let fullAudio = locationState.full_audio || '';

        if (conversations.length === 0) {
            try {
                const saved = sessionStorage.getItem(SESSION_KEY);
                if (saved) {
                    const parsed = JSON.parse(saved);
                    conversations = parsed.conversations || [];
                    fullAudio = parsed.fullAudio || '';
                }
            } catch (e) {
                // Ignore parse errors
            }
        }

        this.state = {
            conversations,
            fullAudio,
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
            generationMessage: '',
            generatedAudio: null,
            generatedStats: null,
            // Playback
            currentlyPlaying: null,
            downloadPlaybackProgress: 0,
            isPlayingGenerated: false,
            // Toast
            toast: null, // { message, type: 'error'|'info' }
        };

        this.audioRef = React.createRef();
        this._abortController = null;
        this._generationMsgInterval = null;
        this._toastTimeout = null;

        // Merge speakers
        this.state.showMergeModal = false;
        this.state.mergeSource = null;  // speaker being merged away
        this.state.mergeTarget = null;  // speaker to merge into
    }

    componentDidMount() {
        this._saveSession();

        // Unsaved changes warning
        this._beforeUnloadHandler = (e) => {
            const stats = this.getEditStats();
            if (stats.deleted > 0 || stats.inserted > 0 || stats.emotions > 0) {
                e.preventDefault();
                e.returnValue = '';
            }
        };
        window.addEventListener('beforeunload', this._beforeUnloadHandler);

        // Keyboard shortcuts
        this._keydownHandler = (e) => {
            if (e.key === 'Escape') {
                if (this.state.showEmotionPicker) {
                    this.handleCloseEmotionPicker();
                } else if (this.state.insertMode) {
                    this.setState({ insertMode: null, insertText: '' });
                }
            }
        };
        window.addEventListener('keydown', this._keydownHandler);
    }

    componentWillUnmount() {
        // Pause audio
        const audio = this.audioRef.current;
        if (audio) {
            audio.pause();
            audio.currentTime = 0;
        }

        // Abort pending requests
        if (this._abortController) {
            this._abortController.abort();
        }

        // Remove event listeners
        window.removeEventListener('beforeunload', this._beforeUnloadHandler);
        window.removeEventListener('keydown', this._keydownHandler);

        // Clear intervals/timeouts
        if (this._generationMsgInterval) clearInterval(this._generationMsgInterval);
        if (this._toastTimeout) clearTimeout(this._toastTimeout);
    }

    _saveSession() {
        const { conversations, fullAudio } = this.state;
        if (conversations.length > 0) {
            try {
                // Save conversations without speaker_audio to stay within sessionStorage limits
                const lightConversations = conversations.map(convItem => {
                    const key = Object.keys(convItem)[0];
                    const data = convItem[key];
                    return {
                        [key]: {
                            ...data,
                            original: {
                                ...data.original,
                                speaker_audio: '', // strip heavy audio data
                            }
                        }
                    };
                });
                sessionStorage.setItem(SESSION_KEY, JSON.stringify({
                    conversations: lightConversations,
                    fullAudio: '', // too large for sessionStorage
                }));
            } catch (e) {
                // sessionStorage might be full — ignore
            }
        }
    }

    // ──────────────────────────────────────────────────
    // Toast notifications
    // ──────────────────────────────────────────────────

    showToast = (message, type = 'error') => {
        if (this._toastTimeout) clearTimeout(this._toastTimeout);
        this.setState({ toast: { message, type } });
        this._toastTimeout = setTimeout(() => {
            this.setState({ toast: null });
        }, 4000);
    };

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

        const pickerHeight = 280;
        const pickerWidth = 340;
        let top = rect.bottom + 8;
        let left = Math.min(rect.left, window.innerWidth - pickerWidth);

        // Flip above if it would overflow bottom
        if (top + pickerHeight > window.innerHeight) {
            top = rect.top - pickerHeight - 8;
        }
        // Clamp top to viewport
        top = Math.max(8, top);

        this.setState({
            showEmotionPicker: true,
            emotionPickerTarget: { convKey, wordIndex },
            emotionPickerPosition: { top, left },
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

    handleInsertBlur = () => {
        // Delay to allow click events (e.g. on other buttons) to fire first
        setTimeout(() => {
            if (this.state.insertMode) {
                this.handleInsertSubmit();
            }
        }, 200);
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
        const insertedWords = mods.insertedWords || [];
        const result = [];

        // Insertions at the beginning (afterIndex === -1)
        for (const ins of insertedWords.filter(ins => ins.afterIndex === -1)) {
            result.push(ins.text);
        }

        for (let i = 0; i < words.length; i++) {
            // Add word if not deleted
            if (!mods.deletedWords.has(i)) {
                result.push(words[i]);
            }

            // Insertions after this word
            for (const ins of insertedWords.filter(ins => ins.afterIndex === i)) {
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
        audio.play().catch((err) => {
            console.warn('Playback blocked:', err);
            this.showToast('Playback was blocked by the browser. Click to interact first.', 'info');
        });
        this.setState({ currentlyPlaying: convKey });
    };

    // ──────────────────────────────────────────────────
    // Generate modified audio
    // ──────────────────────────────────────────────────

    handleGenerate = async () => {
        // Guard: already generating
        if (this.state.isGenerating) return;

        // Guard: no edits
        const stats = this.getEditStats();
        if (stats.deleted === 0 && stats.inserted === 0 && stats.emotions === 0) return;

        this._abortController = new AbortController();

        // Start rotating generation messages
        let msgIndex = 0;
        this.setState({
            isGenerating: true,
            generatedAudio: null,
            generationMessage: GENERATION_MESSAGES[0],
        });
        this._generationMsgInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % GENERATION_MESSAGES.length;
            this.setState({ generationMessage: GENERATION_MESSAGES[msgIndex] });
        }, 3000);

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

            // Retry on 503 (Heroku dyno waking up)
            let response;
            for (let attempt = 0; attempt < 3; attempt++) {
                try {
                    response = await axios.post(
                        `${apiUrl}/conversations_modified`,
                        formData,
                        {
                            headers: { 'content-type': 'multipart/form-data' },
                            timeout: 180000,
                            signal: this._abortController.signal,
                        }
                    );
                    break;
                } catch (retryErr) {
                    if (retryErr.name === 'CanceledError' || retryErr.name === 'AbortError') throw retryErr;
                    const status = retryErr.response?.status;
                    if ((status === 503 || !retryErr.response) && attempt < 2) {
                        await new Promise(r => setTimeout(r, 3000));
                        continue;
                    }
                    throw retryErr;
                }
            }

            if (this._generationMsgInterval) clearInterval(this._generationMsgInterval);

            this.setState({
                isGenerating: false,
                generationMessage: '',
                generatedAudio: response.data.modified_audio,
                generatedStats: {
                    segments_processed: response.data.segments_processed,
                    segments_changed: response.data.segments_changed,
                    ...response.data.stats,
                },
            });

        } catch (err) {
            if (this._generationMsgInterval) clearInterval(this._generationMsgInterval);
            if (err.name === 'CanceledError' || err.name === 'AbortError') return;
            console.error('Generation error:', err);
            this.setState({ isGenerating: false, generationMessage: '' });
            this.showToast(err.response?.data?.error || 'Failed to generate audio. Please try again.');
        }
    };

    // ──────────────────────────────────────────────────
    // Download (non-blocking)
    // ──────────────────────────────────────────────────

    handleDownload = async () => {
        const { generatedAudio } = this.state;
        if (!generatedAudio) return;

        try {
            const response = await fetch(`data:audio/wav;base64,${generatedAudio}`);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'wavecraft_modified.wav';
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error('Download error:', err);
            this.showToast('Download failed. Please try again.');
        }
    };

    handlePlayGenerated = () => {
        const { generatedAudio, isPlayingGenerated } = this.state;
        if (!generatedAudio || !this.audioRef.current) return;

        const audio = this.audioRef.current;

        if (isPlayingGenerated) {
            audio.pause();
            audio.currentTime = 0;
            this.setState({ isPlayingGenerated: false, downloadPlaybackProgress: 0 });
            return;
        }

        audio.src = `data:audio/wav;base64,${generatedAudio}`;
        audio.ontimeupdate = () => {
            if (audio.duration) {
                this.setState({ downloadPlaybackProgress: (audio.currentTime / audio.duration) * 100 });
            }
        };
        audio.onended = () => {
            this.setState({ isPlayingGenerated: false, downloadPlaybackProgress: 0 });
        };
        audio.play().catch((err) => {
            console.warn('Playback blocked:', err);
            this.showToast('Playback was blocked by the browser.', 'info');
        });
        this.setState({ isPlayingGenerated: true });
    };

    // ──────────────────────────────────────────────────
    // Merge speakers
    // ──────────────────────────────────────────────────

    getUniqueSpeakers() {
        const speakers = new Set();
        for (const convItem of this.state.conversations) {
            const { data } = this.getConvData(convItem);
            speakers.add(data.speaker);
        }
        return Array.from(speakers).sort();
    }

    handleMergeSpeakers = () => {
        const { mergeSource, mergeTarget, conversations } = this.state;
        if (!mergeSource || !mergeTarget || mergeSource === mergeTarget) return;

        const updated = conversations.map(convItem => {
            const key = Object.keys(convItem)[0];
            const data = convItem[key];
            if (data.speaker === mergeSource) {
                return { [key]: { ...data, speaker: mergeTarget } };
            }
            return convItem;
        });

        this.setState({
            conversations: updated,
            showMergeModal: false,
            mergeSource: null,
            mergeTarget: null,
        }, () => this._saveSession());

        this.showToast(`Merged Speaker ${mergeSource} into Speaker ${mergeTarget}`, 'info');
    };

    // ──────────────────────────────────────────────────
    // Stats
    // ──────────────────────────────────────────────────

    getEditStats() {
        let modified = 0, deleted = 0, inserted = 0, emotions = 0;

        for (const [, mods] of Object.entries(this.state.modifications)) {
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
                    onBlur={this.handleInsertBlur}
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
            // Inserted words before this position (afterIndex === i - 1)
            const insertsBefore = (mods.insertedWords || [])
                .filter(ins => ins.afterIndex === i - 1);
            for (let j = 0; j < insertsBefore.length; j++) {
                const ins = insertsBefore[j];
                if (ins.isParalinguistic) {
                    elements.push(
                        <span key={`para-${i}-${j}`} className="paralinguistic-tag">
                            {ins.text}
                        </span>
                    );
                } else {
                    elements.push(
                        <span key={`ins-${i}-${j}`} className="word inserted">
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
                    title={isDeleted ? 'Click to restore' : 'Click to delete \u2022 Right-click for emotion \u2022 Double-click to insert after'}
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
                        onBlur={this.handleInsertBlur}
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

            // Inserted words after the last word
            if (i === words.length - 1) {
                const insertsAfterLast = (mods.insertedWords || [])
                    .filter(ins => ins.afterIndex >= i);
                for (let j = 0; j < insertsAfterLast.length; j++) {
                    const ins = insertsAfterLast[j];
                    if (ins.isParalinguistic) {
                        elements.push(
                            <span key={`para-end-${j}`} className="paralinguistic-tag">
                                {ins.text}
                            </span>
                        );
                    } else {
                        elements.push(
                            <span key={`ins-end-${j}`} className="word inserted">
                                {ins.text}{' '}
                            </span>
                        );
                    }
                }
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
            generationMessage, generatedStats, toast,
            isPlayingGenerated, downloadPlaybackProgress,
            showMergeModal, mergeSource, mergeTarget,
        } = this.state;

        const stats = this.getEditStats();
        const hasEdits = stats.deleted > 0 || stats.inserted > 0 || stats.emotions > 0;
        const uniqueSpeakers = this.getUniqueSpeakers();

        return (
            <div className="main-page">
                <Header />

                {/* Page Header */}
                {conversations.length > 0 && (
                    <div className="page-header">
                        <h1>Transcript Editor</h1>
                        <p>Click words to delete, right-click for emotions, double-click to insert</p>
                        {uniqueSpeakers.length > 1 && (
                            <div className="page-header-actions">
                                <button
                                    className="merge-speakers-btn"
                                    onClick={() => this.setState({ showMergeModal: true, mergeSource: null, mergeTarget: null })}
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
                                        <circle cx="9" cy="7" r="4"/>
                                        <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
                                        <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                                    </svg>
                                    Merge Speakers
                                </button>
                            </div>
                        )}
                    </div>
                )}

                {/* Conversations */}
                <div className="main-content">
                    {conversations.length === 0 ? (
                        <div className="empty-state">
                            <div className="empty-state-icon">
                                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgba(0,0,0,0.3)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                    <line x1="12" y1="19" x2="12" y2="23"/>
                                    <line x1="8" y1="23" x2="16" y2="23"/>
                                </svg>
                            </div>
                            <h2>No transcript loaded</h2>
                            <p>Upload an audio file to get started.</p>
                            <Link to="/" className="btn-primary" style={{ textDecoration: 'none' }}>
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
                                            + Insert
                                        </button>
                                        <button
                                            className={`action-btn ${Object.keys(mods.emotions || {}).length > 0 ? 'active' : ''}`}
                                            onClick={(e) => this.handleWordRightClick(e, key, 0)}
                                            title="Add emotion to first word"
                                        >
                                            Emotion
                                        </button>

                                        {data.original?.speaker_audio && (
                                            <button
                                                className="play-btn"
                                                onClick={() => this.playSegmentAudio(key, data.original.speaker_audio)}
                                            >
                                                {this.state.currentlyPlaying === key ? 'Stop' : 'Play'}
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

                        {isGenerating ? (
                            <div className="generation-progress">
                                <div className="generation-spinner"></div>
                                <span className="generation-message">{generationMessage}</span>
                            </div>
                        ) : (
                            <button
                                className="generate-btn"
                                onClick={this.handleGenerate}
                                disabled={!hasEdits}
                            >
                                Generate Modified Audio
                            </button>
                        )}
                    </div>
                )}

                {/* Download Panel */}
                {generatedAudio && (
                    <div className="download-panel">
                        <div className="download-card">
                            <div className="download-checkmark">
                                <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                                    <circle cx="24" cy="24" r="23" stroke="#0a0a0a" strokeWidth="2"/>
                                    <path d="M15 24L21 30L33 18" stroke="#0a0a0a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                            </div>
                            <h3>Audio Generated</h3>
                            {generatedStats && (
                                <div className="download-stats">
                                    <span>{generatedStats.segments_processed || 0} segments processed</span>
                                    <span className="download-stats-separator">/</span>
                                    <span>{generatedStats.segments_changed || 0} changed</span>
                                </div>
                            )}
                            <p>
                                Play to preview or download the file.
                            </p>

                            {/* Playback progress bar */}
                            {isPlayingGenerated && (
                                <div className="download-playback-bar">
                                    <div
                                        className="download-playback-fill"
                                        style={{ width: `${downloadPlaybackProgress}%` }}
                                    />
                                </div>
                            )}

                            <div className="download-actions">
                                <button className="btn-secondary" onClick={this.handlePlayGenerated}>
                                    {isPlayingGenerated ? 'Stop' : 'Listen'}
                                </button>
                                <button className="btn-primary" onClick={this.handleDownload}>
                                    Download WAV
                                </button>
                            </div>
                            <button
                                className="btn-back-to-editor"
                                onClick={() => {
                                    const audio = this.audioRef.current;
                                    if (audio) {
                                        audio.pause();
                                        audio.currentTime = 0;
                                    }
                                    this.setState({
                                        generatedAudio: null,
                                        currentlyPlaying: null,
                                        isPlayingGenerated: false,
                                        downloadPlaybackProgress: 0,
                                    });
                                }}
                            >
                                Back to Editor
                            </button>
                        </div>
                    </div>
                )}

                {/* Merge Speakers Modal */}
                {showMergeModal && (
                    <div className="merge-modal-backdrop" onClick={() => this.setState({ showMergeModal: false })}>
                        <div className="merge-modal" onClick={(e) => e.stopPropagation()}>
                            <h3>Merge Speakers</h3>
                            <p>Select two speakers to merge. The first will be relabeled as the second.</p>

                            <div style={{ marginBottom: '12px' }}>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 500 }}>
                                    Merge this speaker...
                                </div>
                                <div className="merge-speaker-list">
                                    {uniqueSpeakers.map(s => (
                                        <div
                                            key={`src-${s}`}
                                            className={`merge-speaker-option ${mergeSource === s ? 'selected' : ''}`}
                                            onClick={() => this.setState({ mergeSource: s })}
                                        >
                                            <div className={`speaker-avatar ${this.getSpeakerClass(s)}`}>{s}</div>
                                            <span>Speaker {s}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {mergeSource && (
                                <div style={{ marginBottom: '20px' }}>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '6px', fontWeight: 500 }}>
                                        ...into this speaker
                                    </div>
                                    <div className="merge-speaker-list">
                                        {uniqueSpeakers.filter(s => s !== mergeSource).map(s => (
                                            <div
                                                key={`tgt-${s}`}
                                                className={`merge-speaker-option ${mergeTarget === s ? 'selected' : ''}`}
                                                onClick={() => this.setState({ mergeTarget: s })}
                                            >
                                                <div className={`speaker-avatar ${this.getSpeakerClass(s)}`}>{s}</div>
                                                <span>Speaker {s}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div className="merge-modal-actions">
                                <button className="btn-cancel" onClick={() => this.setState({ showMergeModal: false })}>
                                    Cancel
                                </button>
                                <button
                                    className="btn-merge"
                                    disabled={!mergeSource || !mergeTarget}
                                    onClick={this.handleMergeSpeakers}
                                >
                                    Merge
                                </button>
                            </div>
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

                {/* Toast notification */}
                {toast && (
                    <div className={`toast-notification toast-${toast.type || 'error'}`}>
                        {toast.message}
                    </div>
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
