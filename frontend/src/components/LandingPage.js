import React from 'react';
import Header from './Header';
import UploadButton from './UploadButton';
import FluidBackground from './FluidBackground';
import '../styles/LandingPage.css';

function LandingPage() {
    return (
        <div className="landing-page">
            {/* Pixelated fluid background */}
            <FluidBackground />

            {/* Navigation */}
            <Header />

            {/* Hero Section */}
            <section className="hero">
                <h1 className="hero-title">
                    Edit Speech by{' '}
                    <span className="hero-title-accent">Editing Text</span>
                </h1>

                <p className="hero-subtitle">
                    Upload multi-speaker audio, edit transcripts, add emotions —
                    generate new audio that preserves each speaker's voice.
                </p>

                <div className="upload-zone">
                    <UploadButton />
                </div>
            </section>

            {/* Features Section */}
            <section className="features" id="features">
                <div className="features-header">
                    <h2>Built for Precision</h2>
                    <p>
                        Everything you need to transform, edit, and enhance speech audio.
                    </p>
                </div>

                <div className="features-grid">
                    <div className="feature-card">
                        <span className="feature-icon">🎤</span>
                        <h3>Multi-Speaker Diarization</h3>
                        <p>
                            Automatically identifies and separates speakers with
                            word-level timestamps using WhisperX.
                        </p>
                    </div>

                    <div className="feature-card">
                        <span className="feature-icon">🔄</span>
                        <h3>Text-Based Voice Cloning</h3>
                        <p>
                            Edit the transcript — delete, insert, or modify words.
                            Chatterbox regenerates audio in the original voice.
                        </p>
                    </div>

                    <div className="feature-card">
                        <span className="feature-icon">😊</span>
                        <h3>Emotion Control</h3>
                        <p>
                            Tag words with emotions and paralinguistic cues
                            like [laugh], [sigh], or [gasp]. Adjust intensity.
                        </p>
                    </div>

                    <div className="feature-card">
                        <span className="feature-icon">🚀</span>
                        <h3>Zero-Shot Cloning</h3>
                        <p>
                            Clone any voice from just 5 seconds of reference audio.
                            95%+ voice similarity with Chatterbox TTS.
                        </p>
                    </div>

                    <div className="feature-card">
                        <span className="feature-icon">🌐</span>
                        <h3>23 Languages</h3>
                        <p>
                            Multilingual speech editing with cross-language voice
                            transfer. Edit in one language, generate in another.
                        </p>
                    </div>

                    <div className="feature-card">
                        <span className="feature-icon">🔒</span>
                        <h3>Fully Local</h3>
                        <p>
                            All processing runs locally — no cloud APIs, no data
                            leaves your machine. Zero per-request costs.
                        </p>
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section className="how-it-works" id="how-it-works">
                <div className="how-it-works-header">
                    <h2>Three Simple Steps</h2>
                    <p>From audio to perfected speech in minutes.</p>
                </div>

                <div className="steps">
                    <div className="step">
                        <div className="step-number">1</div>
                        <h3>Upload Audio</h3>
                        <p>
                            Drop any audio file — multi-speaker conversations,
                            podcasts, interviews. We handle the rest.
                        </p>
                    </div>

                    <div className="step">
                        <div className="step-number">2</div>
                        <h3>Edit Transcript</h3>
                        <p>
                            Review the auto-generated transcript. Click to delete
                            words, type to add new ones, tag emotions.
                        </p>
                    </div>

                    <div className="step">
                        <div className="step-number">3</div>
                        <h3>Generate Audio</h3>
                        <p>
                            One click generates the modified audio with
                            each speaker's original voice preserved.
                        </p>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="footer">
                <p>
                    WaveCrafter — Open source speech editing.{' '}
                    <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                        View on GitHub
                    </a>
                </p>
            </footer>
        </div>
    );
}

export default LandingPage;
