import React from 'react';
import Header from './Header';
import UploadButton from './UploadButton';
import '../styles/LandingPage.css';

function LandingPage() {
    return (
        <div className="landing-page">
            <div className='content'>
                <Header />

                <div className='description'>
                    Remix and Edit Audio with Ease
                    <span>Transform your audio files with AI-powered speech editing</span>
                </div>
                <UploadButton />
            </div>
        </div>
    );
}

export default LandingPage;
