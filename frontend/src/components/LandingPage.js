import React from 'react';
import Header from './Header';
import UploadButton from './UploadButton';
import '../styles/LandingPage.css';

function LandingPage() {
    return (
        <div className="landing-page">
            <Header />
            <div class='description'>
                Remix and Edit Audio with Ease.
            </div>
            <UploadButton />
        </div>
    );
}

export default LandingPage;
