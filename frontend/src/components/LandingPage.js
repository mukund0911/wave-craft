import React from 'react';
import Header from './Header';
import UploadButton from './UploadButton';
import '../styles/LandingPage.css';

function LandingPage() {
    return (
        <div className="landing-page">
            <Header />
            <UploadButton />
        </div>
    );
}

export default LandingPage;
