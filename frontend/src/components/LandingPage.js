import React from 'react';
import Header from './Header';
import UploadButton from './UploadButton';
// import WaveBackground from './WaveBackground';
import '../styles/LandingPage.css';

function LandingPage() {
    return (
        <div className="landing-page">
            {/* <WaveBackground /> */}
            <div className='content'>
                <Header />
            
                <div className='description'>
                    Remix and Edit Audio with Ease.
                </div>
                <UploadButton />
            </div>
        </div>
    );
}

export default LandingPage;
