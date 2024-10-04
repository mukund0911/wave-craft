import React from 'react';
import { useLocation } from 'react-router-dom';
import '../styles/MainPage.css';

function MainPage() {
    // Use useLocation to get the passed state
    const location = useLocation();
    const { prediction } = location.state || { };

    return (
        <div className="prediction-result">
            <h2>Audio Analysis Result</h2>
            <p>The uploaded audio is classified as: <strong>{prediction}</strong></p>
        </div>
    );
}

export default MainPage;
