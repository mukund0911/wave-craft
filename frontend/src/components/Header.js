import React from 'react';
import '../styles/Header.css';
import giticon from './github-white-icon.svg';

function Header() {
    return (
        <div className="header">
            <h1>Wavecrafter</h1>
            <a className='giticon' href="https://github.com/mukund0911/wave-craft" target="_blank" rel="noopener noreferrer">
                <img src={giticon} alt="giticon" /> 
            </a>
            
        </div>
    );
}

export default Header;
