import React from 'react';
import '../styles/Header.css';
import giticon from './github_icon.svg';

function Header() {
    return (
        <div className="header">
            <h1>Wavecraft</h1>
            <a className='giticon' href="https://www.example.com" target="_blank" rel="noopener noreferrer">
                <img src={giticon} alt="giticon" />
            </a>
        </div>
    );
}

export default Header;
