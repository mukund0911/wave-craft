import React, { Component } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import MainPage from './components/MainPage';
import './App.css';

class ErrorBoundary extends Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('ErrorBoundary caught:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '100vh',
                    fontFamily: "'Inter', -apple-system, sans-serif",
                    color: '#0a0a0a',
                    padding: '24px',
                    textAlign: 'center',
                }}>
                    <h2 style={{
                        fontFamily: "'Space Grotesk', sans-serif",
                        fontSize: '1.4rem',
                        fontWeight: 700,
                        marginBottom: '12px',
                    }}>
                        Something went wrong
                    </h2>
                    <p style={{
                        color: '#555',
                        fontSize: '0.95rem',
                        marginBottom: '24px',
                        maxWidth: '400px',
                    }}>
                        An unexpected error occurred. Please try refreshing the page.
                    </p>
                    <button
                        onClick={() => { window.location.href = '/'; }}
                        style={{
                            padding: '12px 28px',
                            background: '#0a0a0a',
                            color: '#fff',
                            border: 'none',
                            borderRadius: '100px',
                            fontFamily: "'Inter', sans-serif",
                            fontSize: '0.9rem',
                            fontWeight: 600,
                            cursor: 'pointer',
                        }}
                    >
                        Go Home
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

function App() {
    return (
        <Router>
            <div className="App">
                <ErrorBoundary>
                    <Routes>
                        <Route path="/" element={<LandingPage />} />
                        <Route path="/result" element={<MainPage />} />
                    </Routes>
                </ErrorBoundary>
            </div>
        </Router>
    );
}

export default App;
