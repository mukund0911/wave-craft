import React, { useEffect, useRef, useCallback } from 'react';

const ParticleBackground = () => {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const particlesRef = useRef([]);
    const timeRef = useRef(0);

    const initParticles = useCallback(() => {
        const particles = [];
        const PARTICLE_COUNT = 3000;

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            // Create orbital/arc distribution
            const angle = Math.random() * Math.PI * 2;
            const radius = 0.05 + Math.random() * 0.8;
            const bandOffset = Math.sin(angle * 3 + Math.random() * 2) * 0.25;

            const isFlowing = Math.random() > 0.25;

            particles.push({
                x: Math.cos(angle) * radius + bandOffset,
                y: Math.sin(angle) * radius + bandOffset * 0.6,
                baseX: Math.cos(angle) * radius + bandOffset,
                baseY: Math.sin(angle) * radius + bandOffset * 0.6,
                vx: isFlowing ? Math.cos(angle + Math.PI / 2) * (0.00015 + Math.random() * 0.0004) : 0,
                vy: isFlowing ? Math.sin(angle + Math.PI / 2) * (0.00015 + Math.random() * 0.0004) : 0,
                // Larger sizes
                size: isFlowing ? 2.5 + Math.random() * 3 : 2 + Math.random() * 1.5,
                isFlowing: isFlowing,
                phase: Math.random() * Math.PI * 2
            });
        }

        return particles;
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const resize = () => {
            const dpr = window.devicePixelRatio || 1;
            canvas.width = window.innerWidth * dpr;
            canvas.height = window.innerHeight * dpr;
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            particlesRef.current = initParticles();
        };

        resize();
        window.addEventListener('resize', resize);

        const animate = () => {
            const width = window.innerWidth;
            const height = window.innerHeight;
            const centerX = width / 2;
            const centerY = height / 2;
            const scale = Math.min(width, height) * 0.95;

            timeRef.current += 0.016;
            const time = timeRef.current;

            // Clear with white
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);

            // Draw particles
            particlesRef.current.forEach((p) => {
                if (p.isFlowing) {
                    p.baseX += p.vx;
                    p.baseY += p.vy;

                    const dist = Math.sqrt(p.baseX * p.baseX + p.baseY * p.baseY);
                    if (dist > 1.0) {
                        const angle = Math.atan2(p.baseY, p.baseX) + Math.PI;
                        p.baseX = Math.cos(angle) * 0.12;
                        p.baseY = Math.sin(angle) * 0.12;
                    }

                    const waveX = Math.sin(p.baseX * 5 + time * 1.0 + p.phase) * 0.02;
                    const waveY = Math.cos(p.baseY * 5 + time * 0.8 + p.phase) * 0.02;

                    const orbitAngle = Math.atan2(p.baseY, p.baseX);
                    const orbitDist = Math.sqrt(p.baseX * p.baseX + p.baseY * p.baseY);
                    const orbitalSpeed = 0.05 / (orbitDist + 0.1);
                    const newAngle = orbitAngle + time * orbitalSpeed;
                    const orbX = (Math.cos(newAngle) * orbitDist - p.baseX) * 0.03;
                    const orbY = (Math.sin(newAngle) * orbitDist - p.baseY) * 0.03;

                    p.x = p.baseX + waveX + orbX;
                    p.y = p.baseY + waveY + orbY;
                } else {
                    p.x = p.baseX;
                    p.y = p.baseY;
                }

                const screenX = centerX + p.x * scale;
                const screenY = centerY + p.y * scale;

                ctx.beginPath();
                ctx.arc(screenX, screenY, p.size, 0, Math.PI * 2);

                if (p.isFlowing) {
                    // BRIGHT CYAN - very visible
                    ctx.fillStyle = 'rgba(0, 180, 255, 0.85)';
                } else {
                    // VISIBLE GRAY-BLUE
                    ctx.fillStyle = 'rgba(120, 150, 180, 0.65)';
                }

                ctx.fill();
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resize);
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [initParticles]);

    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            zIndex: -1,
            overflow: 'hidden'
        }}>
            <canvas
                ref={canvasRef}
                style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%'
                }}
            />
            {/* Blue vignette overlay */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                background: `
                    radial-gradient(ellipse at center, 
                        rgba(255, 255, 255, 0) 0%, 
                        rgba(255, 255, 255, 0) 20%,
                        rgba(180, 215, 255, 0.3) 50%,
                        rgba(60, 140, 255, 0.4) 100%
                    )
                `,
                pointerEvents: 'none'
            }} />
        </div>
    );
};

export default ParticleBackground;
