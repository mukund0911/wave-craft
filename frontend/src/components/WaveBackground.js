import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

/**
 * Premium 3D Wave Background
 *
 * Dark scene with flowing gradient particles (navy → cyan → purple).
 * Smooth sinusoidal wave motion with glow effects.
 * Used as the hero section background.
 */
function WaveBackground() {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 8, 20);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0); // Transparent background
    container.appendChild(renderer.domElement);

    // Create wave particles
    const particleCount = 8000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    // Color palette
    const colorPalette = [
      new THREE.Color(0x3b82f6), // Blue
      new THREE.Color(0x06b6d4), // Cyan
      new THREE.Color(0x8b5cf6), // Purple
      new THREE.Color(0x1e40af), // Deep blue
    ];

    const gridWidth = 100;
    const gridDepth = 50;
    const cols = Math.ceil(Math.sqrt(particleCount * (gridWidth / gridDepth)));
    const rows = Math.ceil(particleCount / cols);

    for (let i = 0; i < particleCount; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);

      const x = (col / cols - 0.5) * gridWidth + (Math.random() - 0.5) * 0.8;
      const z = (row / rows - 0.5) * gridDepth + (Math.random() - 0.5) * 0.8;
      const y = 0;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      // Color based on position (gradient across the wave)
      const t = Math.max(0, Math.min(1, (x + gridWidth / 2) / gridWidth));
      const scaledT = t * (colorPalette.length - 1);
      const colorIndex = Math.min(Math.floor(scaledT), colorPalette.length - 2);
      const colorFrac = scaledT - colorIndex;
      const c1 = colorPalette[colorIndex];
      const c2 = colorPalette[colorIndex + 1];
      const color = c1.clone().lerp(c2, colorFrac);

      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;

      sizes[i] = 0.06 + Math.random() * 0.08;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    // Custom shader material for glow effect
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: renderer.getPixelRatio() },
      },
      vertexShader: `
                attribute float size;
                varying vec3 vColor;
                uniform float uTime;
                uniform float uPixelRatio;

                void main() {
                    vColor = color;

                    vec3 pos = position;

                    // Multi-layered wave
                    float wave1 = sin(pos.x * 0.15 + uTime * 0.6) * 2.0;
                    float wave2 = sin(pos.z * 0.2 + uTime * 0.4) * 1.2;
                    float wave3 = cos(pos.x * 0.08 + pos.z * 0.1 + uTime * 0.3) * 1.5;
                    pos.y = wave1 + wave2 + wave3;

                    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                    gl_PointSize = size * uPixelRatio * 80.0 / -mvPosition.z;
                }
            `,
      fragmentShader: `
                varying vec3 vColor;

                void main() {
                    // Soft circle with glow
                    float dist = length(gl_PointCoord - vec2(0.5));
                    if (dist > 0.5) discard;

                    float alpha = 1.0 - smoothstep(0.1, 0.5, dist);
                    alpha *= 0.7; // Overall transparency

                    gl_FragColor = vec4(vColor, alpha);
                }
            `,
      transparent: true,
      vertexColors: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Add subtle ambient light
    const ambientLight = new THREE.AmbientLight(0x3b82f6, 0.1);
    scene.add(ambientLight);

    // Animation
    let animationId;
    const clock = new THREE.Clock();

    const animate = () => {
      animationId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      material.uniforms.uTime.value = elapsed;

      // Gentle camera sway
      camera.position.x = Math.sin(elapsed * 0.1) * 3;
      camera.position.y = 8 + Math.sin(elapsed * 0.15) * 1;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={mountRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'absolute',
        top: 0,
        left: 0,
      }}
    />
  );
}

export default WaveBackground;
