import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const WaveBackground = () => {
  const mountRef = useRef(null);

  useEffect(() => {
    console.log('WaveBackground mounted');
    // Variables
    const SEPARATION = 100, AMOUNTX = 100, AMOUNTY = 50;
    let container;
    let camera, scene, renderer;
    let particles, particlePositions, count = 0;

    // Initialize
    const init = () => {
      container = mountRef.current;

      // Camera
      camera = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        1,
        10000
      );
      camera.position.set(800, 100, 800);
      camera.lookAt(new THREE.Vector3(0, 0, 0));

      // Scene
      scene = new THREE.Scene();

      // Particles
      const numParticles = AMOUNTX * AMOUNTY;

      const positions = new Float32Array(numParticles * 3);
      const scales = new Float32Array(numParticles);

      let i = 0, j = 0;
      for (let ix = 0; ix < AMOUNTX; ix++) {
        for (let iy = 0; iy < AMOUNTY; iy++) {
          positions[i] = ix * SEPARATION - ((AMOUNTX * SEPARATION) / 2); // x
          positions[i + 1] = 0; // y
          positions[i + 2] = iy * SEPARATION - ((AMOUNTY * SEPARATION) / 2); // z

          scales[j] = 1;

          i += 3;
          j++;
        }
      }

      // Create BufferGeometry
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('scale', new THREE.BufferAttribute(scales, 1));

      // Create a circular texture
      const texture = createCircleTexture();

      // Particle material with circular texture and smaller size
      const material = new THREE.PointsMaterial({
        color: 0x757575, // Light Gray particles
        size: 7, // Reduced size
        sizeAttenuation: true,
        map: texture,
        alphaTest: 0.5,
        transparent: true,
      });

      // Create Points
      particles = new THREE.Points(geometry, material);
      scene.add(particles);

      // Renderer
      renderer = new THREE.WebGLRenderer({ alpha: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(window.innerWidth, window.innerHeight);
      container.appendChild(renderer.domElement);

      // Event Listener
      window.addEventListener('resize', onWindowResize, false);
    };

    // Create a circular texture
    const createCircleTexture = () => {
      const size = 64;

      // Create canvas
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;

      // Get context
      const context = canvas.getContext('2d');

      // Draw circle
      context.beginPath();
      context.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2, false);
      context.closePath();

      // Fill style
      context.fillStyle = 'white';
      context.fill();

      // Create texture
      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      texture.minFilter = THREE.LinearFilter;
      texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;

      return texture;
    };

    // Event Handler
    const onWindowResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();

      renderer.setSize(window.innerWidth, window.innerHeight);
    };

    // Animation
    const animate = () => {
      requestAnimationFrame(animate);
      render();
    };

    const render = () => {
      renderer.setClearColor(0xeeeeee, 1);

      const positions = particles.geometry.attributes.position.array;
      const scales = particles.geometry.attributes.scale.array;

      let i = 0, j = 0;
      for (let ix = 0; ix < AMOUNTX; ix++) {
        for (let iy = 0; iy < AMOUNTY; iy++) {

          positions[i + 1] = (Math.sin((ix + count) * 0.3) * 20) +
                             (Math.sin((iy + count) * 0.5) * 20);

          scales[j] = (Math.sin((ix + count) * 0.3) + 1) * 2 +
                      (Math.sin((iy + count) * 0.5) + 1) * 2; // Reduced multiplier

          i += 3;
          j++;
        }
      }

      particles.geometry.attributes.position.needsUpdate = true;
      particles.geometry.attributes.scale.needsUpdate = true;

      renderer.render(scene, camera);
      count += 0.1;
    };

    // Start
    init();
    animate();

    // Cleanup on unmount
    return () => {
      container.removeChild(renderer.domElement);
      window.removeEventListener('resize', onWindowResize, false);
    };
  }, []);

  // Style to make the canvas cover the entire background
  const style = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    zIndex: 0, // Ensure it appears behind content
    overflow: 'hidden',
  };

  return <div ref={mountRef} style={style} />;
};

export default WaveBackground;
