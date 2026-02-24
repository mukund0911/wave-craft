/**
 * FluidBackground — Antigravity/Medusae-style instanced particle field
 *
 * Uses React Three Fiber + instanced mesh + custom GLSL shaders.
 * Based on the BreathDearMedusae approach by ewohlken2.
 *
 * Key effects:
 *   - 5500 instanced plane quads on a grid
 *   - Vertex shader: alive bidirectional flow, breathing jellyfish halo
 *     around cursor, outer oscillation, rotation toward cursor
 *   - Fragment shader: soft squircle particles with position-based color mixing
 *   - Mouse reactive with smooth drag interpolation
 */
import React, { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

/* ─── Configuration ─── */
const CONFIG = {
    cursor: {
        radius: 0.065,
        strength: 3,
        dragFactor: 0.015,
    },
    halo: {
        outerOscFrequency: 2.6,
        outerOscAmplitude: 0.76,
        radiusBase: 2.4,
        radiusAmplitude: 0.5,
        shapeAmplitude: 0.75,
        rimWidth: 1.8,
        outerStartOffset: 0.4,
        outerEndOffset: 2.2,
        scaleX: 1.3,
        scaleY: 1,
    },
    particles: {
        baseSize: 0.016,
        activeSize: 0.044,
        blobScaleX: 1,
        blobScaleY: 0.6,
        rotationSpeed: 0.1,
        rotationJitter: 0.2,
        cursorFollowStrength: 1,
        oscillationFactor: 1,
        // Monochrome — all black
        colorBase: '#ffffff',
        colorOne: '#000000',
        colorTwo: '#1a1a1a',
        colorThree: '#0d0d0d',
    },
    background: {
        color: '#ffffff',
    },
};

/* ─── Vertex Shader ─── */
const vertexShader = `
    uniform float uTime;
    uniform vec2 uMouse;
    uniform float uOuterOscFrequency;
    uniform float uOuterOscAmplitude;
    uniform float uHaloRadiusBase;
    uniform float uHaloRadiusAmplitude;
    uniform float uHaloShapeAmplitude;
    uniform float uHaloRimWidth;
    uniform float uHaloOuterStartOffset;
    uniform float uHaloOuterEndOffset;
    uniform float uHaloScaleX;
    uniform float uHaloScaleY;
    uniform float uParticleBaseSize;
    uniform float uParticleActiveSize;
    uniform float uBlobScaleX;
    uniform float uBlobScaleY;
    uniform float uParticleRotationSpeed;
    uniform float uParticleRotationJitter;
    uniform float uParticleOscillationFactor;
    varying vec2 vUv;
    varying float vSize;
    varying vec2 vPos;

    attribute vec3 aOffset;
    attribute float aRandom;

    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    float noise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    }

    void main() {
        vUv = uv;

        // 1. ALIVE FLOW — slow bidirectional drift
        vec3 pos = aOffset;
        float driftSpeed = uTime * 0.15;
        float dx = sin(driftSpeed + pos.y * 0.5) + sin(driftSpeed * 0.5 + pos.y * 2.0);
        float dy = cos(driftSpeed + pos.x * 0.5) + cos(driftSpeed * 0.5 + pos.x * 2.0);
        pos.x += dx * 0.25;
        pos.y += dy * 0.25;

        // 2. JELLYFISH HALO — breathing ring around cursor
        vec2 relToMouse = pos.xy - uMouse;
        vec2 haloScale = max(vec2(uHaloScaleX, uHaloScaleY), vec2(0.0001));
        float distFromMouse = length(relToMouse / haloScale);
        vec2 dirToMouse = normalize(relToMouse + vec2(0.0001, 0.0));
        float shapeFactor = noise(dirToMouse * 2.0 + vec2(0.0, uTime * 0.1));

        float breathCycle = sin(uTime * 0.8);
        float baseRadius = uHaloRadiusBase + breathCycle * uHaloRadiusAmplitude;
        float currentRadius = baseRadius + (shapeFactor * uHaloShapeAmplitude);
        float dist = distFromMouse;
        float rimInfluence = smoothstep(uHaloRimWidth, 0.0, abs(dist - currentRadius));

        vec2 pushDir = normalize(relToMouse + vec2(0.0001, 0.0));
        float pushAmt = (breathCycle * 0.5 + 0.5) * 0.5;
        pos.xy += pushDir * pushAmt * rimInfluence;
        pos.z += rimInfluence * 0.3 * sin(uTime);

        // 3. OUTER OSCILLATION
        float outerInfluence = smoothstep(baseRadius + uHaloOuterStartOffset, baseRadius + uHaloOuterEndOffset, dist);
        float outerOsc = sin(uTime * uOuterOscFrequency + pos.x * 0.6 + pos.y * 0.6);
        pos.xy += normalize(relToMouse + vec2(0.0001, 0.0)) * outerOsc * uOuterOscAmplitude * outerInfluence;

        // 4. SIZE & SCALE
        float baseSize = uParticleBaseSize + (sin(uTime + pos.x) * 0.003);
        float activeSize = uParticleActiveSize;
        float currentScale = baseSize + (rimInfluence * activeSize);
        float stretch = rimInfluence * 0.02;

        vec3 transformed = position;
        transformed.x *= (currentScale + stretch) * uBlobScaleX;
        transformed.y *= currentScale * uBlobScaleY;

        vSize = rimInfluence;
        vPos = pos.xy;

        // 5. ROTATION — particles orient toward cursor with jitter
        float dirLen = max(length(relToMouse), 0.0001);
        vec2 dir = relToMouse / dirLen;
        float oscPhase = aRandom * 6.28318530718;
        float osc = 0.5 + 0.5 * sin(
            uTime * (0.25 + uParticleOscillationFactor * 0.35) + oscPhase
        );
        float speedScale = mix(0.55, 1.35, osc) * (0.8 + uParticleOscillationFactor * 0.2);
        float jitterScale = mix(0.7, 1.45, osc) * (0.85 + uParticleOscillationFactor * 0.15);
        float jitter = sin(
            uTime * uParticleRotationSpeed * speedScale + pos.x * 0.35 + pos.y * 0.35
        ) * (uParticleRotationJitter * jitterScale);
        vec2 perp = vec2(-dir.y, dir.x);
        vec2 jitteredDir = normalize(dir + perp * jitter);
        mat2 rot = mat2(jitteredDir.x, jitteredDir.y, -jitteredDir.y, jitteredDir.x);
        transformed.xy = rot * transformed.xy;

        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos + transformed, 1.0);
    }
`;

/* ─── Fragment Shader ─── */
const fragmentShader = `
    uniform float uTime;
    uniform vec3 uParticleColorBase;
    uniform vec3 uParticleColorOne;
    uniform vec3 uParticleColorTwo;
    uniform vec3 uParticleColorThree;
    varying vec2 vUv;
    varying float vSize;
    varying vec2 vPos;

    void main() {
        vec2 center = vec2(0.5);
        vec2 pos = abs(vUv - center) * 2.0;

        // Pixelated square shape — hard edges
        float d = max(pos.x, pos.y);
        float alpha = 1.0 - step(0.92, d);
        if (alpha < 0.01) discard;

        vec3 base = uParticleColorBase;
        vec3 cOne = uParticleColorOne;
        vec3 cTwo = uParticleColorTwo;
        vec3 cThree = uParticleColorThree;

        float t = uTime * 1.2;
        float p1 = sin(vPos.x * 0.8 + t);
        float p2 = sin(vPos.y * 0.8 + t * 0.8 + p1);

        vec3 activeColor = mix(cOne, cTwo, p1 * 0.5 + 0.5);
        activeColor = mix(activeColor, cThree, p2 * 0.5 + 0.5);

        vec3 finalColor = mix(base, activeColor, smoothstep(0.1, 0.8, vSize));
        float finalAlpha = alpha * mix(0.35, 0.9, vSize);

        gl_FragColor = vec4(finalColor, finalAlpha);
    }
`;

/* ─── Particles Component ─── */
function Particles() {
    const meshRef = useRef();
    const { viewport } = useThree();

    const countX = 100;
    const countY = 55;
    const count = countX * countY;

    const geometry = useMemo(() => new THREE.PlaneGeometry(1, 1), []);

    const uniforms = useMemo(
        () => ({
            uTime: { value: 0 },
            uMouse: { value: new THREE.Vector2(0, 0) },
            uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
            uOuterOscFrequency: { value: CONFIG.halo.outerOscFrequency },
            uOuterOscAmplitude: { value: CONFIG.halo.outerOscAmplitude },
            uHaloRadiusBase: { value: CONFIG.halo.radiusBase },
            uHaloRadiusAmplitude: { value: CONFIG.halo.radiusAmplitude },
            uHaloShapeAmplitude: { value: CONFIG.halo.shapeAmplitude },
            uHaloRimWidth: { value: CONFIG.halo.rimWidth },
            uHaloOuterStartOffset: { value: CONFIG.halo.outerStartOffset },
            uHaloOuterEndOffset: { value: CONFIG.halo.outerEndOffset },
            uHaloScaleX: { value: CONFIG.halo.scaleX },
            uHaloScaleY: { value: CONFIG.halo.scaleY },
            uParticleBaseSize: { value: CONFIG.particles.baseSize },
            uParticleActiveSize: { value: CONFIG.particles.activeSize },
            uBlobScaleX: { value: CONFIG.particles.blobScaleX },
            uBlobScaleY: { value: CONFIG.particles.blobScaleY },
            uParticleRotationSpeed: { value: CONFIG.particles.rotationSpeed },
            uParticleRotationJitter: { value: CONFIG.particles.rotationJitter },
            uParticleOscillationFactor: { value: CONFIG.particles.oscillationFactor },
            uParticleColorBase: { value: new THREE.Color(CONFIG.particles.colorBase) },
            uParticleColorOne: { value: new THREE.Color(CONFIG.particles.colorOne) },
            uParticleColorTwo: { value: new THREE.Color(CONFIG.particles.colorTwo) },
            uParticleColorThree: { value: new THREE.Color(CONFIG.particles.colorThree) },
        }),
        []
    );

    const material = useMemo(
        () =>
            new THREE.ShaderMaterial({
                uniforms,
                vertexShader,
                fragmentShader,
                transparent: true,
                depthWrite: false,
            }),
        [uniforms]
    );

    // Set up instanced attributes
    useEffect(() => {
        if (!meshRef.current) return;

        const offsets = new Float32Array(count * 3);
        const randoms = new Float32Array(count);

        const gridWidth = 30;
        const gridHeight = 16;
        const jitter = 0.25;

        let i = 0;
        for (let y = 0; y < countY; y++) {
            for (let x = 0; x < countX; x++) {
                const u = x / (countX - 1);
                const v = y / (countY - 1);

                let px = (u - 0.5) * gridWidth;
                let py = (v - 0.5) * gridHeight;

                px += (Math.random() - 0.5) * jitter;
                py += (Math.random() - 0.5) * jitter;

                offsets[i * 3] = px;
                offsets[i * 3 + 1] = py;
                offsets[i * 3 + 2] = 0;

                randoms[i] = Math.random();
                i++;
            }
        }

        meshRef.current.geometry.setAttribute(
            'aOffset',
            new THREE.InstancedBufferAttribute(offsets, 3)
        );
        meshRef.current.geometry.setAttribute(
            'aRandom',
            new THREE.InstancedBufferAttribute(randoms, 1)
        );
    }, [count, countX, countY]);

    // Track mouse hover state
    const hovering = useRef(true);
    const globalPointer = useRef(null);

    useEffect(() => {
        const handleLeave = () => (hovering.current = false);
        const handleEnter = () => (hovering.current = true);
        document.body.addEventListener('mouseleave', handleLeave);
        document.body.addEventListener('mouseenter', handleEnter);
        return () => {
            document.body.removeEventListener('mouseleave', handleLeave);
            document.body.removeEventListener('mouseenter', handleEnter);
        };
    }, []);

    useEffect(() => {
        const handlePointerMove = (event) => {
            const x = (event.clientX / window.innerWidth) * 2 - 1;
            const y = -(event.clientY / window.innerHeight) * 2 + 1;
            globalPointer.current = { x, y };
        };
        window.addEventListener('pointermove', handlePointerMove);
        return () => window.removeEventListener('pointermove', handlePointerMove);
    }, []);

    // Animation loop
    useFrame((state) => {
        const { clock, pointer } = state;
        material.uniforms.uTime.value = clock.getElapsedTime();

        let targetX = null;
        let targetY = null;

        if (hovering.current) {
            const pointerSource = globalPointer.current ?? pointer;
            const baseX = (pointerSource.x * viewport.width) / 2;
            const baseY = (pointerSource.y * viewport.height) / 2;
            const t = clock.getElapsedTime();
            const jitterRadius =
                Math.min(viewport.width, viewport.height) * CONFIG.cursor.radius;
            const jitterX = (Math.sin(t * 0.35) + Math.sin(t * 0.77 + 1.2)) * 0.5;
            const jitterY = (Math.cos(t * 0.31) + Math.sin(t * 0.63 + 2.4)) * 0.5;
            const followStrength = CONFIG.particles.cursorFollowStrength;
            targetX = (baseX + jitterX * jitterRadius * CONFIG.cursor.strength) * followStrength;
            targetY = (baseY + jitterY * jitterRadius * CONFIG.cursor.strength) * followStrength;
        }

        const current = material.uniforms.uMouse.value;
        const dragFactor = CONFIG.cursor.dragFactor;

        if (targetX !== null && targetY !== null) {
            current.x += (targetX - current.x) * dragFactor;
            current.y += (targetY - current.y) * dragFactor;
        }
    });

    return <instancedMesh ref={meshRef} args={[geometry, material, count]} />;
}

/* ─── Main Component ─── */
function FluidBackground() {
    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 0,
            }}
        >
            <Canvas camera={{ position: [0, 0, 5] }}>
                <color attach="background" args={[CONFIG.background.color]} />
                <Particles />
            </Canvas>
        </div>
    );
}

export default FluidBackground;
