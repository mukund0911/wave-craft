"""
Audio Quality Analysis Utilities
Provides functions for selecting high-quality reference audio for voice cloning

Used by: VoiceCloningAgent
Purpose: Optimize reference audio selection to maximize voice quality while minimizing processing time
"""

import numpy as np
from pydub import AudioSegment
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_snr(audio_segment: AudioSegment) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) of audio segment

    Higher SNR = clearer audio with less background noise

    Args:
        audio_segment: AudioSegment to analyze

    Returns:
        SNR in dB (typically 10-50 dB, higher is better)
    """
    try:
        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())

        # Calculate signal power (RMS of entire signal)
        signal_power = np.mean(samples ** 2)

        # Estimate noise power from quieter sections (bottom 10% of energy)
        frame_size = len(samples) // 100
        frames = [samples[i:i+frame_size] for i in range(0, len(samples), frame_size)]
        frame_energies = [np.mean(frame ** 2) for frame in frames if len(frame) == frame_size]

        if not frame_energies:
            return 20.0  # Default moderate SNR

        noise_power = np.percentile(frame_energies, 10)  # Bottom 10% = noise estimate

        # Avoid division by zero
        if noise_power < 1e-10:
            return 50.0  # Very high SNR (nearly silent noise floor)

        # SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)

        return max(0, min(snr_db, 60))  # Clamp to reasonable range

    except Exception as e:
        logger.warning(f"SNR calculation failed: {e}")
        return 20.0  # Default moderate SNR


def calculate_rms_energy(audio_segment: AudioSegment) -> float:
    """
    Calculate Root Mean Square (RMS) energy of audio

    Higher RMS = louder, more energetic speech
    Helps avoid silent or very quiet segments

    Args:
        audio_segment: AudioSegment to analyze

    Returns:
        Normalized RMS energy (0-1, higher is better)
    """
    try:
        samples = np.array(audio_segment.get_array_of_samples()).astype(float)

        # Calculate RMS
        rms = np.sqrt(np.mean(samples ** 2))

        # Normalize to 0-1 range (assumes 16-bit audio: max value = 32768)
        max_possible = 32768.0
        normalized_rms = min(rms / (max_possible * 0.3), 1.0)  # 0.3 = reasonable speech level

        return normalized_rms

    except Exception as e:
        logger.warning(f"RMS calculation failed: {e}")
        return 0.5  # Default moderate energy


def detect_clipping(audio_segment: AudioSegment, threshold: float = 0.99) -> float:
    """
    Detect audio clipping (distortion from too-loud signal)

    Clipping degrades voice cloning quality significantly

    Args:
        audio_segment: AudioSegment to analyze
        threshold: Fraction of max value considered clipping (default: 0.99)

    Returns:
        Clipping ratio (0-1, lower is better, 0 = no clipping)
    """
    try:
        samples = np.array(audio_segment.get_array_of_samples()).astype(float)

        # For 16-bit audio
        max_value = 32768.0
        clip_threshold = max_value * threshold

        # Count samples near maximum
        clipped_samples = np.sum(np.abs(samples) > clip_threshold)
        clipping_ratio = clipped_samples / len(samples)

        return clipping_ratio

    except Exception as e:
        logger.warning(f"Clipping detection failed: {e}")
        return 0.0  # Assume no clipping on error


def calculate_spectral_flatness(audio_segment: AudioSegment) -> float:
    """
    Calculate spectral flatness (measure of tone richness)

    Higher flatness = more noise-like (bad)
    Lower flatness = more tonal (good for speech)

    Args:
        audio_segment: AudioSegment to analyze

    Returns:
        Spectral richness score (0-1, higher is better for speech)
    """
    try:
        samples = np.array(audio_segment.get_array_of_samples()).astype(float)

        # Compute FFT
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)

        # Avoid log(0)
        magnitude = np.maximum(magnitude, 1e-10)

        # Spectral flatness = geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(magnitude)))
        arithmetic_mean = np.mean(magnitude)

        if arithmetic_mean < 1e-10:
            return 0.5

        flatness = geometric_mean / arithmetic_mean

        # Invert for speech (lower flatness is better)
        # Return richness score (higher = better for speech)
        richness = 1.0 - min(flatness, 1.0)

        return richness

    except Exception as e:
        logger.warning(f"Spectral flatness calculation failed: {e}")
        return 0.5  # Default moderate richness


def score_audio_quality(audio_segment: AudioSegment, transcript: str = "") -> Dict[str, float]:
    """
    Comprehensive quality scoring for audio segment

    Combines multiple metrics to assess suitability for voice cloning reference

    Args:
        audio_segment: AudioSegment to score
        transcript: Optional transcript text for phoneme diversity analysis

    Returns:
        Dict with individual scores and total score
        {
            'snr': 0-1 (signal clarity),
            'energy': 0-1 (loudness/presence),
            'clipping': 0-1 (1 = no clipping),
            'spectral_richness': 0-1 (tonal quality),
            'duration': 0-1 (longer preferred, but diminishing returns),
            'phoneme_diversity': 0-1 (optional, requires transcript),
            'total': 0-1 (weighted average)
        }
    """
    # Calculate individual metrics
    snr = calculate_snr(audio_segment)
    energy = calculate_rms_energy(audio_segment)
    clipping_ratio = detect_clipping(audio_segment)
    spectral_richness = calculate_spectral_flatness(audio_segment)

    # Normalize SNR to 0-1 (assume good range is 15-45 dB)
    snr_normalized = max(0, min((snr - 15) / 30, 1.0))

    # Clipping score (invert ratio: high clipping = low score)
    clipping_score = 1.0 - clipping_ratio

    # Duration score (logarithmic: longer is better but diminishing returns)
    # Optimal range: 2-10 seconds
    duration_seconds = len(audio_segment) / 1000.0
    if duration_seconds < 1.0:
        duration_score = duration_seconds  # Penalty for very short
    elif duration_seconds < 10.0:
        duration_score = 0.5 + (duration_seconds / 20.0)  # 0.5-1.0
    else:
        duration_score = 1.0  # Max score for 10+ seconds

    # Phoneme diversity (optional, requires transcript)
    phoneme_diversity = 0.5  # Default if no transcript
    if transcript:
        # Simple diversity: unique characters / total characters
        unique_chars = len(set(transcript.lower().replace(' ', '')))
        total_chars = len(transcript.replace(' ', ''))
        phoneme_diversity = unique_chars / max(total_chars, 1) if total_chars > 0 else 0.5

    # Weighted total score
    # Weights tuned for voice cloning quality
    weights = {
        'snr': 0.30,              # Clarity is critical
        'energy': 0.20,            # Needs sufficient volume
        'clipping': 0.15,          # Avoid distortion
        'spectral_richness': 0.15, # Tonal quality matters
        'duration': 0.10,          # Longer helps but not critical
        'phoneme_diversity': 0.10  # Voice coverage
    }

    total_score = (
        snr_normalized * weights['snr'] +
        energy * weights['energy'] +
        clipping_score * weights['clipping'] +
        spectral_richness * weights['spectral_richness'] +
        duration_score * weights['duration'] +
        phoneme_diversity * weights['phoneme_diversity']
    )

    return {
        'snr': snr_normalized,
        'energy': energy,
        'clipping': clipping_score,
        'spectral_richness': spectral_richness,
        'duration': duration_score,
        'phoneme_diversity': phoneme_diversity,
        'total': total_score
    }


def select_best_reference_audio(
    audio_segments: List[Tuple[AudioSegment, str]],
    target_duration: float = 10.0,
    min_duration: float = 6.0
) -> List[AudioSegment]:
    """
    Select optimal reference audio for voice cloning

    Strategy:
    1. Score all segments by quality
    2. Select highest-quality segments up to target duration
    3. Ensure minimum duration is met
    4. Prioritize phoneme diversity

    Args:
        audio_segments: List of (AudioSegment, transcript) tuples
        target_duration: Target total duration in seconds (default: 10.0)
        min_duration: Minimum total duration in seconds (default: 6.0)

    Returns:
        List of selected AudioSegments (best quality, diverse, within duration)

    Example:
        >>> segments = [(audio1, "Hello world"), (audio2, "How are you")]
        >>> reference = select_best_reference_audio(segments, target_duration=10.0)
        >>> total_duration = sum(len(seg) for seg in reference) / 1000.0
        >>> print(f"Selected {len(reference)} segments, {total_duration:.1f}s total")
    """
    if not audio_segments:
        logger.warning("No audio segments provided for reference selection")
        return []

    # Score all segments
    scored_segments = []
    for audio_seg, transcript in audio_segments:
        score_dict = score_audio_quality(audio_seg, transcript)
        scored_segments.append({
            'audio': audio_seg,
            'transcript': transcript,
            'duration': len(audio_seg) / 1000.0,  # milliseconds to seconds
            'score': score_dict['total'],
            'quality_details': score_dict
        })

    # Sort by quality score (highest first)
    scored_segments.sort(key=lambda x: x['score'], reverse=True)

    # Greedy selection: pick best segments until target duration
    selected = []
    total_duration = 0.0
    phonemes_covered = set()

    for segment in scored_segments:
        seg_duration = segment['duration']
        seg_phonemes = set(segment['transcript'].lower().replace(' ', ''))

        # Check if adding this segment would exceed target
        if total_duration + seg_duration <= target_duration:
            selected.append(segment)
            total_duration += seg_duration
            phonemes_covered.update(seg_phonemes)

        # Stop if we've reached target duration
        elif total_duration >= min_duration:
            break

    # Ensure minimum duration
    if total_duration < min_duration and len(selected) < len(scored_segments):
        logger.info(f"Below minimum duration ({total_duration:.1f}s), adding more segments")
        for segment in scored_segments:
            if segment not in selected:
                selected.append(segment)
                total_duration += segment['duration']
                if total_duration >= min_duration:
                    break

    # Extract just the AudioSegment objects
    selected_audio = [seg['audio'] for seg in selected]

    # Log selection details
    logger.info(f"Selected {len(selected_audio)} segments for reference:")
    logger.info(f"  Total duration: {total_duration:.2f}s (target: {target_duration}s)")
    logger.info(f"  Average quality score: {np.mean([s['score'] for s in selected]):.3f}")
    logger.info(f"  Phoneme coverage: {len(phonemes_covered)} unique characters")

    return selected_audio
