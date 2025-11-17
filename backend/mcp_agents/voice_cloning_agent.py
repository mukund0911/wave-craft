"""
Voice Cloning Agent - Production Implementation
Uses Modal.com for GPU-accelerated VoiceCraft inference

Architecture:
- Smart reference audio selection (optimal quality, 6-10s)
- Modal API integration for serverless GPU inference
- Graceful fallback to original audio if cloning fails
- Comprehensive error handling and logging

Performance:
- Reference selection: <1s (local processing)
- Modal inference: 6-8s (A10G GPU, warm) / 15-18s (cold start)
- Total latency: 7-9s per segment (typical)

Cost:
- Modal GPU: ~$0.0036 per segment
- No cost when using original audio (unchanged text)
"""

import base64
import os
import requests
from io import BytesIO
from typing import Dict, Any, List, Tuple
from pydub import AudioSegment
from .base_agent import MCPAgent
import logging
import asyncio
import time

# Import audio quality utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.audio_quality import select_best_reference_audio, score_audio_quality

logger = logging.getLogger(__name__)


class VoiceCloningAgent(MCPAgent):
    """
    Production-ready voice cloning agent

    Features:
    - Modal.com integration for GPU inference
    - Smart reference audio selection
    - Parallel processing support
    - Automatic fallback mechanisms
    - Detailed performance monitoring
    """

    def __init__(self):
        super().__init__("voice_cloning", ["clone_voice", "modify_speech"])

        # Modal endpoint URL (set in environment)
        self.modal_url = os.environ.get(
            'MODAL_VOICECRAFT_URL',
            None  # Will be set after Modal deployment
        )

        # Configuration
        self.config = {
            'reference_duration_target': 10.0,  # Target duration in seconds
            'reference_duration_min': 6.0,      # Minimum duration in seconds
            'sample_rate': 24000,                # VoiceCraft native sample rate
            'timeout': 120,                      # API timeout in seconds
            'enable_quality_selection': True,    # Use smart reference selection
        }

        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_clones': 0,
            'fallbacks': 0,
            'total_inference_time': 0.0
        }

        logger.info("VoiceCloningAgent initialized")
        logger.info(f"Modal URL: {self.modal_url or 'NOT SET (will use fallback)'}")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route requests to appropriate handler

        Supported actions:
        - clone_voice: Full voice cloning
        - modify_speech: Speech modification with voice preservation
        """
        action = request.get("action")

        if action in ["clone_voice", "modify_speech"]:
            return await self._modify_speech(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")

    async def _modify_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify speech using VoiceCraft via Modal

        Process:
        1. Validate input
        2. Check if text actually changed
        3. Select best reference audio (smart selection)
        4. Call Modal API for inference
        5. Return result with metadata

        Args:
            reference_audio_list: List of base64-encoded audio segments (all from same speaker)
            original_text: Original transcript of the segment to modify
            modified_text: New transcript to generate
            speaker_id: Speaker identifier for logging/debugging

        Returns:
            {
                'success': bool,
                'data': {
                    'modified_audio_base64': str,
                    'method': 'voicecraft' | 'original' | 'fallback',
                    'original_text': str,
                    'modified_text': str,
                    'metadata': {
                        'inference_time': float,
                        'reference_duration': float,
                        'reference_segments': int,
                        'quality_score': float
                    }
                }
            }
        """
        start_time = time.time()
        self.stats['total_requests'] += 1

        try:
            # ================================================================
            # 1. EXTRACT AND VALIDATE INPUT
            # ================================================================

            reference_audio_list = request.get("reference_audio_list", [])
            original_text = request.get("original_text", "").strip()
            modified_text = request.get("modified_text", "").strip()
            speaker_id = request.get("speaker_id", "unknown")

            logger.info(f"[Speaker {speaker_id}] Voice cloning request:")
            logger.info(f"  Original: '{original_text}'")
            logger.info(f"  Modified: '{modified_text}'")
            logger.info(f"  Reference segments: {len(reference_audio_list)}")

            # Validation
            if not reference_audio_list:
                logger.error("No reference audio provided")
                return self.create_response(False, error="No reference audio provided")

            if not modified_text:
                logger.error("No modified text provided")
                return self.create_response(False, error="No modified text provided")

            # ================================================================
            # 2. CHECK IF TEXT CHANGED
            # ================================================================

            if original_text == modified_text:
                logger.info("Text unchanged, returning original audio")
                return self.create_response(True, {
                    "modified_audio_base64": reference_audio_list[0],
                    "method": "original",
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "metadata": {
                        "inference_time": time.time() - start_time,
                        "reason": "text_unchanged"
                    }
                })

            # ================================================================
            # 3. SELECT BEST REFERENCE AUDIO
            # ================================================================

            if self.config['enable_quality_selection']:
                # Convert base64 audio to AudioSegment objects with transcripts
                audio_segments_with_text: List[Tuple[AudioSegment, str]] = []

                for idx, audio_b64 in enumerate(reference_audio_list):
                    try:
                        audio_seg = self._base64_to_audio(audio_b64)
                        # Use original_text as transcript for first segment,
                        # empty string for others (we don't have individual transcripts)
                        text = original_text if idx == 0 else ""
                        audio_segments_with_text.append((audio_seg, text))
                    except Exception as e:
                        logger.warning(f"Failed to decode reference segment {idx}: {e}")
                        continue

                if not audio_segments_with_text:
                    logger.error("Failed to decode any reference audio")
                    return self.create_response(False, error="Failed to decode reference audio")

                # Select best segments
                logger.info(f"Selecting best reference from {len(audio_segments_with_text)} segments")
                selected_segments = select_best_reference_audio(
                    audio_segments_with_text,
                    target_duration=self.config['reference_duration_target'],
                    min_duration=self.config['reference_duration_min']
                )

                if not selected_segments:
                    logger.error("Reference selection returned no segments")
                    return self.create_response(False, error="Reference selection failed")

                # Combine selected segments
                combined_reference = AudioSegment.silent(duration=0)
                for seg in selected_segments:
                    combined_reference += seg

                reference_duration = len(combined_reference) / 1000.0  # ms to seconds
                logger.info(f"Selected {len(selected_segments)} segments, {reference_duration:.2f}s total")

            else:
                # Use all reference audio (legacy mode)
                logger.info("Using all reference audio (quality selection disabled)")
                combined_reference = AudioSegment.silent(duration=0)
                for audio_b64 in reference_audio_list:
                    combined_reference += self._base64_to_audio(audio_b64)

                reference_duration = len(combined_reference) / 1000.0

            # ================================================================
            # 4. PREPARE FOR MODAL API CALL
            # ================================================================

            # Standardize audio format for VoiceCraft
            combined_reference = combined_reference.set_channels(1)  # Mono
            combined_reference = combined_reference.set_frame_rate(
                self.config['sample_rate']
            )

            # Convert to base64
            reference_b64 = self._audio_to_base64(combined_reference)

            # ================================================================
            # 5. CALL MODAL API
            # ================================================================

            if not self.modal_url:
                logger.warning("❌ Modal URL not configured, using fallback (original audio)")
                logger.warning(f"   Set MODAL_VOICECRAFT_URL environment variable to enable voice cloning")
                logger.warning(f"   Original text: '{original_text}'")
                logger.warning(f"   Modified text: '{modified_text}'")
                self.stats['fallbacks'] += 1
                return self.create_response(True, {
                    "modified_audio_base64": reference_audio_list[0],
                    "method": "fallback_no_modal",
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "metadata": {
                        "inference_time": time.time() - start_time,
                        "reason": "modal_not_configured"
                    }
                })

            try:
                logger.info(f"Calling Modal API: {self.modal_url}")

                inference_start = time.time()

                # Make async request to Modal
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.post(
                        f"{self.modal_url}/clone",
                        json={
                            "reference_audio_b64": reference_b64,
                            "original_text": original_text,
                            "modified_text": modified_text,
                            "sample_rate": self.config['sample_rate']
                        },
                        timeout=self.config['timeout']
                    )
                )

                inference_time = time.time() - inference_start

                # Check response
                if response.status_code != 200:
                    logger.error(f"Modal API error: {response.status_code} - {response.text}")
                    raise Exception(f"Modal API returned {response.status_code}")

                result = response.json()

                if not result.get('success'):
                    error_msg = result.get('error', 'Unknown error')
                    logger.error(f"Modal inference failed: {error_msg}")
                    raise Exception(f"Inference failed: {error_msg}")

                # Success!
                modified_audio_b64 = result['audio_b64']
                modal_metadata = result.get('metadata', {})

                self.stats['successful_clones'] += 1
                self.stats['total_inference_time'] += inference_time

                logger.info(f"✓ Voice cloning successful ({inference_time:.2f}s)")

                return self.create_response(True, {
                    "modified_audio_base64": modified_audio_b64,
                    "method": "voicecraft",
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "metadata": {
                        "inference_time": time.time() - start_time,
                        "modal_inference_time": inference_time,
                        "reference_duration": reference_duration,
                        "reference_segments": len(selected_segments) if self.config['enable_quality_selection'] else len(reference_audio_list),
                        "modal_metadata": modal_metadata
                    }
                })

            except Exception as e:
                logger.error(f"Modal API call failed: {e}")
                logger.info("Falling back to original audio")
                self.stats['fallbacks'] += 1

                # Fallback to original audio
                return self.create_response(True, {
                    "modified_audio_base64": reference_audio_list[0],
                    "method": "fallback_modal_error",
                    "original_text": original_text,
                    "modified_text": modified_text,
                    "metadata": {
                        "inference_time": time.time() - start_time,
                        "error": str(e),
                        "reason": "modal_api_failed"
                    }
                })

        except Exception as e:
            logger.error(f"Voice cloning error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.create_response(False, error=f"Failed to modify speech: {str(e)}")

    async def _clone_voice(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for _modify_speech"""
        return await self._modify_speech(request)

    def _base64_to_audio(self, audio_base64: str) -> AudioSegment:
        """
        Convert base64 string to AudioSegment

        Args:
            audio_base64: Base64-encoded audio data

        Returns:
            AudioSegment object
        """
        audio_bytes = base64.b64decode(audio_base64)
        buffer = BytesIO(audio_bytes)
        return AudioSegment.from_file(buffer)

    def _audio_to_base64(self, audio_segment: AudioSegment) -> str:
        """
        Convert AudioSegment to base64 string

        Args:
            audio_segment: AudioSegment to encode

        Returns:
            Base64-encoded WAV audio
        """
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode('utf-8')

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics

        Returns:
            Dictionary with performance metrics
        """
        avg_inference_time = (
            self.stats['total_inference_time'] / self.stats['successful_clones']
            if self.stats['successful_clones'] > 0
            else 0
        )

        success_rate = (
            self.stats['successful_clones'] / self.stats['total_requests'] * 100
            if self.stats['total_requests'] > 0
            else 0
        )

        return {
            'total_requests': self.stats['total_requests'],
            'successful_clones': self.stats['successful_clones'],
            'fallbacks': self.stats['fallbacks'],
            'success_rate': f"{success_rate:.1f}%",
            'average_inference_time': f"{avg_inference_time:.2f}s",
            'modal_url_configured': self.modal_url is not None
        }
