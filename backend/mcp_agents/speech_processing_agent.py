import base64
import numpy as np
from io import BytesIO
from typing import Dict, Any, List
from collections import defaultdict
import assemblyai as aai
from pydub import AudioSegment
import asyncio
import time
from .base_agent import MCPAgent
from .voice_cloning_agent import VoiceCloningAgent

SAMPLE_RATE = 22050

class SpeechProcessingAgent(MCPAgent):
    def __init__(self, assembly_ai_key: str):
        super().__init__("speech_processing", ["transcribe_audio", "process_modifications", "generate_final_audio"])
        self.assembly_ai_key = assembly_ai_key
        aai.settings.api_key = assembly_ai_key
        self.voice_cloning_agent = VoiceCloningAgent()
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        
        if action == "transcribe_audio":
            return await self._transcribe_audio(request)
        elif action == "process_modifications":
            return await self._process_modifications(request)
        elif action == "generate_final_audio":
            return await self._generate_final_audio(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")
    
    async def _transcribe_audio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio file with speaker separation"""
        required_fields = ["audio_path"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        try:
            audio_path = request["audio_path"]
            
            # Load and process audio
            full_audio = AudioSegment.from_file(audio_path, format="wav")
            full_audio_base64 = self._wav_to_byte(full_audio)
            
            # Configure AssemblyAI
            config = aai.TranscriptionConfig(
                speaker_labels=True, 
                speech_model=aai.SpeechModel.best
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path, config=config)
            
            # Process transcription results
            conversations_sep = []
            for index, utterance in enumerate(transcript.utterances):
                words = {'original': {}, 'modified': {}}
                start, end = utterance.start, utterance.end
                
                words['speaker'] = utterance.speaker
                words['original']['text'] = utterance.text
                words['original']['start'] = start
                words['original']['end'] = end
                
                # Extract speaker audio segment
                speaker_audio = full_audio[start:end]
                audio_base64 = self._wav_to_byte(speaker_audio)
                words['original']['speaker_audio'] = audio_base64
                
                # Initialize modified section
                words['modified'] = {'text': utterance.text}
                
                conversations_sep.append({f"conv_{index}": words})
            
            return self.create_response(True, {
                "conversations": conversations_sep,
                "full_audio_base64": full_audio_base64,
                "total_duration": len(full_audio),
                "speaker_count": len(set([u.speaker for u in transcript.utterances]))
            })
            
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            return self.create_response(False, error=f"Failed to transcribe audio: {str(e)}")

    def submit_transcription_async(self, audio_path: str) -> str:
        """
        Submit async transcription job to AssemblyAI
        Returns: transcript_id for polling
        """
        try:
            self.logger.info(f"Submitting async transcription for: {audio_path}")

            # Optimize audio first
            optimized_path = self._optimize_audio(audio_path)
            self.logger.info(f"Audio optimized to: {optimized_path}")

            # Configure AssemblyAI
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speech_model=aai.SpeechModel.best
            )

            transcriber = aai.Transcriber()
            self.logger.info("Submitting to AssemblyAI...")
            transcript = transcriber.submit(optimized_path, config=config)

            self.logger.info(f"Submitted successfully, transcript_id: {transcript.id}")
            return transcript.id

        except Exception as e:
            self.logger.error(f"Failed to submit async transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def get_transcription_status(self, audio_path: str, transcript_id: str) -> Dict[str, Any]:
        """
        Check status of async transcription
        Returns: dict with status and result if complete
        """
        try:
            self.logger.info(f"Checking status for transcript_id: {transcript_id}")

            transcript = aai.Transcript.get_by_id(transcript_id)

            self.logger.info(f"Transcript status: {transcript.status}")

            # Get status string
            status = str(transcript.status).lower()
            if hasattr(transcript.status, 'value'):
                status = transcript.status.value

            self.logger.info(f"Parsed status: {status}")

            # Handle completed transcription
            if 'completed' in status or status == 'completed':
                self.logger.info("Status is completed, processing results...")

                full_audio = AudioSegment.from_file(audio_path, format="wav")
                full_audio_base64 = self._wav_to_byte(full_audio)

                conversations_sep = []
                for index, utterance in enumerate(transcript.utterances):
                    words = {'original': {}, 'modified': {}}
                    start, end = utterance.start, utterance.end

                    words['speaker'] = utterance.speaker
                    words['original']['text'] = utterance.text
                    words['original']['start'] = start
                    words['original']['end'] = end

                    speaker_audio = full_audio[start:end]
                    audio_base64 = self._wav_to_byte(speaker_audio)
                    words['original']['speaker_audio'] = audio_base64

                    words['modified'] = {'text': utterance.text}

                    conversations_sep.append({f"conv_{index}": words})

                return {
                    'status': 'completed',
                    'conversations': conversations_sep,
                    'full_audio': full_audio_base64
                }

            elif 'error' in status or status == 'error':
                return {
                    'status': 'error',
                    'error': str(getattr(transcript, 'error', 'Unknown error'))
                }

            else:
                self.logger.info(f"Status is: {status}")
                return {
                    'status': status  # 'queued' or 'processing'
                }

        except Exception as e:
            self.logger.error(f"Exception in get_transcription_status: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'error': f'Failed to get transcript status: {str(e)}'
            }

    def _optimize_audio(self, audio_path: str) -> str:
        """Optimize audio for faster transcription - mono, 16kHz"""
        try:
            self.logger.info(f"Loading audio from: {audio_path}")
            audio = AudioSegment.from_file(audio_path)

            self.logger.info(f"Original: {audio.channels} channels, {audio.frame_rate}Hz")

            # Convert to mono and downsample to 16kHz (optimal for AssemblyAI)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            self.logger.info(f"Optimized: {audio.channels} channels, {audio.frame_rate}Hz")

            # Export optimized version
            optimized_path = audio_path.replace('.wav', '_optimized.wav')
            self.logger.info(f"Exporting to: {optimized_path}")
            audio.export(optimized_path, format="wav")

            self.logger.info("Audio optimization complete")
            return optimized_path

        except Exception as e:
            self.logger.error(f"Failed to optimize audio: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    async def _process_modifications(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process text modifications and prepare for audio generation"""
        required_fields = ["conversations", "modifications"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        try:
            conversations = request["conversations"]
            modifications = request["modifications"]
            
            # Apply modifications to conversations
            processed_conversations = []
            for conv_data in conversations:
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]
                
                # Apply text modifications if any
                if conv_key in modifications:
                    conv['modified']['text'] = modifications[conv_key]
                
                processed_conversations.append({conv_key: conv})
            
            return self.create_response(True, {
                "processed_conversations": processed_conversations,
                "total_segments": len(processed_conversations)
            })
            
        except Exception as e:
            self.logger.error(f"Modification processing error: {str(e)}")
            return self.create_response(False, error=f"Failed to process modifications: {str(e)}")
    
    async def _generate_final_audio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final aggregated audio with all modifications and enhancements

        Optimizations:
        - Parallel voice cloning (2-3x faster for multiple segments)
        - Smart reference audio selection
        - Graceful fallback handling

        Performance:
        - Sequential: 10s × 3 segments = 30s
        - Parallel: max(10s, 10s, 10s) = 10s ✅
        """
        required_fields = ["conversations"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")

        start_time = time.time()

        try:
            conversations = request["conversations"]
            use_voice_cloning = request.get("use_voice_cloning", True)  # Enable voice cloning by default

            self.logger.info(f"Generating final audio for {len(conversations)} conversations")
            self.logger.info(f"Voice cloning enabled: {use_voice_cloning}")

            # ================================================================
            # STEP 1: Group conversations by speaker to collect reference audio
            # ================================================================

            speaker_segments = defaultdict(list)
            for conv_data in conversations:
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]

                speaker = conv.get('speaker', 'unknown')
                if 'speaker_audio' in conv.get('original', {}):
                    speaker_segments[speaker].append({
                        'conv_key': conv_key,
                        'audio_base64': conv['original']['speaker_audio'],
                        'original_text': conv['original'].get('text', ''),
                        'modified_text': conv['modified'].get('text', ''),
                        'artificial': conv.get('artificial', False)
                    })

            self.logger.info(f"Found {len(speaker_segments)} unique speakers")

            # ================================================================
            # STEP 2: PARALLEL VOICE CLONING
            # ================================================================

            # Identify segments that need voice cloning
            cloning_tasks = []
            cloning_indices = []  # Track which conversations need cloning

            for idx, conv_data in enumerate(conversations):
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]

                if 'speaker_audio' not in conv.get('original', {}):
                    continue

                speaker = conv.get('speaker', 'unknown')
                original_text = conv['original'].get('text', '')
                modified_text = conv['modified'].get('text', '')
                is_artificial = conv.get('artificial', False)

                # Determine if we need voice cloning
                text_modified = original_text != modified_text

                self.logger.info(f"[{conv_key}] Text modified: {text_modified}, use_voice_cloning: {use_voice_cloning}, artificial: {is_artificial}")
                self.logger.info(f"[{conv_key}] Original text: '{original_text}'")
                self.logger.info(f"[{conv_key}] Modified text: '{modified_text}'")

                if text_modified and use_voice_cloning and not is_artificial:
                    # Create voice cloning task
                    reference_audio_list = [seg['audio_base64'] for seg in speaker_segments[speaker]]

                    self.logger.info(f"[{conv_key}] 🔊 VOICE CLONING NEEDED - Text was modified")
                    self.logger.info(f"[{conv_key}]   Speaker: {speaker}, Reference segments: {len(reference_audio_list)}")

                    cloning_request = {
                        "action": "modify_speech",
                        "reference_audio_list": reference_audio_list,
                        "original_text": original_text,
                        "modified_text": modified_text,
                        "speaker_id": speaker
                    }

                    # Add task to parallel batch
                    task = self.voice_cloning_agent.process_request(cloning_request)
                    cloning_tasks.append(task)
                    cloning_indices.append(idx)

                    self.logger.info(f"[{conv_key}] Queued for parallel voice cloning (speaker {speaker})")
                else:
                    if not text_modified:
                        self.logger.info(f"[{conv_key}] ⏩ SKIPPING CLONING - Text unchanged")
                    elif not use_voice_cloning:
                        self.logger.warning(f"[{conv_key}] ⚠️  CLONING DISABLED - Will use original audio despite text change")
                    elif is_artificial:
                        self.logger.info(f"[{conv_key}] ⏩ SKIPPING - Artificial speaker")

            # Execute all voice cloning tasks in parallel
            self.logger.info(f"Starting parallel voice cloning for {len(cloning_tasks)} segments...")
            parallel_start = time.time()

            if cloning_tasks:
                cloning_results = await asyncio.gather(*cloning_tasks, return_exceptions=True)
                parallel_time = time.time() - parallel_start
                self.logger.info(f"✓ Parallel voice cloning completed in {parallel_time:.2f}s")
            else:
                cloning_results = []
                self.logger.info("No segments require voice cloning")

            # Map results back to conversations
            cloning_results_map = {}
            for idx, result_idx in enumerate(cloning_indices):
                cloning_results_map[result_idx] = cloning_results[idx]

            # ================================================================
            # STEP 3: Assemble final audio (sequential, order matters)
            # ================================================================

            final_audio = AudioSegment.silent(duration=0)
            segments_processed = 0
            segments_cloned = 0

            for idx, conv_data in enumerate(conversations):
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]

                # DEBUG: Track assembly order
                self.logger.info(f"🔢 ASSEMBLING idx={idx}, conv_key={conv_key}, position in final audio: {len(final_audio)/1000:.2f}s")

                if 'speaker_audio' not in conv.get('original', {}):
                    self.logger.warning(f"No speaker audio found for {conv_key}")
                    continue

                is_artificial = conv.get('artificial', False)
                original_text = conv['original'].get('text', '')
                modified_text = conv['modified'].get('text', '').strip()

                # Skip segments where all text was removed
                if not modified_text:
                    self.logger.info(f"[{conv_key}] Skipping - all text removed by user")
                    continue

                # Check if this segment was cloned
                if idx in cloning_results_map:
                    cloning_result = cloning_results_map[idx]

                    self.logger.info(f"[{conv_key}] 📊 Processing cloning result...")

                    # Handle exceptions from parallel execution
                    if isinstance(cloning_result, Exception):
                        self.logger.error(f"[{conv_key}] ❌ Voice cloning failed with exception: {cloning_result}")
                        # Skip segment instead of using original audio with wrong text
                        self.logger.warning(f"[{conv_key}] ⏭️  Skipping segment - cloning failed and text was modified")
                        continue
                    elif cloning_result.get('success'):
                        # Use cloned audio
                        method = cloning_result['data'].get('method', 'unknown')
                        self.logger.info(f"[{conv_key}] ✅ Voice cloning SUCCESS (method: {method})")

                        # Check if it's actually cloned or fallback
                        if method in ['fallback_no_modal', 'fallback_modal_error', 'original']:
                            self.logger.warning(f"[{conv_key}] ⚠️  FALLBACK USED - Modal not available or failed")
                            self.logger.warning(f"[{conv_key}]   Fallback reason: {cloning_result['data'].get('metadata', {}).get('reason', 'unknown')}")
                            self.logger.warning(f"[{conv_key}] ⚠️  Using original audio (text modification won't be reflected)")
                        else:
                            segments_cloned += 1

                        segment_audio = self._byte_to_wav(cloning_result['data']['modified_audio_base64'])

                        # DEBUG: Compare original vs cloned audio duration
                        original_audio_seg = self._byte_to_wav(conv['original']['speaker_audio'])
                        original_dur = len(original_audio_seg) / 1000.0
                        cloned_dur = len(segment_audio) / 1000.0
                        dur_diff = original_dur - cloned_dur

                        # Check if audio is actually silent (amplitude check)
                        import numpy as np
                        audio_array = np.array(segment_audio.get_array_of_samples())
                        max_amplitude = np.max(np.abs(audio_array))
                        rms = np.sqrt(np.mean(audio_array**2))

                        self.logger.info(f"[{conv_key}] 🎵 Audio Duration Comparison:")
                        self.logger.info(f"[{conv_key}]   Original: {original_dur:.2f}s")
                        self.logger.info(f"[{conv_key}]   Cloned: {cloned_dur:.2f}s")
                        self.logger.info(f"[{conv_key}]   Difference: {dur_diff:.2f}s ({'shorter' if dur_diff > 0 else 'longer'} after cloning)")
                        self.logger.info(f"[{conv_key}]   Max amplitude: {max_amplitude}, RMS: {rms:.2f}")

                        if max_amplitude < 100:
                            self.logger.warning(f"[{conv_key}] ⚠️  CLONED AUDIO IS NEARLY SILENT! (max_amp={max_amplitude})")
                            self.logger.warning(f"[{conv_key}]   This segment will be inaudible in final audio!")

                        # Check sample rate compatibility
                        self.logger.info(f"[{conv_key}]   Sample rate: {segment_audio.frame_rate}Hz, Channels: {segment_audio.channels}")
                        self.logger.info(f"[{conv_key}]   Original sample rate: {original_audio_seg.frame_rate}Hz")

                        # CRITICAL: Ensure sample rate matches for concatenation
                        if segment_audio.frame_rate != original_audio_seg.frame_rate:
                            self.logger.warning(f"[{conv_key}] ⚠️  Sample rate mismatch! Converting {segment_audio.frame_rate}Hz → {original_audio_seg.frame_rate}Hz")
                            segment_audio = segment_audio.set_frame_rate(original_audio_seg.frame_rate)
                    else:
                        # Skip segment instead of using original audio with wrong text
                        self.logger.error(f"[{conv_key}] ❌ Cloning failed: {cloning_result.get('error')}")
                        self.logger.warning(f"[{conv_key}] ⏭️  Skipping segment")
                        continue

                elif is_artificial:
                    # Use AI-generated audio for artificial speakers
                    self.logger.info(f"[{conv_key}] Using AI-generated audio")
                    segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])

                else:
                    # Use original audio (text unchanged)
                    self.logger.info(f"[{conv_key}] Using original audio")
                    segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])

                # Add silence between segments for natural flow
                if len(final_audio) > 0:
                    final_audio += AudioSegment.silent(duration=500)  # 0.5 second gap

                segment_duration = len(segment_audio) / 1000.0
                final_audio += segment_audio
                segments_processed += 1

                # DEBUG: Confirm what was added
                self.logger.info(f"✅ ADDED {conv_key} to final audio: duration={segment_duration:.2f}s, new total={len(final_audio)/1000:.2f}s")

            self.logger.info(f"Processed {segments_processed} segments, cloned {segments_cloned}")

            # ================================================================
            # STEP 4: Apply audio enhancements for quality
            # ================================================================

            final_audio = self._enhance_audio_quality(final_audio)

            # Convert to base64 for response
            final_audio_base64 = self._wav_to_byte(final_audio)

            total_time = time.time() - start_time
            self.logger.info(f"✓ Final audio generation completed in {total_time:.2f}s")

            return self.create_response(True, {
                "final_audio_base64": final_audio_base64,
                "duration_seconds": len(final_audio) / 1000.0,
                "sample_rate": SAMPLE_RATE,
                "channels": final_audio.channels,
                "segments_processed": segments_processed,
                "segments_cloned": segments_cloned,
                "voice_cloning_enabled": use_voice_cloning,
                "processing_time_seconds": total_time
            })

        except Exception as e:
            self.logger.error(f"Final audio generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return self.create_response(False, error=f"Failed to generate final audio: {str(e)}")
    
    def _wav_to_byte(self, audio_segment: AudioSegment) -> str:
        """Convert AudioSegment to base64 encoded bytes"""
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def _byte_to_wav(self, byte_encode: str) -> AudioSegment:
        """Convert base64 encoded bytes to AudioSegment"""
        audio_decode = base64.b64decode(byte_encode)
        buffer = BytesIO(audio_decode)
        return AudioSegment.from_file(buffer, format="wav")
    
    def _enhance_audio_quality(self, audio: AudioSegment) -> AudioSegment:
        """Apply audio enhancements for better quality"""
        try:
            # Normalize audio levels
            normalized_audio = audio.normalize()
            
            # Apply gentle compression to even out levels
            # Simple dynamic range compression
            compressed_audio = normalized_audio.compress_dynamic_range(threshold=-20.0, ratio=4.0)
            
            # High-pass filter to remove low-frequency noise (if available)
            # Note: pydub has limited built-in effects, for production consider using librosa
            
            return compressed_audio
        except:
            # If enhancement fails, return original
            return audio