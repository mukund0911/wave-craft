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
            # DEBUG: Log conversation order as received
            # ================================================================
            self.logger.info("="*70)
            self.logger.info("CONVERSATION ORDER (as received from frontend):")
            for idx, conv_data in enumerate(conversations):
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]
                speaker = conv.get('speaker', 'unknown')
                text_preview = conv.get('original', {}).get('text', '')[:50]
                self.logger.info(f"  [{idx}] {conv_key} - Speaker {speaker}: {text_preview}...")
            self.logger.info("="*70)

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
                    # CRITICAL FIX: For speech MODIFICATION (not TTS), we must use ONLY the current segment's
                    # own audio as reference. Using other segments causes VoiceCraft to include them in output!
                    # The current segment's audio is the "prompt" that gets modified based on text changes.
                    reference_audio_list = [conv['original']['speaker_audio']]

                    self.logger.info(f"[{conv_key}] 🔊 VOICE CLONING NEEDED - Text was modified")
                    self.logger.info(f"[{conv_key}]   Using current segment's own audio as reference for modification")

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

            self.logger.info("="*70)
            self.logger.info("ASSEMBLING FINAL AUDIO (in order):")
            self.logger.info("="*70)

            # Create temporary directory for intermediate audio files
            import tempfile
            import os
            from datetime import datetime
            from backend.utils.s3_storage import S3AudioStorage

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = os.path.join(tempfile.gettempdir(), "wavecraft_assembly", timestamp)
            os.makedirs(debug_dir, exist_ok=True)

            # Initialize S3 for debug file uploads
            s3_storage = S3AudioStorage()
            s3_debug_urls = []

            if s3_storage.is_enabled():
                self.logger.info(f"💾 Saving intermediate audio files to S3 (debug session: {timestamp})")
            else:
                self.logger.info(f"💾 Saving intermediate audio files locally to: {debug_dir}")

            final_audio = AudioSegment.silent(duration=0)
            segments_processed = 0
            segments_cloned = 0

            for idx, conv_data in enumerate(conversations):
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]

                speaker = conv.get('speaker', 'unknown')
                self.logger.info(f"\n[{idx}] Processing {conv_key} (Speaker {speaker})...")

                if 'speaker_audio' not in conv.get('original', {}):
                    self.logger.warning(f"  ⏭️  SKIPPED {conv_key} - No speaker audio found")
                    continue

                is_artificial = conv.get('artificial', False)
                original_text = conv['original'].get('text', '')
                modified_text = conv['modified'].get('text', '').strip()

                # Skip segments where all text was removed
                if not modified_text:
                    self.logger.info(f"  ⏭️  SKIPPED {conv_key} - All text removed by user")
                    continue

                # Check if this segment was cloned
                if idx in cloning_results_map:
                    cloning_result = cloning_results_map[idx]

                    if isinstance(cloning_result, Exception):
                        self.logger.error(f"  ❌ Voice cloning failed with exception: {cloning_result}")
                        self.logger.warning(f"  ⏭️  SKIPPED {conv_key} - Cloning failed")
                        continue
                    elif cloning_result.get('success'):
                        method = cloning_result['data'].get('method', 'unknown')

                        if method in ['fallback_no_modal', 'fallback_modal_error', 'original']:
                            self.logger.warning(f"  ⚠️  Fallback used - text modifications not reflected (method: {method})")
                        else:
                            self.logger.info(f"  ✅ Voice cloning SUCCESS (method: {method})")
                            segments_cloned += 1

                        segment_audio = self._byte_to_wav(cloning_result['data']['modified_audio_base64'])

                        # Compare original vs cloned
                        original_audio = self._byte_to_wav(conv['original']['speaker_audio'])
                        original_duration = len(original_audio) / 1000.0
                        cloned_duration = len(segment_audio) / 1000.0
                        original_text_len = len(original_text)
                        modified_text_len = len(modified_text)

                        self.logger.info(f"  📊 COMPARISON:")
                        self.logger.info(f"     Original: {original_duration:.2f}s audio, {original_text_len} chars text")
                        self.logger.info(f"     Cloned:   {cloned_duration:.2f}s audio, {modified_text_len} chars text")
                        self.logger.info(f"     Δ Audio:  {cloned_duration - original_duration:+.2f}s ({(cloned_duration/original_duration - 1)*100:+.1f}%)")
                        self.logger.info(f"     Δ Text:   {modified_text_len - original_text_len:+d} chars ({(modified_text_len/original_text_len - 1)*100:+.1f}%)")
                    else:
                        self.logger.error(f"  ❌ Cloning failed: {cloning_result.get('error')}")
                        self.logger.warning(f"  ⏭️  SKIPPED {conv_key} - Cloning failed")
                        continue

                elif is_artificial:
                    # Use AI-generated audio for artificial speakers
                    self.logger.info(f"  🤖 Using AI-generated audio")
                    segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])

                else:
                    # Use original audio (text unchanged)
                    self.logger.info(f"  📼 Using original audio (text unchanged)")
                    segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])

                # Save intermediate audio file for debugging
                try:
                    segment_duration = len(segment_audio) / 1000.0

                    # Determine source type for filename
                    if idx in cloning_results_map:
                        source_type = "cloned"
                    elif is_artificial:
                        source_type = "artificial"
                    else:
                        source_type = "original"

                    # Save segment audio locally
                    segment_filename = f"{idx:02d}_{conv_key}_{source_type}_{segment_duration:.2f}s.wav"
                    segment_path = os.path.join(debug_dir, segment_filename)
                    segment_audio.export(segment_path, format="wav")

                    # Upload to S3 if enabled
                    if s3_storage.is_enabled():
                        with open(segment_path, 'rb') as f:
                            audio_bytes = f.read()

                        debug_filename = f"debug/{timestamp}/{segment_filename}"
                        s3_key = s3_storage.upload_audio(audio_bytes, filename=debug_filename)

                        if s3_key:
                            s3_url = s3_storage.get_presigned_url(s3_key, expiration=86400)  # 24 hours
                            s3_debug_urls.append({
                                'segment': segment_filename,
                                'url': s3_url
                            })
                            self.logger.info(f"  💾 Uploaded to S3: {segment_filename}")
                    else:
                        self.logger.info(f"  💾 Saved locally: {segment_filename}")

                except Exception as e:
                    self.logger.warning(f"  Failed to save intermediate file: {e}")

                # Add silence between segments for natural flow
                if len(final_audio) > 0:
                    final_audio += AudioSegment.silent(duration=500)  # 0.5 second gap

                final_audio += segment_audio
                segments_processed += 1

                self.logger.info(f"  ✅ ADDED {conv_key} to final audio (duration: {segment_duration:.2f}s, total: {len(final_audio)/1000.0:.2f}s)")

            self.logger.info("="*70)
            self.logger.info(f"✓ Processed {segments_processed} segments, cloned {segments_cloned}")
            self.logger.info("="*70)

            # Save final assembled audio (before enhancement)
            try:
                final_duration = len(final_audio) / 1000.0
                final_filename = f"FINAL_assembled_{final_duration:.2f}s.wav"
                final_path = os.path.join(debug_dir, final_filename)
                final_audio.export(final_path, format="wav")

                # Upload final assembled audio to S3
                if s3_storage.is_enabled():
                    with open(final_path, 'rb') as f:
                        final_audio_bytes = f.read()

                    debug_final_filename = f"debug/{timestamp}/{final_filename}"
                    final_s3_key = s3_storage.upload_audio(final_audio_bytes, filename=debug_final_filename)

                    if final_s3_key:
                        final_s3_url = s3_storage.get_presigned_url(final_s3_key, expiration=86400)
                        s3_debug_urls.append({
                            'segment': final_filename,
                            'url': final_s3_url
                        })
                        self.logger.info(f"💾 Uploaded final assembled audio to S3: {final_filename}")

                # Save metadata file
                import json
                metadata = {
                    "timestamp": timestamp,
                    "total_segments": segments_processed,
                    "cloned_segments": segments_cloned,
                    "final_duration_seconds": final_duration,
                    "assembly_order": [],
                    "s3_debug_files": s3_debug_urls if s3_storage.is_enabled() else []
                }

                # Document each conversation that was processed
                for i, conv_data in enumerate(conversations):
                    conv_key = list(conv_data.keys())[0]
                    conv = conv_data[conv_key]
                    if i < segments_processed:  # Only include processed segments
                        metadata["assembly_order"].append({
                            "index": i,
                            "conv_key": conv_key,
                            "speaker": conv.get('speaker', 'unknown'),
                            "original_text": conv.get('original', {}).get('text', '')[:100],
                            "modified_text": conv.get('modified', {}).get('text', '')[:100]
                        })

                metadata_path = os.path.join(debug_dir, "ASSEMBLY_INFO.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # Upload metadata to S3
                if s3_storage.is_enabled():
                    with open(metadata_path, 'rb') as f:
                        metadata_bytes = f.read()

                    debug_metadata_filename = f"debug/{timestamp}/ASSEMBLY_INFO.json"
                    metadata_s3_key = s3_storage.upload_audio(metadata_bytes, filename=debug_metadata_filename, content_type='application/json')

                    if metadata_s3_key:
                        metadata_s3_url = s3_storage.get_presigned_url(metadata_s3_key, expiration=86400)
                        self.logger.info(f"💾 Uploaded assembly metadata to S3: ASSEMBLY_INFO.json")
                        self.logger.info(f"📋 Metadata URL: {metadata_s3_url}")

                # Log all S3 URLs
                if s3_storage.is_enabled() and s3_debug_urls:
                    self.logger.info("="*70)
                    self.logger.info("🔗 DEBUG FILES ON S3 (valid for 24 hours):")
                    for item in s3_debug_urls:
                        self.logger.info(f"  {item['segment']}: {item['url']}")
                    self.logger.info("="*70)

            except Exception as e:
                self.logger.warning(f"Failed to save/upload debug files: {e}")

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
            normalized_audio = audio.normalize()
            return normalized_audio
        except:
            return audio