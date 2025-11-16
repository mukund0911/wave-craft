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

                if text_modified and use_voice_cloning and not is_artificial:
                    # Create voice cloning task
                    reference_audio_list = [seg['audio_base64'] for seg in speaker_segments[speaker]]

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

                if 'speaker_audio' not in conv.get('original', {}):
                    self.logger.warning(f"No speaker audio found for {conv_key}")
                    continue

                is_artificial = conv.get('artificial', False)
                original_text = conv['original'].get('text', '')
                modified_text = conv['modified'].get('text', '')

                # Check if this segment was cloned
                if idx in cloning_results_map:
                    cloning_result = cloning_results_map[idx]

                    # Handle exceptions from parallel execution
                    if isinstance(cloning_result, Exception):
                        self.logger.error(f"[{conv_key}] Voice cloning failed with exception: {cloning_result}")
                        segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])
                    elif cloning_result.get('success'):
                        # Use cloned audio
                        segment_audio = self._byte_to_wav(cloning_result['data']['modified_audio_base64'])
                        segments_cloned += 1
                        method = cloning_result['data'].get('method', 'unknown')
                        self.logger.info(f"[{conv_key}] ✓ Voice cloned (method: {method})")
                    else:
                        # Fallback to original
                        self.logger.warning(f"[{conv_key}] ✗ Cloning failed: {cloning_result.get('error')}")
                        segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])

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

                final_audio += segment_audio
                segments_processed += 1

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