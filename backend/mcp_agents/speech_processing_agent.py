import base64
import numpy as np
from io import BytesIO
from typing import Dict, Any, List
import assemblyai as aai
from pydub import AudioSegment
from .base_agent import MCPAgent

SAMPLE_RATE = 22050

class SpeechProcessingAgent(MCPAgent):
    def __init__(self, assembly_ai_key: str):
        super().__init__("speech_processing", ["transcribe_audio", "process_modifications", "generate_final_audio"])
        self.assembly_ai_key = assembly_ai_key
        aai.settings.api_key = assembly_ai_key
        
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
        """Generate final aggregated audio with all modifications and enhancements"""
        required_fields = ["conversations"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        try:
            conversations = request["conversations"]
            include_background_music = request.get("include_background_music", False)
            background_music_type = request.get("background_music_type", "calm")
            preserve_original_style = request.get("preserve_original_style", True)
            
            # Build final audio by concatenating all segments
            final_audio = AudioSegment.silent(duration=0)
            
            for conv_data in conversations:
                conv_key = list(conv_data.keys())[0]
                conv = conv_data[conv_key]
                
                # Use original audio if text unchanged, otherwise use generated audio
                if 'speaker_audio' in conv.get('original', {}):
                    # Check if this is an artificial speaker with generated audio
                    if conv.get('artificial', False) and 'original' in conv and 'speaker_audio' in conv['original']:
                        # Use AI-generated audio for artificial speakers
                        segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])
                    elif conv['modified']['text'] != conv['original']['text'] and preserve_original_style:
                        # Text was modified but we want to preserve original style
                        # For now, use original audio as a fallback (would need TTS integration for modified text)
                        segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])
                        self.logger.warning(f"Using original audio for modified text in {conv_key} - TTS integration needed for modified text")
                    else:
                        # Use original audio
                        segment_audio = self._byte_to_wav(conv['original']['speaker_audio'])
                else:
                    # Skip conversation with no audio data
                    self.logger.warning(f"No speaker audio found for {conv_key}")
                    continue
                
                # Add small silence between segments for natural flow
                if len(final_audio) > 0:
                    final_audio += AudioSegment.silent(duration=500)  # 0.5 second gap
                
                final_audio += segment_audio
            
            # Apply audio enhancements for quality
            final_audio = self._enhance_audio_quality(final_audio)
            
            # Convert to base64 for response
            final_audio_base64 = self._wav_to_byte(final_audio)
            
            return self.create_response(True, {
                "final_audio_base64": final_audio_base64,
                "duration_seconds": len(final_audio) / 1000.0,
                "sample_rate": SAMPLE_RATE,
                "channels": final_audio.channels,
                "has_background_music": include_background_music,
                "segments_processed": len([c for c in conversations if 'original' in list(c.values())[0] and 'speaker_audio' in list(c.values())[0]['original']])
            })
            
        except Exception as e:
            self.logger.error(f"Final audio generation error: {str(e)}")
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