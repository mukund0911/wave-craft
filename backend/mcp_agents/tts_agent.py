import openai
import base64
from io import BytesIO
from typing import Dict, Any
from .base_agent import MCPAgent

class TextToSpeechAgent(MCPAgent):
    def __init__(self, api_key: str):
        super().__init__("text_to_speech", ["generate_speech", "list_voices"])
        if not api_key or api_key.strip() == "":
            raise ValueError("OpenAI API key is required")
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Voice mapping for different characteristics
        self.voice_mapping = {
            "male": {
                "calm": "onyx",
                "energetic": "echo", 
                "warm": "nova"
            },
            "female": {
                "calm": "shimmer",
                "energetic": "alloy",
                "warm": "nova"
            }
        }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        
        if action == "generate_speech":
            return await self._generate_speech(request)
        elif action == "list_voices":
            return await self._list_voices(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")
    
    async def _generate_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["text", "speaker_characteristics"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        text = request["text"]
        speaker_chars = request["speaker_characteristics"]
        
        # Parse speaker characteristics
        voice = self._select_voice(speaker_chars)
        speed = self._determine_speed(speaker_chars)
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed
            )
            
            # Convert audio to base64 for frontend compatibility
            audio_bytes = response.content
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            return self.create_response(True, {
                "audio_base64": audio_base64,
                "voice_used": voice,
                "speed_used": speed,
                "text_length": len(text)
            })
            
        except Exception as e:
            self.logger.error(f"OpenAI TTS error: {str(e)}")
            return self.create_response(False, error=f"Failed to generate speech: {str(e)}")
    
    async def _list_voices(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.create_response(True, {
            "available_voices": list(self.voice_mapping.keys()),
            "voice_mapping": self.voice_mapping
        })
    
    def _select_voice(self, speaker_characteristics: str) -> str:
        # Default voice
        default_voice = "alloy"
        
        # Parse characteristics (basic parsing - could be enhanced)
        lower_chars = speaker_characteristics.lower()
        
        if "male" in lower_chars or "man" in lower_chars:
            if "calm" in lower_chars or "relaxed" in lower_chars:
                return self.voice_mapping["male"]["calm"]
            elif "energetic" in lower_chars or "excited" in lower_chars:
                return self.voice_mapping["male"]["energetic"]
            else:
                return self.voice_mapping["male"]["warm"]
        elif "female" in lower_chars or "woman" in lower_chars:
            if "calm" in lower_chars or "relaxed" in lower_chars:
                return self.voice_mapping["female"]["calm"]
            elif "energetic" in lower_chars or "excited" in lower_chars:
                return self.voice_mapping["female"]["energetic"]
            else:
                return self.voice_mapping["female"]["warm"]
        
        return default_voice
    
    def _determine_speed(self, speaker_characteristics: str) -> float:
        lower_chars = speaker_characteristics.lower()
        
        if "fast" in lower_chars or "quick" in lower_chars or "energetic" in lower_chars:
            return 1.25
        elif "slow" in lower_chars or "calm" in lower_chars or "relaxed" in lower_chars:
            return 0.85
        else:
            return 1.0