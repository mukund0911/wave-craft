import asyncio
from typing import Dict, Any, List, Optional
from .dialogue_generator_agent import DialogueGeneratorAgent
from .tts_agent import TextToSpeechAgent
from .music_agent import BackgroundMusicAgent
from .speech_processing_agent import SpeechProcessingAgent
import logging

logger = logging.getLogger(__name__)

class AgentCoordinator:
    def __init__(self, openai_api_key: str, assembly_ai_key: str):
        self.dialogue_agent = DialogueGeneratorAgent(openai_api_key)
        self.tts_agent = TextToSpeechAgent(openai_api_key)
        self.speech_agent = SpeechProcessingAgent(assembly_ai_key)
        self.music_agent = BackgroundMusicAgent()
        
        self.agents = {
            "dialogue_generator": self.dialogue_agent,
            "text_to_speech": self.tts_agent,
            "speech_processing": self.speech_agent,
            "background_music": self.music_agent
        }
    
    async def process_artificial_speaker_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main workflow for adding an artificial speaker:
        1. Generate dialogue based on context and speaker prompt
        2. Convert text to speech with specified characteristics
        3. Optionally add background music
        """
        try:
            # Step 1: Generate dialogue
            dialogue_request = {
                "action": "generate_dialogue",
                "conversation_history": request["conversation_history"],
                "speaker_prompt": request["speaker_prompt"]
            }
            
            dialogue_response = await self.dialogue_agent.process_request(dialogue_request)
            if not dialogue_response["success"]:
                return dialogue_response
            
            generated_text = dialogue_response["data"]["generated_dialogue"]
            
            # Step 2: Convert to speech
            tts_request = {
                "action": "generate_speech",
                "text": generated_text,
                "speaker_characteristics": request["speaker_prompt"]
            }
            
            tts_response = await self.tts_agent.process_request(tts_request)
            if not tts_response["success"]:
                return tts_response
            
            audio_base64 = tts_response["data"]["audio_base64"]
            
            # Step 3: Add background music if requested
            final_audio = audio_base64
            if request.get("add_background_music", False):
                music_request = {
                    "action": "add_background_music",
                    "audio_base64": audio_base64,
                    "music_type": request.get("music_type", "calm"),
                    "volume_level": request.get("music_volume", 0.3)
                }
                
                music_response = await self.music_agent.process_request(music_request)
                if music_response["success"]:
                    final_audio = music_response["data"]["mixed_audio_base64"]
            
            # Create conversation entry
            conversation_entry = {
                "speaker": f"AI_{len(request['conversation_history']) + 1}",
                "original": {
                    "text": generated_text,
                    "speaker_audio": final_audio,
                    "start": 0,  # Will be calculated based on position
                    "end": 0     # Will be calculated based on audio length
                },
                "modified": {
                    "text": generated_text
                },
                "artificial": True,
                "speaker_characteristics": request["speaker_prompt"]
            }
            
            return {
                "success": True,
                "data": {
                    "conversation_entry": conversation_entry,
                    "generated_text": generated_text,
                    "audio_base64": final_audio,
                    "voice_used": tts_response["data"]["voice_used"],
                    "has_background_music": request.get("add_background_music", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Artificial speaker workflow error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to create artificial speaker: {str(e)}"
            }
    
    async def add_background_music_to_conversation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Add background music to existing conversation or full audio"""
        return await self.music_agent.process_request(request)
    
    async def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Return capabilities of all agents"""
        return {
            agent_id: agent.capabilities 
            for agent_id, agent in self.agents.items()
        }
    
    async def route_request(self, agent_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to specific agent"""
        if agent_id not in self.agents:
            return {
                "success": False,
                "error": f"Unknown agent: {agent_id}"
            }
        
        return await self.agents[agent_id].process_request(request)