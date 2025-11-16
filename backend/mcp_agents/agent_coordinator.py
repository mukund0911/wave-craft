import asyncio
from typing import Dict, Any, List, Optional
from .dialogue_generator_agent import DialogueGeneratorAgent
from .tts_agent import TextToSpeechAgent
from .speech_processing_agent import SpeechProcessingAgent
import logging

logger = logging.getLogger(__name__)

class AgentCoordinator:
    def __init__(self, openai_api_key: str, assembly_ai_key: str):
        self.dialogue_agent = DialogueGeneratorAgent(openai_api_key)
        self.tts_agent = TextToSpeechAgent(openai_api_key)
        self.speech_agent = SpeechProcessingAgent(assembly_ai_key)

        self.agents = {
            "dialogue_generator": self.dialogue_agent,
            "text_to_speech": self.tts_agent,
            "speech_processing": self.speech_agent
        }
    
    async def process_artificial_speaker_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main workflow for adding an artificial speaker:
        1. Generate dialogue based on context and speaker prompt
        2. Convert text to speech with specified characteristics
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

            # Create conversation entry
            conversation_entry = {
                "speaker": f"AI_{len(request['conversation_history']) + 1}",
                "original": {
                    "text": generated_text,
                    "speaker_audio": audio_base64,
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
                    "audio_base64": audio_base64,
                    "voice_used": tts_response["data"]["voice_used"]
                }
            }
            
        except Exception as e:
            logger.error(f"Artificial speaker workflow error: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to create artificial speaker: {str(e)}"
            }

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