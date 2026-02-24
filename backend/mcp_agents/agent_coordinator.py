"""
Agent Coordinator (Revamped)

Orchestrates the MCP agents:
- SpeechProcessingAgent (WhisperX transcription + Chatterbox modification)
- ChatterboxAgent (voice cloning + TTS)
- DialogueGeneratorAgent (GPT dialogue for artificial speakers)

Removed:
- VoiceCloningAgent (replaced by ChatterboxAgent)
- TextToSpeechAgent (replaced by ChatterboxAgent)
"""

import os
import logging
from typing import Dict, Any
from .speech_processing_agent import SpeechProcessingAgent
from .chatterbox_agent import ChatterboxAgent
from .dialogue_generator_agent import DialogueGeneratorAgent

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates the MCP agent system.

    Primary workflows:
    1. Transcribe: Upload → WhisperX → speaker-diarized transcript
    2. Modify: Edited transcript → Chatterbox voice cloning → final audio
    3. Artificial Speaker: Prompt → GPT dialogue → Chatterbox TTS
    """

    def __init__(self):
        self.dialogue_agent = DialogueGeneratorAgent()
        self.chatterbox_agent = ChatterboxAgent()
        self.speech_agent = SpeechProcessingAgent()

        logger.info("AgentCoordinator initialized with Chatterbox + WhisperX")

    async def process_artificial_speaker_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an artificial speaker's dialogue and voice.

        Flow: User prompt → GPT generates dialogue → Chatterbox generates speech

        Args:
            request: {
                "prompt": str,
                "conversation_history": list,
                "speaker_characteristics": str,
                "exaggeration": float (0.0-1.0)
            }

        Returns:
            response with conversation_entry + audio
        """
        try:
            # Step 1: Generate dialogue
            dialogue_request = {
                "action": "generate_dialogue",
                "prompt": request.get("prompt", ""),
                "conversation_history": request.get("conversation_history", []),
                "speaker_characteristics": request.get("speaker_characteristics", "")
            }

            dialogue_response = await self.dialogue_agent.process_request(dialogue_request)

            if not dialogue_response.get("success"):
                return {
                    "success": False,
                    "error": f"Dialogue generation failed: {dialogue_response.get('error')}"
                }

            generated_text = dialogue_response["data"]["dialogue"]

            # Step 2: Generate speech with Chatterbox
            tts_request = {
                "action": "generate_speech",
                "text": generated_text,
                "speaker_characteristics": request.get("speaker_characteristics", ""),
                "exaggeration": request.get("exaggeration", 0.5)
            }

            tts_response = await self.chatterbox_agent.process_request(tts_request)

            if not tts_response.get("success"):
                return {
                    "success": False,
                    "error": f"TTS failed: {tts_response.get('error')}"
                }

            # Build conversation entry
            conversation_entry = {
                "speaker": request.get("speaker_name", "AI"),
                "original": {
                    "text": generated_text,
                    "speaker_audio": tts_response["data"]["audio_base64"]
                },
                "modified": {
                    "text": generated_text,
                    "emotions": []
                }
            }

            return {
                "success": True,
                "data": {
                    "conversation_entry": conversation_entry,
                    "dialogue": generated_text,
                    "voice_used": tts_response["data"].get("voice_used", "chatterbox")
                }
            }

        except Exception as e:
            logger.error(f"Artificial speaker request failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_agent_capabilities(self) -> Dict[str, list]:
        """List capabilities of all agents"""
        return {
            "speech_processing": self.speech_agent.capabilities,
            "chatterbox_tts": self.chatterbox_agent.capabilities,
            "dialogue_generator": self.dialogue_agent.capabilities
        }

    async def route_request(self, agent_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to the appropriate agent"""
        agents = {
            "speech_processing": self.speech_agent,
            "chatterbox_tts": self.chatterbox_agent,
            "dialogue_generator": self.dialogue_agent
        }

        agent = agents.get(agent_id)
        if not agent:
            return {"success": False, "error": f"Unknown agent: {agent_id}"}

        return await agent.process_request(request)