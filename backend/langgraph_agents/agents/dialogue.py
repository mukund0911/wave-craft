"""
Dialogue Generator Agent (LangGraph refactor)
Uses OpenAI GPT to generate realistic dialogue.
"""
import os
import logging
from typing import Dict, Any, List
from .base import create_response

logger = logging.getLogger(__name__)


class DialogueGeneratorAgent:
    """Generates dialogue via OpenAI GPT."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set — dialogue generation will fail")
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def generate_dialogue(self, prompt: str, conversation_history: List[Dict] = None,
                          speaker_characteristics: str = "") -> Dict[str, Any]:
        if not self.client:
            return create_response("dialogue_generator", False, error="OpenAI client not initialized")

        context = self._build_context(conversation_history or [])
        system_prompt = f"""You are an AI assistant that generates realistic dialogue for conversations.

Context from previous conversation:
{context}

New speaker characteristics:
{speaker_characteristics}

Generate a natural response that:
1. Fits the conversation context
2. Matches the specified speaker characteristics (gender, emotion, topic preferences)
3. Continues the conversation naturally
4. Is between 10-50 words long

Return only the dialogue text, no additional formatting."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate the dialogue based on this prompt: {prompt}"}
                ],
                max_tokens=150,
                temperature=0.7
            )
            generated_text = response.choices[0].message.content.strip()
            return create_response("dialogue_generator", True, {
                "dialogue": generated_text,
                "speaker_characteristics": speaker_characteristics
            })
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return create_response("dialogue_generator", False, error=f"Failed to generate dialogue: {str(e)}")

    def _build_context(self, conversation_history: List[Dict]) -> str:
        context_parts = []
        for conv in conversation_history[-5:]:
            speaker = conv.get("speaker", "Unknown")
            text = conv.get("original", {}).get("text", "") or conv.get("modified", {}).get("text", "")
            context_parts.append(f"Speaker {speaker}: {text}")
        return "\n".join(context_parts)
