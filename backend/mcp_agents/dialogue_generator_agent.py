import openai
from typing import Dict, Any, List
from .base_agent import MCPAgent

class DialogueGeneratorAgent(MCPAgent):
    def __init__(self, api_key: str):
        super().__init__("dialogue_generator", ["generate_dialogue", "analyze_context"])
        if not api_key or api_key.strip() == "":
            raise ValueError("OpenAI API key is required")
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        
        if action == "generate_dialogue":
            return await self._generate_dialogue(request)
        elif action == "analyze_context":
            return await self._analyze_context(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")
    
    async def _generate_dialogue(self, request: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["conversation_history", "speaker_prompt"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        conversation_history = request["conversation_history"]
        speaker_prompt = request["speaker_prompt"]
        
        # Build context from conversation history
        context = self._build_conversation_context(conversation_history)
        
        # Create prompt for dialogue generation
        system_prompt = f"""You are an AI assistant that generates realistic dialogue for conversations.
        
Context from previous conversation:
{context}

New speaker characteristics:
{speaker_prompt}

Generate a natural response that:
1. Fits the conversation context
2. Matches the specified speaker characteristics (gender, emotion, topic preferences)
3. Continues the conversation naturally
4. Is between 10-50 words long

Return only the dialogue text, no additional formatting."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using more accessible model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the dialogue:"}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return self.create_response(True, {
                "generated_dialogue": generated_text,
                "speaker_characteristics": speaker_prompt
            })
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return self.create_response(False, error=f"Failed to generate dialogue: {str(e)}")
    
    async def _analyze_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        conversation_history = request.get("conversation_history", [])
        context_summary = self._build_conversation_context(conversation_history)
        
        return self.create_response(True, {
            "context_summary": context_summary,
            "speaker_count": len(set([conv.get("speaker", "Unknown") for conv in conversation_history]))
        })
    
    def _build_conversation_context(self, conversation_history: List[Dict]) -> str:
        context_parts = []
        
        for i, conv in enumerate(conversation_history[-5:]):  # Use last 5 conversations for context
            speaker = conv.get("speaker", "Unknown")
            text = conv.get("original", {}).get("text", "") or conv.get("modified", {}).get("text", "")
            context_parts.append(f"Speaker {speaker}: {text}")
        
        return "\n".join(context_parts)