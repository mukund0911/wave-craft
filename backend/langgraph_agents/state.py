"""
LangGraph state schemas for WaveCraft agent workflows.
"""
from typing import Any, Dict, List, Optional, TypedDict


class TranscriptionState(TypedDict, total=False):
    """State for the transcription graph."""
    audio_path: str
    original_audio_path: str
    num_speakers: Optional[int]
    conversations: List[Dict[str, Any]]
    full_audio: str
    language: str
    error: Optional[str]


class ModificationState(TypedDict, total=False):
    """State for the modification / voice-cloning graph."""
    conversations: List[Dict[str, Any]]
    full_audio: str
    segments: List[Dict[str, Any]]
    final_audio: str
    segments_processed: int
    segments_changed: int
    segments_failed: int
    failures: List[Dict[str, Any]]
    stats: Dict[str, Any]
    s3_url: Optional[str]
    error: Optional[str]
    retry_count: int
    retry_params: Dict[str, Any]


class ArtificialSpeakerState(TypedDict, total=False):
    """State for the artificial-speaker dialogue + TTS graph."""
    prompt: str
    speaker_characteristics: str
    speaker_name: str
    exaggeration: float
    use_openai_tts: bool
    conversation_history: List[Dict[str, Any]]
    generated_dialogue: str
    generated_audio: str
    voice_used: str
    error: Optional[str]
