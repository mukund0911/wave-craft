"""
LangGraph builders for WaveCraft agent workflows.
"""
from langgraph.graph import StateGraph, END

from .state import TranscriptionState, ModificationState, ArtificialSpeakerState
from .nodes import (
    optimize_audio_node,
    transcribe_node,
    parse_changes_node,
    synthesize_segments_node,
    assemble_audio_node,
    quality_check_node,
    build_stats_node,
    s3_upload_node,
    generate_dialogue_node,
    select_tts_engine_node,
    synthesize_speech_node,
    error_handler_node,
    has_error,
    needs_retry,
)


# ──────────────────────────────────────────────────
# Transcription Graph
# ──────────────────────────────────────────────────

def build_transcription_graph():
    """Audio -> optimized audio -> transcribe -> conversations."""
    graph = StateGraph(TranscriptionState)
    graph.add_node("optimize_audio", optimize_audio_node)
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("optimize_audio")
    graph.add_conditional_edges(
        "optimize_audio",
        has_error,
        {"error": "error_handler", "continue": "transcribe"}
    )
    graph.add_conditional_edges(
        "transcribe",
        has_error,
        {"error": "error_handler", "continue": END}
    )
    graph.add_edge("error_handler", END)

    return graph.compile()


# ──────────────────────────────────────────────────
# Modification Graph
# ──────────────────────────────────────────────────

def build_modification_graph():
    """Edited transcripts -> parse -> synthesize -> assemble -> quality check -> stats -> S3."""
    graph = StateGraph(ModificationState)
    graph.add_node("parse_changes", parse_changes_node)
    graph.add_node("synthesize_segments", synthesize_segments_node)
    graph.add_node("assemble_audio", assemble_audio_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("build_stats", build_stats_node)
    graph.add_node("s3_upload", s3_upload_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("parse_changes")
    graph.add_conditional_edges(
        "parse_changes",
        has_error,
        {"error": "error_handler", "continue": "synthesize_segments"}
    )
    graph.add_conditional_edges(
        "synthesize_segments",
        has_error,
        {"error": "error_handler", "continue": "assemble_audio"}
    )
    graph.add_conditional_edges(
        "assemble_audio",
        has_error,
        {"error": "error_handler", "continue": "quality_check"}
    )
    graph.add_conditional_edges(
        "quality_check",
        needs_retry,
        {"retry": "synthesize_segments", "continue": "build_stats"}
    )
    graph.add_edge("build_stats", "s3_upload")
    graph.add_edge("s3_upload", END)
    graph.add_edge("error_handler", END)

    return graph.compile()


# ──────────────────────────────────────────────────
# Artificial Speaker Graph
# ──────────────────────────────────────────────────

def build_artificial_speaker_graph():
    """Prompt -> GPT dialogue -> select TTS -> synthesize."""
    graph = StateGraph(ArtificialSpeakerState)
    graph.add_node("generate_dialogue", generate_dialogue_node)
    graph.add_node("select_tts_engine", select_tts_engine_node)
    graph.add_node("synthesize_speech", synthesize_speech_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("generate_dialogue")
    graph.add_conditional_edges(
        "generate_dialogue",
        has_error,
        {"error": "error_handler", "continue": "select_tts_engine"}
    )
    graph.add_conditional_edges(
        "select_tts_engine",
        has_error,
        {"error": "error_handler", "continue": "synthesize_speech"}
    )
    graph.add_conditional_edges(
        "synthesize_speech",
        has_error,
        {"error": "error_handler", "continue": END}
    )
    graph.add_edge("error_handler", END)

    return graph.compile()
