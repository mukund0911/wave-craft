"""
Lightweight base utilities for WaveCraft LangGraph agents.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def create_response(agent_id: str, success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
    """Standard response envelope (kept for compat with existing routes)."""
    return {
        "agent_id": agent_id,
        "success": success,
        "data": data,
        "error": error,
    }
