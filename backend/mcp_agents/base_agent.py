from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPAgent(ABC):
    def __init__(self, agent_id: str, capabilities: list):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.logger = logger
        
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def validate_request(self, request: Dict[str, Any], required_fields: list) -> bool:
        for field in required_fields:
            if field not in request:
                self.logger.error(f"Missing required field: {field}")
                return False
        return True
    
    def create_response(self, success: bool, data: Any = None, error: str = None) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "success": success,
            "data": data,
            "error": error
        }