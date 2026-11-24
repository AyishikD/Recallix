"""
Multi-agent endpoints. A memory is visible to an agent if it is `shared` or the
agent's own `private` memory (enforced in SQL by EpisodicMemory).
"""
from typing import Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from backend.agents.registry.manager import AgentRegistry
from backend.api.deps import get_memory_manager
from backend.api.middleware.auth import verify_api_key, acting_user, rate_limit_writes
from backend.api.public.public_api import envelope

router = APIRouter(prefix="/agents", tags=["Multi-Agent Memory API"])


class AgentStoreRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    agent_id: str = Field(min_length=1)
    memory_type: Literal["private", "shared"] = "private"
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


class AgentRecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    agent_id: str = Field(min_length=1)
    user_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)
    deep: Optional[bool] = None  # re-read candidates with the re-ranker (see /memory/recall)


@router.post("/store")
def store_agent_memory(request: AgentStoreRequest, auth=Depends(verify_api_key)):
    user_id = acting_user(auth, request.user_id)
    rate_limit_writes(user_id)
    result = get_memory_manager().store(user_id, request.content, agent_id=request.agent_id,
                                        memory_type=request.memory_type, client_metadata=request.metadata)
    return envelope({"status": result["status"], "memory_id": result.get("memory_id"), "agent_id": request.agent_id,
                     "memory_type": request.memory_type, "importance": result.get("importance"),
                     "total_ms": result.get("total_ms")})


@router.post("/recall")
def recall_agent_memory(request: AgentRecallRequest, auth=Depends(verify_api_key)):
    user_id = acting_user(auth, request.user_id)
    data = get_memory_manager().retrieve(user_id, request.query, agent_id=request.agent_id, limit=request.limit,
                                         deep=request.deep)
    data["agent_id"] = request.agent_id
    return envelope(data)


@router.get("/list")
def list_registered_agents(auth=Depends(verify_api_key)):
    return envelope(AgentRegistry().list_agents())
