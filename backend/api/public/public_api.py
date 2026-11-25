"""
Public memory API. Every endpoint is authenticated; `user_id` in a request is
honoured only for service/admin keys (see backend/api/middleware/auth.py).

Handlers are plain `def` so FastAPI runs them in its thread pool — the
blocking SQLite / engine work never stalls the event loop.
"""
import json
import time
from typing import Annotated, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AfterValidator, BaseModel, Field

from backend import settings
from backend.api.deps import get_memory_manager
from backend.api.middleware.auth import (ID_PATTERN, acting_user, one_import_at_a_time, rate_limit_reads,
                                         rate_limit_writes, verify_api_key)
from backend.utils.internal_client import engine_get, EngineError

router = APIRouter(prefix="/memory", tags=["Public Memory API"])


# Request bounds (config.yaml api.*). Ids end up in index names, SQL rows and logs.
UserId = Optional[Annotated[str, Field(max_length=128, pattern=ID_PATTERN)]]
AgentId = Annotated[str, Field(min_length=1, max_length=128, pattern=ID_PATTERN)]


def _bounded_metadata(value):
    """Metadata is stored with every memory: at most api.max_metadata_bytes as JSON."""
    if value is not None:
        limit = int(settings.get("api.max_metadata_bytes", 16384))
        if len(json.dumps(value, default=str)) > limit:
            raise ValueError(f"metadata is larger than {limit} bytes as JSON")
    return value


Metadata = Annotated[Optional[dict], AfterValidator(_bounded_metadata)]


def envelope(data):
    return {"status": "success", "data": data, "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


class StoreRequest(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    user_id: UserId = None
    agent_id: AgentId = "default_agent"
    memory_type: Literal["private", "shared"] = "private"
    metadata: Metadata = None
    # accepted for backward compatibility; no effect (no LLM; writes are searchable immediately)
    skip_llm: Optional[bool] = None
    sync_index: Optional[bool] = None


class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: UserId = None
    agent_id: AgentId = "default_agent"
    limit: int = Field(default=10, ge=1, le=100)
    # Deep recall: re-read the candidates with the re-ranker (recall.rerank). Better when the answer is
    # spread over several memories ("what activities has she done with her kids?"); slower. None = config.
    deep: Optional[bool] = None
    # A decline (status ABSTAINED) still returns the ranked results, low-confidence, answer null. Only for
    # callers that check `status`. None = config (recall.keep_results_on_decline, off by default).
    keep_on_decline: Optional[bool] = None


class ProactiveRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    user_id: UserId = None
    agent_id: AgentId = "default_agent"
    limit: int = Field(default=5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    recall_id: str = Field(max_length=64)
    relevant_memory_ids: List[int] = Field(default=[], max_length=100)
    user_id: UserId = None


@router.post("/store")
def store_memory(request: StoreRequest, auth=Depends(verify_api_key)):
    user_id = acting_user(auth, request.user_id)
    rate_limit_writes(user_id)
    result = get_memory_manager().store(user_id, request.content, agent_id=request.agent_id,
                                        memory_type=request.memory_type, client_metadata=request.metadata)
    return envelope({k: result.get(k) for k in
                     ("status", "memory_id", "importance", "schemas", "promoted", "indexed", "conflicts", "total_ms", "processing_log")})


@router.post("/recall")
def recall_memory(request: RecallRequest, auth=Depends(verify_api_key)):
    user_id = acting_user(auth, request.user_id)
    rate_limit_reads(user_id, deep=bool(request.deep))
    return envelope(get_memory_manager().retrieve(user_id, request.query, agent_id=request.agent_id, limit=request.limit,
                                                  deep=request.deep, keep_on_decline=request.keep_on_decline))


@router.post("/proactive")
def proactive_memories(request: ProactiveRequest, auth=Depends(verify_api_key)):
    """Memories an agent should keep in mind for this message (e.g. an allergy when food is mentioned)."""
    user_id = acting_user(auth, request.user_id)
    rate_limit_reads(user_id)
    return envelope(get_memory_manager().proactive(user_id, request.message, request.agent_id, request.limit))


@router.post("/feedback")
def recall_feedback(request: FeedbackRequest, auth=Depends(verify_api_key)):
    """
    Marks which memories of a recall were useful: labels ranker training data (fit_ranker --from-db)
    and remembers the confirmed answer for similar future questions (recall.distillation).
    """
    user_id = acting_user(auth, request.user_id)
    n = get_memory_manager().recall_engine.confirm(request.recall_id, user_id, request.relevant_memory_ids)
    if n == 0:
        raise HTTPException(status_code=404, detail="Unknown recall_id for this user")
    return envelope({"labelled_candidates": n})


class ImportRequest(BaseModel):
    format: Literal["recallix", "mem0", "text"] = "recallix"
    data: Union[dict, list, str]
    user_id: UserId = None
    agent_id: AgentId = "default_agent"
    memory_type: Literal["private", "shared"] = "private"


@router.get("/export")
def export_memories(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    """All of the user's memories in the open "recallix-memory" format (backend/tools/portable.py)."""
    from backend.tools.portable import export_user
    return envelope(export_user(get_memory_manager(), acting_user(auth, user_id)))


@router.post("/import")
def import_memories(request: ImportRequest, auth=Depends(verify_api_key)):
    """
    Imports a Recallix export, a Mem0 export, or plain text (one memory per line). Every memory
    goes through the normal store pipeline with its original timestamp; duplicates are skipped.
    """
    from backend.tools.portable import import_items, parse
    user_id = acting_user(auth, request.user_id)
    rate_limit_writes(user_id)
    try:
        items = parse(request.data, request.format)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    cap = int(settings.get("api.max_import_items", 5000))
    if len(items) > cap:
        raise HTTPException(status_code=413, detail=f"{len(items)} memories in one import; at most {cap} (split it up)")
    with one_import_at_a_time(user_id):  # one bulk store per user at a time, so imports can't bypass the write limit
        return envelope(import_items(get_memory_manager(), user_id, items, request.agent_id, request.memory_type))


@router.delete("/{memory_id}")
def delete_memory(memory_id: int, user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    uid = acting_user(auth, user_id)
    receipt = get_memory_manager().delete(uid, memory_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return envelope({"deleted": memory_id, "receipt": receipt})


@router.get("/deletions")
def deletion_receipts(user_id: Optional[str] = None, limit: int = Query(100, ge=1, le=1000), auth=Depends(verify_api_key)):
    """This user's deletion receipts from the tamper-evident log (newest first)."""
    return envelope(get_memory_manager().audit.receipts(acting_user(auth, user_id), limit))


@router.get("/timeline")
def get_timeline(user_id: Optional[str] = None, limit: int = Query(50, ge=1, le=1000), auth=Depends(verify_api_key)):
    return envelope(get_memory_manager().timeline.get_sequence(acting_user(auth, user_id), limit))


@router.get("/insights")
def get_insights(user_id: Optional[str] = None, limit: int = Query(5, ge=1, le=100), auth=Depends(verify_api_key)):
    return envelope(get_memory_manager().reflective.get_reflections(acting_user(auth, user_id), limit))


@router.get("/world-state")
def get_world_state(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    return envelope(get_memory_manager().state_inference.get_states(acting_user(auth, user_id)))


@router.get("/prediction")
def get_prediction(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    return envelope(get_memory_manager().prediction_engine.predict(acting_user(auth, user_id)))


@router.get("/plan")
def get_plan(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    mm = get_memory_manager()
    uid = acting_user(auth, user_id)
    return envelope(mm.planning_engine.generate_plan(uid, mm.prediction_engine.predict(uid)))


@router.get("/stats")
def get_stats(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    mm = get_memory_manager()
    uid = acting_user(auth, user_id)
    return envelope({
        "user_id": uid,
        "episodic_memories": mm.episodic.count(uid),
        "timeline_events": mm.timeline.count(uid),
        "reflections": mm.reflective.count(uid),
        "archived_memories": mm.archive.count(uid),
        "world_state_keys": len(mm.state_inference.get_states(uid)),
    })


@router.get("/system/stats")
def get_system_stats(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    uid = acting_user(auth, user_id)
    try:
        status = engine_get("/status")
        namespace = engine_get("/namespaces").get(uid)
    except EngineError:
        raise HTTPException(status_code=503, detail="vector engine unavailable")
    if auth.get("role") != "admin":  # engine-wide counts (how many users, pending writes) are for admins
        status = {k: status.get(k) for k in ("status", "simd", "defaults") if k in status}
    return envelope({"engine": status, "user_index": namespace})


@router.get("/profile")
def get_profile(user_id: Optional[str] = None, auth=Depends(verify_api_key)):
    mm = get_memory_manager()
    uid = acting_user(auth, user_id)
    return envelope({"user_id": uid, "world_states": mm.state_inference.get_states(uid),
                     "top_reflections": mm.reflective.get_reflections(uid, limit=3)})
