"""
Maintenance endpoints. Admin keys only.
"""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_memory_manager, get_scheduler
from backend.api.middleware.auth import require_admin
from backend.api.public.public_api import envelope
from backend.brain.memory import index_sync
from backend.utils.internal_client import EngineError

router = APIRouter(prefix="/internal", tags=["Internal System API"], dependencies=[Depends(require_admin)])


class ClearRequest(BaseModel):
    user_id: str  # a user id, or "all"


class ReindexRequest(BaseModel):
    force: bool = False
    backfill: bool = False


class JobRequest(BaseModel):
    job: Literal["state_inference", "prediction", "replay", "forgetting"]
    user_id: Optional[str] = None


@router.post("/clear")
def clear_memory(request: ClearRequest):
    """Deletes one user's data (or everyone's with user_id="all") from every store and the vector index."""
    mm = get_memory_manager()
    try:
        n = mm.clear_all() if request.user_id == "all" else mm.clear_user(request.user_id)
    except EngineError as e:
        raise HTTPException(status_code=503, detail=f"SQLite cleared but vector engine failed: {e}")
    return envelope({"user_id": request.user_id, "memories_deleted": n})


@router.post("/reindex")
def reindex(request: ReindexRequest):
    mm = get_memory_manager()
    try:
        report = {}
        if request.backfill:
            report["backfill"] = index_sync.backfill(mm.episodic, mm.semantic)
        report["sync"] = index_sync.sync(mm.episodic, mm.semantic, force=request.force)
    except EngineError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return envelope(report)


@router.post("/jobs/run")
def run_job(request: JobRequest):
    return envelope(get_scheduler().run_job(request.job, request.user_id))
