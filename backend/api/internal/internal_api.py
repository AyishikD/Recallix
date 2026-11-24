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
    job: Literal["state_inference", "prediction", "replay", "forgetting", "learning"]
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


@router.get("/learning")
def learning_status():
    """What the deployment has learned from feedback: every automatic refit attempt and its outcome."""
    from backend.brain.features import answer_check_learning
    from backend.brain.memory_ranking import self_improve
    return envelope({**self_improve.status(), "settings": {**self_improve.config(), "answer_check": answer_check_learning.config()}})


@router.post("/learning/rollback")
def learning_rollback(target: str = "ranker"):
    """
    Undoes the last automatic adoption: `target=ranker` (default) returns the ranker to the previous model,
    `target=answer_check` the answer check to its previous threshold.
    """
    if target == "answer_check":
        from backend.brain.features import answer_check_learning
        return envelope(answer_check_learning.rollback())
    if target != "ranker":
        raise HTTPException(status_code=422, detail="target must be 'ranker' or 'answer_check'")
    from backend.brain.memory_ranking import self_improve
    return envelope(self_improve.rollback())


@router.post("/jobs/run")
def run_job(request: JobRequest):
    return envelope(get_scheduler().run_job(request.job, request.user_id))
