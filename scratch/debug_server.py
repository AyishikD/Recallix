# LEGACY SCRIPT: written for the pre-2026-11 API and engine protocol; kept for reference only.
# Maintained equivalents: eval/run_eval.py (retrieval, gauntlet, retention), backend/infra_cpp/benchmark_hnsw,
# examples/multi_agent_handover.py. Keys come from RECALLIX_API_KEY / RECALLIX_INTERNAL_KEY.
from fastapi import FastAPI, Request
import traceback
from backend.api.public.public_api import router as public_router

app = FastAPI()
app.include_router(public_router, prefix="/memory")

@app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
