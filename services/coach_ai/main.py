from __future__ import annotations

import asyncio
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.coach_ai.coaching_engine import CoachingEngine
from services.coach_ai.schemas import BatchReportRequest, CoachingRequest, CoachingResponse
from services.coach_ai.tts import TTSService
from shared.utils.config import get_settings
from shared.utils.logging import configure_logging

settings = get_settings()
logger = configure_logging("coach_ai.api", settings.log_level)

app = FastAPI(
    title="ACC Coach AI",
    version="0.1.0",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_service = TTSService()
engine = CoachingEngine(tts_service=tts_service)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/coach/realtime", response_model=CoachingResponse)
async def realtime_feedback(payload: CoachingRequest) -> CoachingResponse:
    logger.info("Richiesta feedback realtime session=%s lap=%s", payload.session_id, payload.lap)
    return await engine.generate_feedback(payload)


@app.post("/coach/report", response_model=List[CoachingResponse])
async def batch_report(payload: BatchReportRequest) -> List[CoachingResponse]:
    responses: List[CoachingResponse] = []
    for lap in payload.laps:
        request = CoachingRequest(
            session_id=payload.session_id,
            lap=lap,
            issues=payload.key_findings,
            summary=f"Analisi batch per lap {lap}",
        )
        responses.append(await engine.generate_feedback(request))
    return responses


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.coach_ai.main:app", host="0.0.0.0", port=8082, reload=False)
