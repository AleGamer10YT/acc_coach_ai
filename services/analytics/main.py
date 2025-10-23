from __future__ import annotations

import asyncio
from typing import List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.analytics.engine import RealtimeAnalyticsEngine
from services.analytics.models import Lap, LapSection, Session, TelemetryPoint
from shared.schemas.feedback import FeedbackEvent
from shared.schemas.session import SessionMetadata, SessionReport
from shared.schemas.telemetry import LapSummary, SectionMetrics
from shared.utils.config import get_settings
from shared.utils.logging import configure_logging

from .database import get_session, init_db

settings = get_settings()
logger = configure_logging("analytics.api", settings.log_level)

app = FastAPI(
    title="ACC Coach Analytics API",
    version="0.1.0",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackManager:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()
        self.history: List[FeedbackEvent] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)
        for event in self.history[-20:]:
            await websocket.send_json(event.dict())

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, event: FeedbackEvent) -> None:
        self.history.append(event)
        payload = event.dict()
        disconnected: List[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_json(payload)
            except WebSocketDisconnect:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)

    def handle_feedback(self, event: FeedbackEvent) -> None:
        asyncio.create_task(self.broadcast(event))


feedback_manager = FeedbackManager()
analytics_engine = RealtimeAnalyticsEngine(on_feedback=feedback_manager.handle_feedback)


@app.on_event("startup")
async def bootstrap() -> None:
    await init_db()
    await analytics_engine.start()
    logger.info("Analytics API pronta")


@app.on_event("shutdown")
async def shutdown() -> None:
    await analytics_engine.stop()


async def get_db_session() -> AsyncSession:
    async with get_session() as session:
        yield session


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions", response_model=List[SessionMetadata])
async def list_sessions(db: AsyncSession = Depends(get_db_session)) -> List[SessionMetadata]:
    stmt = select(Session)
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [
        SessionMetadata(
            id=s.id,
            driver_id=s.driver_id,
            track_id=s.track_id,
            track_name=s.track_name,
            car_model=s.car_model,
            started_at=s.started_at,
            finished_at=s.finished_at,
            fastest_lap_id=str(s.fastest_lap_id) if s.fastest_lap_id else None,
            consistency_score=s.consistency_score,
            efficiency_score=s.efficiency_score,
            notes=s.notes,
        )
        for s in sessions
    ]


@app.get("/sessions/{session_id}", response_model=SessionReport)
async def get_session_report(
    session_id: str, db: AsyncSession = Depends(get_db_session)
) -> SessionReport:
    session = await db.get(Session, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessione non trovata")

    laps_stmt = select(Lap).where(Lap.session_id == session_id).order_by(Lap.lap_number)
    laps_result = await db.execute(laps_stmt)
    laps = laps_result.scalars().all()

    lap_summaries: List[LapSummary] = []
    for lap in laps:
        sectors = [section.delta_time_ms for section in lap.sections]
        lap_summaries.append(
            LapSummary(
                session_id=session_id,
                lap=lap.lap_number,
                lap_time_ms=lap.lap_time_ms or 0.0,
                is_best=lap.is_best,
                sectors=sectors,
                track_name=session.track_name,
                car_model=session.car_model,
                notes=lap.notes,
            )
        )

    critical_sections: List[SectionMetrics] = []
    for lap in laps:
        for section in lap.sections:
            if section.delta_time_ms > 50:
                critical_sections.append(
                    SectionMetrics(
                        section_id=section.section_id,
                        section_name=section.name,
                        section_type=section.section_type,  # type: ignore[arg-type]
                        start_spline=0.0,
                        end_spline=0.0,
                        lap=lap.lap_number,
                        delta_time_ms=section.delta_time_ms,
                        avg_speed_kph=section.avg_speed,
                        throttle_avg=section.throttle_avg,
                        brake_avg=section.brake_avg,
                        steering_avg=section.steering_avg,
                        gear_mode=0,
                    )
                )

    report = SessionReport(
        session=SessionMetadata(
            id=session.id,
            driver_id=session.driver_id,
            track_id=session.track_id,
            track_name=session.track_name,
            car_model=session.car_model,
            started_at=session.started_at,
            finished_at=session.finished_at,
            fastest_lap_id=str(session.fastest_lap_id) if session.fastest_lap_id else None,
            consistency_score=session.consistency_score,
            efficiency_score=session.efficiency_score,
            notes=session.notes,
        ),
        laps=lap_summaries,
        critical_sections=critical_sections,
        time_lost_ms=sum(section.delta_time_ms for section in critical_sections),
        comparison_reference=None,
    )
    return report


@app.get("/sessions/{session_id}/laps", response_model=List[LapSummary])
async def get_laps(session_id: str, db: AsyncSession = Depends(get_db_session)) -> List[LapSummary]:
    stmt = select(Lap).where(Lap.session_id == session_id).order_by(Lap.lap_number)
    result = await db.execute(stmt)
    laps = result.scalars().all()
    return [
        LapSummary(
            session_id=session_id,
            lap=lap.lap_number,
            lap_time_ms=lap.lap_time_ms or 0.0,
            is_best=lap.is_best,
            sectors=[section.delta_time_ms for section in lap.sections],
            track_name=lap.session.track_name if lap.session else "",
            car_model=lap.session.car_model if lap.session else "",
            notes=lap.notes,
        )
        for lap in laps
    ]


@app.get("/laps/{lap_id}/sections", response_model=List[SectionMetrics])
async def get_sections(lap_id: int, db: AsyncSession = Depends(get_db_session)) -> List[SectionMetrics]:
    lap = await db.get(Lap, lap_id)
    if not lap:
        raise HTTPException(status_code=404, detail="Lap non trovato")
    return [
        SectionMetrics(
            section_id=section.section_id,
            section_name=section.name,
            section_type=section.section_type,  # type: ignore[arg-type]
            start_spline=0.0,
            end_spline=0.0,
            lap=lap.lap_number,
            delta_time_ms=section.delta_time_ms,
            avg_speed_kph=section.avg_speed,
            throttle_avg=section.throttle_avg,
            brake_avg=section.brake_avg,
            steering_avg=section.steering_avg,
            gear_mode=0,
        )
        for section in lap.sections
    ]


@app.websocket("/ws/feedback")
async def feedback_ws(websocket: WebSocket) -> None:
    await feedback_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        feedback_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.analytics.main:app", host="0.0.0.0", port=8080, reload=False)
