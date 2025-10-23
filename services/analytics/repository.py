from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.schemas.telemetry import TelemetryFrame
from shared.utils.logging import configure_logging

from .models import Lap, LapSection, Session, TelemetryPoint

logger = configure_logging("analytics.repository")


class TelemetryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def ensure_session(self, frame: TelemetryFrame) -> Session:
        stmt = select(Session).where(Session.id == frame.session_id)
        result = await self.session.execute(stmt)
        session = result.scalars().first()
        if session:
            return session

        session = Session(
            id=frame.session_id,
            driver_id=frame.driver_id,
            track_id=frame.track_name.lower().replace(" ", "_"),
            track_name=frame.track_name,
            car_model=frame.player_car,
            started_at=datetime.now(tz=timezone.utc),
        )
        self.session.add(session)
        await self.session.flush()
        logger.info("Creata nuova sessione %s", frame.session_id)
        return session

    async def ensure_lap(self, session_id: str, lap_number: int) -> Lap:
        stmt = (
            select(Lap)
            .where(Lap.session_id == session_id)
            .where(Lap.lap_number == lap_number)
        )
        result = await self.session.execute(stmt)
        lap = result.scalars().first()
        if lap:
            return lap

        lap = Lap(session_id=session_id, lap_number=lap_number, is_best=False)
        self.session.add(lap)
        await self.session.flush()
        return lap

    async def persist_frame(self, frame: TelemetryFrame) -> None:
        await self.ensure_session(frame)
        lap = await self.ensure_lap(frame.session_id, frame.lap)

        point = TelemetryPoint(
            lap_id=lap.id,
            timestamp=frame.timestamp,
            distance_m=frame.track_spline_pos,
            speed=frame.speed_kph,
            throttle=frame.throttle,
            brake=frame.brake,
            gear=frame.gear,
            steering=frame.steering,
            accel_lat=None,
            accel_long=None,
            extra={
                "clutch": frame.clutch,
                "lap_time_ms": frame.lap_time_ms,
                "tyre_temp": frame.tyres_core_temp,
                "brake_temp": frame.brake_temp,
            },
        )
        self.session.add(point)

        if frame.is_valid_lap is False and lap.notes is None:
            lap.notes = "Lap invalidated"

    async def finalize_lap(
        self, session_id: str, lap_number: int, lap_time_ms: Optional[int]
    ) -> None:
        lap = await self.ensure_lap(session_id, lap_number)
        if lap_time_ms:
            lap.lap_time_ms = lap_time_ms

    async def mark_session_finished(self, session_id: str) -> None:
        stmt = select(Session).where(Session.id == session_id)
        result = await self.session.execute(stmt)
        session = result.scalars().first()
        if session:
            session.finished_at = datetime.now(tz=timezone.utc)

    async def save_sections(
        self, session_id: str, lap_number: int, sections: list[dict]
    ) -> None:
        lap = await self.ensure_lap(session_id, lap_number)
        await self.session.execute(delete(LapSection).where(LapSection.lap_id == lap.id))
        for payload in sections:
            section = LapSection(
                lap_id=lap.id,
                section_id=payload["section_id"],
                name=payload["name"],
                section_type=payload["section_type"],
                delta_time_ms=payload["delta_time_ms"],
                avg_speed=payload["avg_speed"],
                throttle_avg=payload["throttle_avg"],
                brake_avg=payload["brake_avg"],
                steering_avg=payload["steering_avg"],
            )
            self.session.add(section)
