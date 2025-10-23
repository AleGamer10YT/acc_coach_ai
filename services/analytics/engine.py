from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from shared.schemas.feedback import FeedbackEvent, FeedbackSeverity
from shared.schemas.telemetry import TelemetryFrame, TrackSectionType
from shared.utils.logging import configure_logging
from shared.utils.message_bus import TelemetryPublisher

from .database import get_session
from .repository import TelemetryRepository

logger = configure_logging("analytics.engine")

FeedbackCallback = Callable[[FeedbackEvent], None]

SECTION_METADATA = [
    ("S1", "Curva 1", TrackSectionType.CORNER),
    ("S2", "Curva 3", TrackSectionType.CORNER),
    ("S3", "Curva 5", TrackSectionType.CORNER),
    ("S4", "Curva 7", TrackSectionType.CORNER),
    ("S5", "Curva 10", TrackSectionType.CORNER),
]


@dataclass
class FrameSnapshot:
    timestamp: float
    speed_kph: float
    throttle: float
    brake: float
    steering: float
    delta_best_ms: Optional[float]
    track_pos: float


@dataclass
class SessionState:
    session_id: str
    current_lap: int
    frames: Deque[FrameSnapshot] = field(default_factory=lambda: deque(maxlen=180))
    last_feedback: Dict[str, float] = field(default_factory=dict)
    section_samples: Dict[int, List[FrameSnapshot]] = field(
        default_factory=lambda: {i: [] for i in range(5)}
    )
    last_spline: float = 0.0
    last_frame: Optional[TelemetryFrame] = None

    def remember_feedback(self, key: str, cool_down: float) -> bool:
        now = time.time()
        last = self.last_feedback.get(key)
        if last and now - last < cool_down:
            return False
        self.last_feedback[key] = now
        return True


class RealtimeAnalyticsEngine:
    def __init__(self, on_feedback: FeedbackCallback) -> None:
        self.on_feedback = on_feedback
        self._publisher = TelemetryPublisher()
        self._task: Optional[asyncio.Task] = None
        self._state: Dict[str, SessionState] = {}
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        await self._publisher.connect()
        self._stop_event.clear()
        self._task = asyncio.create_task(self._consume())
        logger.info("Realtime analytics avviato")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
        for state in self._state.values():
            await self._finalize_lap(state)
        self._state.clear()
        logger.info("Realtime analytics fermato")

    async def _consume(self) -> None:
        async with self._publisher.subscribe() as subscriber:
            await subscriber.listen(self._handle_payload)

    async def _handle_payload(self, payload: dict) -> None:
        if self._stop_event.is_set():
            return

        frame = TelemetryFrame(**payload)
        state = self._state.setdefault(
            frame.session_id,
            SessionState(session_id=frame.session_id, current_lap=frame.lap),
        )
        if frame.lap != state.current_lap:
            await self._finalize_lap(state)
            state.current_lap = frame.lap
            state.section_samples = {i: [] for i in range(5)}

        state.frames.append(
            FrameSnapshot(
                timestamp=frame.timestamp,
                speed_kph=frame.speed_kph,
                throttle=frame.throttle,
                brake=frame.brake,
                steering=frame.steering,
                delta_best_ms=frame.delta_best_ms,
                track_pos=frame.track_spline_pos,
            )
        )
        section_idx = min(int(frame.track_spline_pos * 5), 4)
        state.section_samples.setdefault(section_idx, []).append(state.frames[-1])
        state.last_spline = frame.track_spline_pos

        await self._persist(frame)
        await self._evaluate(frame, state)
        state.last_frame = frame

    async def _persist(self, frame: TelemetryFrame) -> None:
        async with get_session() as session:
            repo = TelemetryRepository(session)
            await repo.persist_frame(frame)
            await session.commit()

    async def _evaluate(self, frame: TelemetryFrame, state: SessionState) -> None:
        await self._check_brake_bias(frame, state)
        await self._check_exit_speed(frame, state)
        await self._check_throttle_steering(frame, state)

    async def _finalize_lap(self, state: SessionState) -> None:
        if not state.last_frame:
            return

        lap_number = state.current_lap
        logger.debug("Finalizzazione lap %s per sessione %s", lap_number, state.session_id)

        sections_payload = []
        for idx, samples in state.section_samples.items():
            if not samples:
                continue
            duration_ms = (samples[-1].timestamp - samples[0].timestamp) * 1000
            avg_speed = sum(s.speed_kph for s in samples) / len(samples)
            avg_throttle = sum(s.throttle for s in samples) / len(samples)
            avg_brake = sum(s.brake for s in samples) / len(samples)
            avg_steer = sum(s.steering for s in samples) / len(samples)
            section_id, name, section_type = SECTION_METADATA[idx]
            sections_payload.append(
                {
                    "section_id": section_id,
                    "name": name,
                    "section_type": section_type.value,
                    "delta_time_ms": duration_ms,
                    "avg_speed": avg_speed,
                    "throttle_avg": avg_throttle,
                    "brake_avg": avg_brake,
                    "steering_avg": avg_steer,
                }
            )

        async with get_session() as session:
            repo = TelemetryRepository(session)
            await repo.finalize_lap(state.session_id, lap_number, state.last_frame.lap_time_ms)
            if sections_payload:
                await repo.save_sections(state.session_id, lap_number, sections_payload)
            await session.commit()

    async def _check_brake_bias(self, frame: TelemetryFrame, state: SessionState) -> None:
        if frame.brake < 0.7 or frame.speed_kph < 120:
            return
        if len(state.frames) < 10:
            return

        recent = list(state.frames)[-10:]
        speed_drop = recent[0].speed_kph - recent[-1].speed_kph
        if speed_drop < 35:
            return

        if not state.remember_feedback("early_brake", 4.0):
            return

        msg = "Stai frenando troppo presto: prova a spostare il punto di staccata piu avanti."
        event = FeedbackEvent(
            session_id=frame.session_id,
            lap=frame.lap,
            section=self._guess_section(frame.track_spline_pos),
            severity=FeedbackSeverity.WARNING,
            message=msg,
            metrics={
                "speed_drop_kph": round(speed_drop, 1),
                "brake": round(frame.brake, 2),
            },
            timestamp=time.time(),
        )
        self.on_feedback(event)

    async def _check_exit_speed(self, frame: TelemetryFrame, state: SessionState) -> None:
        if len(state.frames) < 30:
            return
        recent = list(state.frames)[-30:]
        steering_mean = sum(abs(f.steering) for f in recent) / len(recent)
        throttle_mean = sum(f.throttle for f in recent) / len(recent)
        speed_mean = sum(f.speed_kph for f in recent[-10:]) / 10
        speed_prev = sum(f.speed_kph for f in recent[:10]) / 10

        if steering_mean < 0.15:
            return  # non siamo in curva

        if speed_mean + 5 >= speed_prev:
            return

        if not state.remember_feedback("slow_exit", 6.0):
            return

        msg = "Lavora sull'uscita di curva: applica gas piu progressivo per mantenere velocita."
        event = FeedbackEvent(
            session_id=frame.session_id,
            lap=frame.lap,
            section=self._guess_section(frame.track_spline_pos),
            severity=FeedbackSeverity.SUGGESTION,
            message=msg,
            metrics={
                "speed_drop_kph": round(speed_prev - speed_mean, 1),
                "avg_throttle": round(throttle_mean, 2),
            },
            timestamp=time.time(),
        )
        self.on_feedback(event)

    async def _check_throttle_steering(self, frame: TelemetryFrame, state: SessionState) -> None:
        if frame.throttle < 0.9 or abs(frame.steering) < 0.5:
            return
        if not state.remember_feedback("oversteer_risk", 5.0):
            return

        msg = "Riduci leggermente il gas mentre sterzi: eviti sovrasterzo e preservi gli pneumatici."
        event = FeedbackEvent(
            session_id=frame.session_id,
            lap=frame.lap,
            section=self._guess_section(frame.track_spline_pos),
            severity=FeedbackSeverity.INFO,
            message=msg,
            metrics={
                "steering": round(frame.steering, 2),
                "throttle": round(frame.throttle, 2),
            },
            timestamp=time.time(),
        )
        self.on_feedback(event)

    def _guess_section(self, spline_pos: float) -> Optional[str]:
        if spline_pos < 0.2:
            return "Curva 1"
        if spline_pos < 0.4:
            return "Curva 3"
        if spline_pos < 0.6:
            return "Curva 5"
        if spline_pos < 0.8:
            return "Curva 7"
        return "Curva 10"
