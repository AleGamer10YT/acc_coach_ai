from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from .telemetry import LapSummary, SectionMetrics


class SessionMetadata(BaseModel):
    id: str
    driver_id: str
    track_id: str
    track_name: str
    car_model: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    fastest_lap_id: Optional[str] = None
    consistency_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    notes: Optional[str] = None


class SessionReport(BaseModel):
    session: SessionMetadata
    laps: List[LapSummary]
    critical_sections: List[SectionMetrics]
    time_lost_ms: float = Field(..., description="Tempo perso totale rispetto al best lap")
    comparison_reference: Optional[str] = Field(
        None, description="Lap id di riferimento (utente o pro)."
    )
