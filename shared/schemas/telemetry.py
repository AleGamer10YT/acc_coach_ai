from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SessionPhase(str, Enum):
    PRACTICE = "practice"
    QUALIFYING = "qualifying"
    RACE = "race"
    HOTSTINT = "hotstint"
    HOTLAP = "hotlap"
    OTHER = "other"


class TrackSectionType(str, Enum):
    STRAIGHT = "straight"
    CORNER = "corner"
    CHICANE = "chicane"
    UNKNOWN = "unknown"


class TelemetryFrame(BaseModel):
    session_id: str = Field(..., description="UUID della sessione corrente")
    driver_id: str = Field(..., description="Identificativo univoco pilota")
    timestamp: float = Field(..., description="Timestamp in secondi (epoch)")
    lap: int = Field(..., ge=0)
    lap_time_ms: Optional[int] = Field(None, description="Tempo giro corrente in millisecondi")
    speed_kph: float = Field(..., ge=0)
    throttle: float = Field(..., ge=0.0, le=1.0)
    brake: float = Field(..., ge=0.0, le=1.0)
    clutch: Optional[float] = Field(None, ge=0.0, le=1.0)
    steering: float = Field(..., ge=-1.0, le=1.0, description="-1 left, +1 right")
    gear: int = Field(..., ge=-1)
    world_pos_x: float
    world_pos_y: float
    world_pos_z: float
    track_spline_pos: float = Field(..., ge=0.0, le=1.0)
    player_car: str = Field(..., description="Modello vettura")
    track_name: str
    session_phase: SessionPhase
    is_valid_lap: Optional[bool] = True
    best_lap_time_ms: Optional[int] = None
    delta_best_ms: Optional[float] = None
    tyres_core_temp: Optional[List[float]] = None
    brake_temp: Optional[List[float]] = None


class SectionMetrics(BaseModel):
    section_id: str
    section_name: str
    section_type: TrackSectionType = TrackSectionType.UNKNOWN
    start_spline: float
    end_spline: float
    lap: int
    delta_time_ms: float
    avg_speed_kph: float
    throttle_avg: float
    brake_avg: float
    steering_avg: float
    gear_mode: float


class LapSummary(BaseModel):
    session_id: str
    lap: int
    lap_time_ms: float
    is_best: bool
    sectors: List[float]
    track_name: str
    car_model: str
    consistency_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    notes: Optional[str] = None
