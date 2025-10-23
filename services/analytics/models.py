from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    driver_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[str] = mapped_column(String(64), index=True)
    track_name: Mapped[str] = mapped_column(String(128))
    car_model: Mapped[str] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    fastest_lap_id: Mapped[Optional[int]] = mapped_column(ForeignKey("laps.id"))
    consistency_score: Mapped[Optional[float]] = mapped_column(Float)
    efficiency_score: Mapped[Optional[float]] = mapped_column(Float)
    notes: Mapped[Optional[str]] = mapped_column(Text)

    laps: Mapped[List["Lap"]] = relationship("Lap", back_populates="session")


class Lap(Base):
    __tablename__ = "laps"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    lap_number: Mapped[int] = mapped_column(Integer, index=True)
    lap_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    weather: Mapped[Optional[str]] = mapped_column(String(64))
    tyre_compound: Mapped[Optional[str]] = mapped_column(String(64))
    notes: Mapped[Optional[str]] = mapped_column(Text)

    session: Mapped["Session"] = relationship("Session", back_populates="laps")
    sections: Mapped[List["LapSection"]] = relationship("LapSection", back_populates="lap")
    telemetry_points: Mapped[List["TelemetryPoint"]] = relationship(
        "TelemetryPoint", back_populates="lap", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("session_id", "lap_number", name="uq_session_lap"),)


class LapSection(Base):
    __tablename__ = "lap_sections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lap_id: Mapped[int] = mapped_column(ForeignKey("laps.id", ondelete="CASCADE"))
    section_id: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(128))
    section_type: Mapped[str] = mapped_column(String(32))
    delta_time_ms: Mapped[float] = mapped_column(Float)
    avg_speed: Mapped[float] = mapped_column(Float)
    throttle_avg: Mapped[float] = mapped_column(Float)
    brake_avg: Mapped[float] = mapped_column(Float)
    steering_avg: Mapped[float] = mapped_column(Float)

    lap: Mapped["Lap"] = relationship("Lap", back_populates="sections")


class TelemetryPoint(Base):
    __tablename__ = "telemetry_points"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    lap_id: Mapped[int] = mapped_column(ForeignKey("laps.id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[float] = mapped_column(Float, index=True)
    distance_m: Mapped[Optional[float]] = mapped_column(Float)
    speed: Mapped[float] = mapped_column(Float)
    throttle: Mapped[float] = mapped_column(Float)
    brake: Mapped[float] = mapped_column(Float)
    gear: Mapped[int] = mapped_column(Integer)
    steering: Mapped[float] = mapped_column(Float)
    accel_lat: Mapped[Optional[float]] = mapped_column(Float)
    accel_long: Mapped[Optional[float]] = mapped_column(Float)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)

    lap: Mapped["Lap"] = relationship("Lap", back_populates="telemetry_points")
