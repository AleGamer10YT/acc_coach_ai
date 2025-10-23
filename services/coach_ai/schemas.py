from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from shared.schemas.feedback import FeedbackSeverity


class SectionIssue(BaseModel):
    section: str
    delta_time_ms: float
    cause: str
    suggestion: str


class CoachingRequest(BaseModel):
    session_id: str
    lap: int
    language: str = Field("it", description="Codice lingua preferita (es it, en)")
    tone: str = Field("coach", description="coach, friendly, analytical")
    driver_level: str = Field("intermediate", description="beginner, intermediate, advanced")
    issues: List[SectionIssue] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    summary: Optional[str] = None


class CoachingResponse(BaseModel):
    message: str
    severity: FeedbackSeverity = FeedbackSeverity.SUGGESTION
    bullet_points: List[str] = Field(default_factory=list)
    follow_up: Optional[str] = None
    audio_url: Optional[str] = None


class BatchReportRequest(BaseModel):
    session_id: str
    laps: List[int]
    key_findings: List[SectionIssue] = Field(default_factory=list)
