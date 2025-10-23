from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class FeedbackSeverity(str, Enum):
    INFO = "info"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    CRITICAL = "critical"


class FeedbackChannel(str, Enum):
    OVERLAY = "overlay"
    AUDIO = "audio"
    HAPTIC = "haptic"
    LOG = "log"


class FeedbackEvent(BaseModel):
    session_id: str
    lap: int
    section: Optional[str] = None
    severity: FeedbackSeverity = FeedbackSeverity.SUGGESTION
    channel: FeedbackChannel = FeedbackChannel.OVERLAY
    message: str
    hint: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: float
    acknowledged: bool = False
