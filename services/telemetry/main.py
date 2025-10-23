from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.telemetry.collector import CollectorConfig, CollectorMode, TelemetryCollector
from shared.utils.config import get_settings
from shared.utils.logging import configure_logging

settings = get_settings()
logger = configure_logging("telemetry.api", settings.log_level)

app = FastAPI(
    title="ACC Coach Telemetry Collector",
    version="0.1.0",
    docs_url="/",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

collector = TelemetryCollector()
collector_lock = asyncio.Lock()


class StartRequest(BaseModel):
    mode: CollectorMode = CollectorMode.SIMULATION
    udp_host: str | None = None
    udp_port: int | None = None
    simulation_file: str | None = None
    loop_simulation: bool = True
    playback_rate: float = 1.0


class CollectorStatus(BaseModel):
    running: bool
    mode: CollectorMode | None


@app.on_event("startup")
async def bootstrap() -> None:
    logger.info("Telemetry collector API pronto")


@app.post("/collector/start", response_model=CollectorStatus)
async def start_collector(payload: StartRequest) -> CollectorStatus:
    cfg = CollectorConfig(
        mode=payload.mode,
        udp_host=payload.udp_host or "127.0.0.1",
        udp_port=payload.udp_port or 9000,
        simulation_file=payload.simulation_file,
        loop_simulation=payload.loop_simulation,
        playback_rate=payload.playback_rate,
    )

    async with collector_lock:
        if collector.is_running:
            raise HTTPException(status_code=409, detail="Collector gia avviato")
        await collector.start(cfg)

    return CollectorStatus(running=True, mode=cfg.mode)


@app.post("/collector/stop", response_model=CollectorStatus)
async def stop_collector() -> CollectorStatus:
    async with collector_lock:
        await collector.stop()
    return CollectorStatus(running=False, mode=None)


@app.get("/collector/status", response_model=CollectorStatus)
async def get_status() -> CollectorStatus:
    cfg = collector.current_config
    mode = cfg.mode if cfg else None
    return CollectorStatus(running=collector.is_running, mode=mode)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.telemetry.main:app", host="0.0.0.0", port=8081, reload=False)
