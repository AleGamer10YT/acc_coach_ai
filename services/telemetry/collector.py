from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np
from pydantic import BaseModel, Field

from shared.schemas.telemetry import SessionPhase, TelemetryFrame
from shared.utils.logging import configure_logging
from shared.utils.message_bus import TelemetryPublisher

logger = configure_logging("telemetry.collector")


class CollectorMode(str, Enum):
    UDP = "udp"
    SHARED_MEMORY = "shared_memory"
    SIMULATION = "simulation"


class CollectorConfig(BaseModel):
    mode: CollectorMode = CollectorMode.SIMULATION
    udp_host: str = Field("127.0.0.1", description="Indirizzo IP ACC UDP broadcast")
    udp_port: int = Field(9000, description="Porta ACC UDP broadcast")
    shared_memory_key: Optional[str] = Field(None, description="Chiave per shared memory ACC")
    simulation_file: Optional[str] = Field(
        None, description="Percorso file JSONL con frame simulati"
    )
    loop_simulation: bool = Field(True, description="Ripeti file simulazione all'infinito")
    playback_rate: float = Field(1.0, description="Velocita playback (1.0 = realtime)")


class TelemetryCollector:
    def __init__(self) -> None:
        self.publisher = TelemetryPublisher()
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._config: Optional[CollectorConfig] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def current_config(self) -> Optional[CollectorConfig]:
        return self._config

    async def start(self, config: CollectorConfig) -> None:
        if self.is_running:
            raise RuntimeError("Collector gia attivo")

        await self.publisher.connect()
        self._config = config
        self._stop_event.clear()

        if config.mode == CollectorMode.SIMULATION:
            if not config.simulation_file:
                raise ValueError("simulation_file richiesto in modalita simulazione")
            self._task = asyncio.create_task(self._run_simulation(config))
        elif config.mode == CollectorMode.UDP:
            self._task = asyncio.create_task(self._run_udp(config))
        elif config.mode == CollectorMode.SHARED_MEMORY:
            self._task = asyncio.create_task(self._run_shared_memory(config))
        else:
            raise ValueError(f"Modalita non supportata: {config.mode}")

        logger.info("Collector avviato in modalita %s", config.mode.value)

    async def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._task:
            await self._task
        self._task = None
        logger.info("Collector fermato")

    async def _run_simulation(self, config: CollectorConfig) -> None:
        file_path = Path(config.simulation_file).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        logger.info("Playback simulazione da %s", file_path)
        while not self._stop_event.is_set():
            async for frame in self._read_simulation_file(file_path, config.playback_rate):
                if self._stop_event.is_set():
                    break
                await self.publisher.publish(frame.dict())
            if not config.loop_simulation:
                break
        logger.info("Simulazione terminata")

    async def _read_simulation_file(
        self, file_path: Path, playback_rate: float
    ) -> AsyncIterator[TelemetryFrame]:
        last_ts: Optional[float] = None
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if self._stop_event.is_set():
                    break
                data = json.loads(line.strip())
                frame = TelemetryFrame(**data)
                if last_ts is not None:
                    delta = (frame.timestamp - last_ts) / playback_rate
                    await asyncio.sleep(max(delta, 0))
                last_ts = frame.timestamp
                yield frame

    async def _run_udp(self, config: CollectorConfig) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.udp_host, config.udp_port))
        sock.setblocking(False)
        logger.info("In ascolto UDP su %s:%s", config.udp_host, config.udp_port)

        loop = asyncio.get_running_loop()
        try:
            while not self._stop_event.is_set():
                try:
                    data, _ = await loop.sock_recvfrom(sock, 4096)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    break
                if not data:
                    continue
                frame = self._parse_udp_packet(data)
                await self.publisher.publish(frame.dict())
        finally:
            sock.close()
            logger.info("UDP listener chiuso")

    def _parse_udp_packet(self, data: bytes) -> TelemetryFrame:
        # Placeholder parsing: ACC invia telemetria binaria; qui decodifichiamo dati minimi simulati
        header = struct.unpack_from("<dI", data, 0)
        timestamp = header[0]
        lap = header[1]
        floats = struct.unpack_from("<10f", data, struct.calcsize("<dI"))
        speed_kph = floats[0]
        throttle = floats[1]
        brake = floats[2]
        steering = floats[3]
        gear = int(floats[4])
        world_pos = floats[5:8]
        track_pos = floats[8]

        frame = TelemetryFrame(
            session_id="unknown",
            driver_id="unknown",
            timestamp=timestamp,
            lap=lap,
            lap_time_ms=None,
            speed_kph=speed_kph,
            throttle=throttle,
            brake=brake,
            steering=steering,
            gear=gear,
            world_pos_x=world_pos[0],
            world_pos_y=world_pos[1],
            world_pos_z=world_pos[2],
            track_spline_pos=min(max(track_pos, 0.0), 1.0),
            player_car="unknown",
            track_name="unknown",
            session_phase=SessionPhase.OTHER,
        )
        return frame

    async def _run_shared_memory(self, config: CollectorConfig) -> None:
        # ACC Shared Memory richiede integrazione Ctypes; qui forniamo placeholder con rumore
        logger.warning("Shared memory non implementata - uso dati sintetici temporanei")
        rng = np.random.default_rng()
        base_ts = time.time()
        lap = 0
        while not self._stop_event.is_set():
            timestamp = time.time()
            frame = TelemetryFrame(
                session_id="sm-unknown",
                driver_id="unknown",
                timestamp=timestamp,
                lap=lap,
                lap_time_ms=int((timestamp - base_ts) * 1000),
                speed_kph=150 + rng.normal(0, 5),
                throttle=float(np.clip(rng.normal(0.8, 0.1), 0, 1)),
                brake=float(np.clip(rng.normal(0.1, 0.05), 0, 1)),
                steering=float(np.clip(rng.normal(0.0, 0.2), -1, 1)),
                gear=int(np.clip(rng.integers(3, 6), 1, 6)),
                world_pos_x=float(rng.normal(0, 1)),
                world_pos_y=float(rng.normal(0, 1)),
                world_pos_z=float(rng.normal(0, 1)),
                track_spline_pos=float(np.clip((timestamp - base_ts) % 1.0, 0, 1)),
                player_car="unknown",
                track_name="unknown",
                session_phase=SessionPhase.OTHER,
            )
            await self.publisher.publish(frame.dict())
            await asyncio.sleep(1 / 60)
