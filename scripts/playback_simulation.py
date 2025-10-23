
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from shared.schemas.telemetry import TelemetryFrame
from shared.utils.logging import configure_logging
from shared.utils.message_bus import TelemetryPublisher

logger = configure_logging("telemetry.playback")


async def playback(file_path: Path, playback_rate: float, loop: bool) -> None:
    publisher = TelemetryPublisher()
    await publisher.connect()

    async def _stream_once() -> None:
        with file_path.open("r", encoding="utf-8") as f:
            last_ts = None
            for line in f:
                data = json.loads(line.strip())
                frame = TelemetryFrame(**data)
                if last_ts is not None:
                    delta = (frame.timestamp - last_ts) / playback_rate
                    await asyncio.sleep(max(delta, 0))
                last_ts = frame.timestamp
                await publisher.publish(frame.dict())
        logger.info("Playback completato")

    while True:
        await _stream_once()
        if not loop:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay file telemetria JSONL")
    parser.add_argument(
        "--file",
        default="data/simulations/sample_lap.jsonl",
        help="Percorso file JSONL",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Playback rate (1.0 = realtime)")
    parser.add_argument("--loop", action="store_true", help="Ripeti in loop")
    args = parser.parse_args()

    path = Path(args.file).resolve()
    if not path.exists():
        raise SystemExit(f"File non trovato: {path}")

    asyncio.run(playback(path, args.rate, args.loop))


if __name__ == "__main__":
    main()
