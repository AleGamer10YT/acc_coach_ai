from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from gtts import gTTS

from shared.utils.logging import configure_logging

logger = configure_logging("coach_ai.tts")


class TTSService:
    def __init__(self) -> None:
        self.provider = os.getenv("TTS_PROVIDER", "gtts")
        self.output_dir = Path(os.getenv("TTS_OUTPUT_DIR", "data/audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str, language: str) -> Optional[str]:
        if not text:
            return None
        if self.provider == "gtts":
            filename = self.output_dir / f"tts_{abs(hash(text))}.mp3"
            await asyncio.to_thread(self._render_gtts, text, language, filename)
            logger.info("Audio generato %s", filename)
            return str(filename)
        # provider non supportato: fallback mock
        mock_file = self.output_dir / f"tts_{abs(hash(text))}.txt"
        mock_file.write_text(text, encoding="utf-8")
        logger.info("Audio mock salvato %s", mock_file)
        return str(mock_file)

    def _render_gtts(self, text: str, language: str, path: Path) -> None:
        tts = gTTS(text=text, lang=language)
        tts.save(str(path))
