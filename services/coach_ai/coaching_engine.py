from __future__ import annotations

import asyncio
import hashlib
import os
from typing import Dict, Optional

import google.generativeai as genai

from shared.schemas.feedback import FeedbackSeverity
from shared.utils.logging import configure_logging

from .schemas import CoachingRequest, CoachingResponse, SectionIssue
from .tts import TTSService

logger = configure_logging("coach_ai.engine")


class CoachingEngine:
    def __init__(self, tts_service: TTSService) -> None:
        self.tts_service = tts_service
        api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = "gemini-2.5-pro"
        self.gemini_model: Optional[genai.GenerativeModel] = None
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini 2.5 Pro abilitato per coach AI")
        else:
            logger.warning("GOOGLE_API_KEY non impostata: uso fallback rule-based")
        self.cache: Dict[str, CoachingResponse] = {}

    async def generate_feedback(self, request: CoachingRequest) -> CoachingResponse:
        cache_key = self._make_cache_key(request)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        text = None
        severity = FeedbackSeverity.SUGGESTION
        bullet_points: list[str] = []

        if self.gemini_model:
            try:
                text, bullet_points, severity = await self._ask_gemini(request)
            except Exception as exc:  # pragma: no cover
                logger.exception("Gemini failure: %s", exc)

        if text is None:
            text, bullet_points, severity = self._fallback_response(request)

        audio_url = None
        if os.getenv("ENABLE_TTS", "0") == "1":
            audio_url = await self.tts_service.synthesize(text, request.language)

        response = CoachingResponse(
            message=text,
            severity=severity,
            bullet_points=bullet_points,
            follow_up="Prossimo stint: applica un cambiamento alla volta e annota l'impatto.",
            audio_url=audio_url,
        )
        self.cache[cache_key] = response
        return response

    async def _ask_gemini(
        self, request: CoachingRequest
    ) -> tuple[str, list[str], FeedbackSeverity]:
        assert self.gemini_model is not None
        system_prompt = (
            "Sei un coach di guida GT3 per Assetto Corsa Competizione. "
            "Offri consigli pratici, puntuali e concreti. "
            "Rispondi in lingua italiana. "
            "Fornisci breve riepilogo e tre punti azionabili."
        )
        issues_text = "\n".join(
            [
                f"- {issue.section}: +{issue.delta_time_ms:.0f} ms, causa {issue.cause}, suggerisci: {issue.suggestion}"
                for issue in request.issues
            ]
        ) or "Nessun dato specifico, usa metriche generiche."

        user_prompt = (
            f"Sessione {request.session_id}, lap {request.lap}. "
            f"Livello pilota: {request.driver_level}. "
            f"Riassunto telemetria: {request.summary or 'non disponibile'}.\n"
            f"Issues:\n{issues_text}\n"
            f"Metriche: {request.metrics}.\n"
            "Genera risposta con schema: paragrafo breve + elenco puntato + call to action."
        )

        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = await asyncio.to_thread(
            self.gemini_model.generate_content,
            prompt,
            generation_config={"temperature": 0.4, "max_output_tokens": 512},
        )
        message = getattr(response, "text", "") or ""
        bullet_points = self._extract_bullets(message)
        severity = self._estimate_severity(request.issues, request.metrics)
        return message, bullet_points, severity

    def _fallback_response(
        self, request: CoachingRequest
    ) -> tuple[str, list[str], FeedbackSeverity]:
        if request.issues:
            worst = max(request.issues, key=lambda issue: issue.delta_time_ms)
            text = (
                f"Nella {worst.section} perdi circa {worst.delta_time_ms:.0f} ms. "
                f"{worst.suggestion} "
                "Concentrati su frenata progressiva e rilascio dolce del volante."
            )
            bullets = [
                worst.suggestion,
                "Controlla il delta in tempo reale per verificare il miglioramento.",
                "Rivedi il replay per confrontare la linea con il riferimento.",
            ]
            severity = (
                FeedbackSeverity.WARNING
                if worst.delta_time_ms > 150
                else FeedbackSeverity.SUGGESTION
            )
        else:
            text = (
                "Giro solido. Mantieni un punto di frenata consistente e lavora sulla modulazione del gas in uscita."
            )
            bullets = [
                "Stabilisci un riferimento visivo fisso per ogni staccata.",
                "Gestisci il trasferimento di carico mantenendo pressione freno costante.",
                "Applica il gas quando lo sterzo torna verso il centro.",
            ]
            severity = FeedbackSeverity.INFO
        return text, bullets, severity

    def _extract_bullets(self, message: str) -> list[str]:
        bullets: list[str] = []
        for line in message.splitlines():
            line = line.strip("-*• ").strip()
            if not line:
                continue
            if len(line.split()) < 3:
                continue
            bullets.append(line)
        return bullets[:4]

    def _estimate_severity(
        self, issues: list[SectionIssue], metrics: dict[str, float]
    ) -> FeedbackSeverity:
        if not issues:
            return FeedbackSeverity.INFO
        worst = max(issue.delta_time_ms for issue in issues)
        if worst > 300 or metrics.get("consistency_score", 1.0) < 0.5:
            return FeedbackSeverity.CRITICAL
        if worst > 150:
            return FeedbackSeverity.WARNING
        return FeedbackSeverity.SUGGESTION

    def _make_cache_key(self, request: CoachingRequest) -> str:
        raw = (
            f"{request.session_id}:{request.lap}:{request.language}:"
            f"{request.tone}:{request.driver_level}:{request.metrics}"
        )
        m = hashlib.sha1()
        m.update(raw.encode("utf-8"))
        for issue in request.issues:
            m.update(f"{issue.section}:{issue.delta_time_ms}".encode("utf-8"))
        return m.hexdigest()
