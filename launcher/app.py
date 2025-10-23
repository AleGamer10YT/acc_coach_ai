from __future__ import annotations
from datetime import datetime

import asyncio
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

import importlib.util
import math


def ensure_runtime_dependencies() -> None:
    if getattr(sys, "frozen", False):
        # in executable, dependencies are bundled
        return

    required = {
        "PySide6": "PySide6==6.7.0",
        "fastapi": "fastapi==0.110.0",
        "uvicorn": "uvicorn[standard]==0.29.0",
        "pydantic": "pydantic==2.6.4",
        "pydantic_settings": "pydantic-settings==2.3.1",
        "sqlalchemy": "sqlalchemy==2.0.29",
        "aiosqlite": "aiosqlite==0.19.0",
        "alembic": "alembic==1.13.1",
        "pandas": "pandas==2.2.1",
        "redis": "redis==5.0.3",
        "dateutil": "python-dateutil==2.9.0",
        "aiofiles": "aiofiles==23.2.1",
        "google.generativeai": "google-generativeai==0.7.2",
    }
    missing: list[str] = []
    for module, requirement in required.items():
        module_name = module.replace("-", "_")
        if importlib.util.find_spec(module_name) is None:
            missing.append(requirement)

    if not missing:
        return

    print("[ACC Coach] Installing missing dependencies:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_runtime_dependencies()


from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal, QEasingCurve, QPropertyAnimation, QTimer  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtGui import QCloseEvent, QPixmap, QIcon, QIntValidator  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtWidgets import (  # noqa: E402  pylint: disable=wrong-import-position
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QFrame,
    QSizePolicy,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QListWidget,
    QAbstractItemView,
    QListWidgetItem,
    QAbstractButton,
    QRadioButton,
    QButtonGroup,
    QTabWidget,
)

from services.analytics.engine import RealtimeAnalyticsEngine  # noqa: E402  pylint: disable=wrong-import-position
from services.telemetry.collector import CollectorConfig, CollectorMode, TelemetryCollector  # noqa: E402  pylint: disable=wrong-import-position

DEFAULT_VALUES: Dict[str, str] = {
    "APP_ENV": "development",
    "LOG_LEVEL": "INFO",
    "REDIS_URL": "redis://localhost:6379/0",
    "DATABASE_URL": "sqlite+aiosqlite:///./data/app.db",
    "COACH_API_URL": "http://localhost:8082",
    "OVERLAY_WS_URL": "ws://localhost:8080/ws/feedback",
    "DASHBOARD_URL": "http://localhost:8501",
    "GOOGLE_API_KEY": "",
    "ENABLE_TTS": "0",
    "TTS_PROVIDER": "elevenlabs",
    "ELEVENLABS_API_KEY": "",
    "TTS_OUTPUT_DIR": "data/audio",
    "ACC_USE_LIVE": "1",
    "ACC_UDP_HOST": "127.0.0.1",
    "ACC_UDP_PORT": "9000",
    "REPO_ZIP_URL": "https://github.com/<user>/acc_coach_ai/archive/refs/heads/main.zip",
    "INSTALL_DIR": str((Path.home() / "ACC_Coach_AI").resolve()),
}

ENV_FIELDS = [
    ("GOOGLE_API_KEY", "Google AI Studio API Key"),
    ("ELEVENLABS_API_KEY", "ElevenLabs API Key"),
]

PRIMARY_BG = "#0E0E10"
SECONDARY_BG = "#1E1E24"
CARD_BG = "#1E1E24"
SURFACE_MEDIUM = "#1E1E24"
SURFACE_BORDER = "#2A2C33"
ACCENT_RED = "#9C2A23"
ACCENT_RED_HOVER = "#D23B3B"
ACCENT_GLOW = "#D36255"
TEXT_PRIMARY = "#E9EAEC"
TEXT_MUTED = "#9EA3A8"
SUCCESS_GREEN = "#21BA75"
WARNING_AMBER = "#F0B429"

APP_ICON_FILENAME = "acc_icon_png.png"
FALLBACK_ICON_FILENAME = "acc_icon.ico"


def resource_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def app_icon_path() -> Path:
    return resource_path() / "resources" / APP_ICON_FILENAME


def load_app_icon() -> QIcon:
    path = app_icon_path()
    icon = QIcon(str(path))
    if icon.isNull():
        pix = QPixmap(str(path))
        if not pix.isNull():
            icon = QIcon(pix)
    if icon.isNull():
        fallback = resource_path() / "resources" / FALLBACK_ICON_FILENAME
        icon = QIcon(str(fallback))
    return icon


def load_app_pixmap(size: int) -> QPixmap:
    path = app_icon_path()
    pix = QPixmap(str(path))
    if not pix.isNull() and size > 0:
        return pix.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    icon = load_app_icon()
    pm = icon.pixmap(size, size)
    if not pm.isNull():
        return pm
    fallback = QPixmap(str(resource_path() / "resources" / FALLBACK_ICON_FILENAME))
    if not fallback.isNull() and size > 0:
        return fallback.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return fallback


def load_lap_analysis_data(env_manager: EnvManager) -> dict[str, dict[str, dict]]:
    db_path = resolve_database_path(env_manager)
    if not db_path:
        return {}
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        data: dict[str, dict[str, dict]] = {}
        sessions = conn.execute(
            "SELECT id, track_name, car_model, fastest_lap_id FROM sessions ORDER BY started_at DESC"
        ).fetchall()
        for session in sessions:
            track = session["track_name"] or "Sconosciuto"
            car = session["car_model"] or "Sconosciuto"
            data.setdefault(track, {})
            if car in data[track]:
                continue
            analysis: dict[str, object] = {"ideal": None, "laps": []}
            sections_cache: dict[int, list[dict[str, Optional[float]]]] = {}
            lap_rows = conn.execute(
                "SELECT id, lap_number, lap_time_ms FROM laps WHERE session_id = ? ORDER BY lap_number DESC LIMIT 10",
                (session["id"],),
            ).fetchall()
            laps_list: list[dict[str, object]] = []
            for lap_row in lap_rows:
                section_rows = conn.execute(
                    "SELECT section_id, name, delta_time_ms FROM lap_sections WHERE lap_id = ? ORDER BY id ASC",
                    (lap_row["id"],),
                ).fetchall()
                sections_list = [
                    {
                        "section_id": sec["section_id"],
                        "name": sec["name"],
                        "delta_time_ms": sec["delta_time_ms"],
                    }
                    for sec in section_rows
                ]
                sections_cache[lap_row["id"]] = sections_list
                laps_list.append(
                    {
                        "lap_id": lap_row["id"],
                        "lap_number": lap_row["lap_number"],
                        "lap_time_ms": lap_row["lap_time_ms"],
                        "sections": sections_list,
                    }
                )
            fastest_id = session["fastest_lap_id"]
            fastest_entry = None
            if fastest_id:
                fastest_entry = conn.execute(
                    "SELECT id, lap_number, lap_time_ms FROM laps WHERE id = ?", (fastest_id,)
                ).fetchone()
            ideal_payload = None
            if fastest_entry:
                best_sections = sections_cache.get(fastest_entry["id"])
                if best_sections is None:
                    section_rows = conn.execute(
                        "SELECT section_id, name, delta_time_ms FROM lap_sections WHERE lap_id = ? ORDER BY id ASC",
                        (fastest_entry["id"],),
                    ).fetchall()
                    best_sections = [
                        {
                            "section_id": sec["section_id"],
                            "name": sec["name"],
                            "delta_time_ms": sec["delta_time_ms"],
                        }
                        for sec in section_rows
                    ]
                ideal_payload = {
                    "lap_id": fastest_entry["id"],
                    "lap_number": fastest_entry["lap_number"],
                    "lap_time_ms": fastest_entry["lap_time_ms"],
                    "sections": best_sections or [],
                }
            if ideal_payload and not any(l["lap_id"] == ideal_payload["lap_id"] for l in laps_list):
                laps_list.append(ideal_payload)
            analysis["ideal"] = ideal_payload
            analysis["laps"] = laps_list
            data[track][car] = analysis
        conn.close()
        return data
    except Exception:
        return {}


class EnvManager:
    def __init__(self) -> None:
        self.values: Dict[str, str] = DEFAULT_VALUES.copy()
        self.load()

    def load(self) -> None:
        self._load_env_file(resource_path() / ".env")
        install_dir = self.get_install_dir()
        self._load_env_file(install_dir / ".env")
        infra_env = install_dir / "infrastructure" / ".env"
        self._load_env_file(infra_env)
        # Backward compatibility: migrate legacy OpenAI keys to Google AI Studio
        legacy_key = self.values.pop("OPENAI_API_KEY", "").strip()
        if legacy_key and not self.values.get("GOOGLE_API_KEY"):
            self.values["GOOGLE_API_KEY"] = legacy_key
        # Remove deprecated OpenAI model entry if present
        self.values.pop("OPENAI_MODEL", None)

    def _load_env_file(self, env_file: Path) -> None:
        if not env_file.exists():
            return
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        for line in lines:
            if not line or "=" not in line or line.strip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            self.values[key.strip()] = value.strip()

    def save(self) -> None:
        env_file = resource_path() / ".env"
        self._write_env(env_file, self.values)
        install_dir = self.get_install_dir()
        self._write_env(install_dir / ".env", self.values)
        infra_dir = install_dir / "infrastructure"
        if infra_dir.exists():
            self._write_env(infra_dir / ".env", self.values)

    def update(self, updates: Dict[str, str]) -> None:
        self.values.update(updates)
        self.save()

    def get_install_dir(self) -> Path:
        return Path(self.values.get("INSTALL_DIR", DEFAULT_VALUES["INSTALL_DIR"])).expanduser().resolve()

    def has_required_api_keys(self) -> bool:
        required = ["GOOGLE_API_KEY"]
        return all(self.values.get(key) for key in required)

    @staticmethod
    def _write_env(env_path: Path, values: Dict[str, str]) -> None:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{key}={values.get(key, DEFAULT_VALUES.get(key, ''))}" for key in sorted(values)]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class DownloadWorker(QThread):
    finished = Signal(bool, str)

    def __init__(self, url: str, target_dir: Path) -> None:
        super().__init__()
        self.url = url
        self.target_dir = target_dir

    def run(self) -> None:  # pragma: no cover
        try:
            download_and_extract(self.url, self.target_dir)
            self.finished.emit(True, f"Pacchetto scaricato in {self.target_dir}")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class ServiceController(QObject):
    feedback_received = Signal(dict)
    status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.collector: Optional[TelemetryCollector] = None
        self.analytics: Optional[RealtimeAnalyticsEngine] = None
        self.running = False

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self, config: CollectorConfig) -> None:
        if self.running:
            self.status_changed.emit("Sessione gia attiva")
            return
        if config.mode == CollectorMode.SIMULATION:
            sim_path = Path(config.simulation_file or "")
            if not sim_path.exists():
                self.status_changed.emit(f"File simulazione non trovato: {sim_path}")
                return

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        future = asyncio.run_coroutine_threadsafe(self._async_start(config), self.loop)
        future.add_done_callback(self._handle_future)

    def stop(self) -> None:
        if not self.running or not self.loop:
            self.status_changed.emit("Sessione non attiva")
            return

        async def _stop() -> None:
            if self.collector:
                await self.collector.stop()
            if self.analytics:
                await self.analytics.stop()

        asyncio.run_coroutine_threadsafe(_stop(), self.loop).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)
        self.collector = None
        self.analytics = None
        self.loop = None
        self.thread = None
        self.running = False
        self.status_changed.emit("Sessione terminata")

    def _run_loop(self) -> None:
        if not self.loop:
            return
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _handle_future(self, future: asyncio.Future) -> None:  # pragma: no cover
        try:
            future.result()
        except Exception as exc:
            self.status_changed.emit(f"Errore avvio servizi: {exc}")

    async def _async_start(self, config: CollectorConfig) -> None:
        self.analytics = RealtimeAnalyticsEngine(on_feedback=self._on_feedback)
        await self.analytics.start()
        self.collector = TelemetryCollector()
        await self.collector.start(config)
        self.running = True
        if config.mode == CollectorMode.UDP:
            self.status_changed.emit("Sessione live ACC avviata")
        elif config.mode == CollectorMode.SIMULATION:
            self.status_changed.emit("Sessione simulata avviata")
        else:
            self.status_changed.emit(f"Sessione avviata ({config.mode.value})")

    def _on_feedback(self, event: dict) -> None:
        self.feedback_received.emit(event)


def download_and_extract(zip_url: str, target_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        archive = tmp_path / "package.zip"
        urllib.request.urlretrieve(zip_url, archive)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(tmp_path)
        candidates = [p for p in tmp_path.iterdir() if p.is_dir()]
        extracted_root = next(
            (c for c in candidates if (c / "infrastructure").exists()),
            candidates[0] if candidates else tmp_path,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in extracted_root.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)


def get_last_session_summary(env_manager: EnvManager) -> Optional[Dict[str, str]]:
    db_path = resolve_database_path(env_manager)
    if not db_path:
        return None
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        with conn:
            session = conn.execute(
                "SELECT id, track_name, car_model, started_at, finished_at, "
                "consistency_score, efficiency_score FROM sessions "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if not session:
                return None
            laps = conn.execute(
                "SELECT COUNT(*) AS total_laps, MIN(lap_time_ms) AS best_lap "
                "FROM laps WHERE session_id = ?",
                (session["id"],),
            ).fetchone()
        conn.close()
        return {
            "track_name": session["track_name"],
            "car_model": session["car_model"],
            "started_at": session["started_at"] or "",
            "consistency": session["consistency_score"] or 0.0,
            "efficiency": session["efficiency_score"] or 0.0,
            "laps": laps["total_laps"] if laps else 0,
            "best_lap": laps["best_lap"] if laps and laps["best_lap"] else None,
        }
    except Exception:
        return None


def format_lap_time(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:05.2f}"


def resolve_database_path(env_manager: EnvManager) -> Optional[Path]:
    db_url = env_manager.values.get("DATABASE_URL", "")
    if "sqlite" not in db_url:
        return None
    path_part = ""
    if ":///" in db_url:
        path_part = db_url.split(":///", 1)[1]
    elif "://" in db_url:
        path_part = db_url.split("://", 1)[1]
    if not path_part:
        return None
    db_path = Path(path_part)
    if not db_path.is_absolute():
        db_path = env_manager.get_install_dir() / db_path
    db_path = db_path.resolve()
    if not db_path.exists():
        return None
    return db_path


class SidebarButton(QPushButton):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setCheckable(True)
        self.setMinimumHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("sidebarButton")


class Sidebar(QWidget):
    selection_changed = Signal(str)

    def __init__(self, items: Dict[str, str]) -> None:
        super().__init__()
        self.buttons: Dict[str, SidebarButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 36, 24, 24)
        layout.setSpacing(18)

        icon_label = QLabel()
        pix = load_app_pixmap(60)
        if not pix.isNull():
            icon_label.setPixmap(pix)
        else:
            icon_label.setText("ACC Coach AI")
            icon_label.setStyleSheet(f"color: {ACCENT_RED}; font-size: 24px; font-weight: 700; letter-spacing: 0.08em;")
        layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignLeft)

        subtitle = QLabel("Virtual racing coach")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        for key, label in items.items():
            btn = SidebarButton(label)
            btn.setAccessibleName(f"Vai a {label}")
            btn.clicked.connect(lambda _checked, k=key: self._on_select(k))
            self.buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)
        footer = QLabel("v0.1.0")
        footer.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(footer)

    def _on_select(self, key: str) -> None:
        self.select(key)
        self.selection_changed.emit(key)

    def select(self, key: str) -> None:
        for name, button in self.buttons.items():
            button.setChecked(name == key)
            button.setAccessibleDescription("Pagina attuale" if name == key else "")


def build_card(
    title: str,
    value: str,
    subtitle: str = "",
    unit: str = "",
    trend: Optional[str] = None,
    status_color: Optional[str] = None,
) -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    frame.setFrameShape(QFrame.Shape.NoFrame)
    if status_color:
        frame.setProperty("statusColor", status_color)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(24, 22, 24, 22)
    layout.setSpacing(14)
    title_label = QLabel(title)
    title_label.setStyleSheet(
        f"color: {TEXT_MUTED}; font-size: 13px; text-transform: uppercase; letter-spacing: 0.12em;"
    )
    header_row = QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignLeft)
    header_row.addStretch(1)
    status_dot = QLabel("●")
    status_dot.setObjectName("statusDot")
    if status_color:
        status_dot.setStyleSheet(f"color: {status_color}; font-size: 16px;")
    else:
        status_dot.hide()
    header_row.addWidget(status_dot, alignment=Qt.AlignmentFlag.AlignRight)
    layout.addLayout(header_row)
    value_row = QHBoxLayout()
    value_row.setContentsMargins(0, 0, 0, 0)
    value_row.setSpacing(8)
    value_label = QLabel(value)
    value_label.setObjectName("cardValue")
    value_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 36px; font-weight: 700;")
    value_row.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignVCenter)
    if unit:
        unit_label = QLabel(unit)
        unit_label.setObjectName("cardUnit")
        unit_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px; font-weight: 500;")
        value_row.addWidget(unit_label, alignment=Qt.AlignmentFlag.AlignBottom)
    value_row.addStretch(1)
    if trend:
        trend_label = QLabel(trend)
        trend_label.setObjectName("cardTrend")
        trend_label.setStyleSheet(f"color: {SUCCESS_GREEN if trend.startswith('▲') else ACCENT_GLOW}; font-size: 13px;")
        value_row.addWidget(trend_label, alignment=Qt.AlignmentFlag.AlignBottom)
    layout.addLayout(value_row)
    subtitle_label = QLabel(subtitle)
    subtitle_label.setStyleSheet(f"font-size: 14px; color: {TEXT_MUTED}; font-weight: 500;")
    layout.addWidget(subtitle_label)
    layout.addStretch(1)
    return frame


class HomePage(QWidget):
    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(28)

        self.banner = QFrame()
        self.banner.setObjectName("banner")
        self.banner.setFrameShape(QFrame.Shape.NoFrame)
        banner_layout = QVBoxLayout(self.banner)
        banner_layout.setContentsMargins(24, 24, 24, 24)
        logo_row = QHBoxLayout()
        logo_row.setContentsMargins(0, 0, 0, 0)
        logo_row.setSpacing(16)
        logo_label = QLabel()
        logo_label.setObjectName("dashboardLogo")
        logo_pixmap = load_app_pixmap(64)
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap)
        else:
            logo_label.setText("ACC Coach AI")
            logo_label.setStyleSheet("font-size: 22px; font-weight: 700; letter-spacing: 0.05em;")
        logo_row.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignLeft)
        self.banner_title = QLabel("Dashboard")
        self.banner_title.setObjectName("bannerTitle")
        self.banner_title.setStyleSheet("font-size: 30px; font-weight: 700; letter-spacing: 0.02em;")
        logo_row.addWidget(self.banner_title, alignment=Qt.AlignmentFlag.AlignLeft)
        logo_row.addStretch(1)
        banner_layout.addLayout(logo_row)
        self.banner_subtitle = QLabel("Benvenuto nel tuo coach virtuale")
        self.banner_subtitle.setStyleSheet(f"font-size: 16px; color: {TEXT_MUTED}; font-weight: 500;")
        banner_layout.addWidget(self.banner_subtitle)
        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet(f"font-size: 13px; color: {ACCENT_GLOW}; margin-top: 8px;")
        banner_layout.addWidget(self.alert_label)
        layout.addWidget(self.banner)

        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(18)

        self.card_session = build_card(
            "Ultima sessione", "-", "Nessuna sessione registrata", unit="", status_color=WARNING_AMBER
        )
        self.card_laps = build_card("Giri completati", "-", "Ultima sessione", unit="giri", status_color=SUCCESS_GREEN)
        self.card_best_lap = build_card("Miglior giro", "-", "Tempo", unit="ms", status_color=ACCENT_RED_HOVER)

        cards_layout.addWidget(self.card_session, 0, 0, 1, 2)
        cards_layout.addWidget(self.card_laps, 0, 2)
        cards_layout.addWidget(self.card_best_lap, 0, 3)
        for col in range(4):
            cards_layout.setColumnStretch(col, 1)
        cards_layout.setRowMinimumHeight(0, 140)
        layout.addWidget(cards_widget)

        sections_row = QHBoxLayout()
        sections_row.setContentsMargins(0, 0, 0, 0)
        sections_row.setSpacing(24)
        self.session_preview = self._build_session_preview()
        self.analysis_preview = self._build_analysis_preview()
        sections_row.addWidget(self.session_preview, stretch=3)
        sections_row.addWidget(self.analysis_preview, stretch=2)
        layout.addLayout(sections_row)

        bars_row = QHBoxLayout()
        bars_row.setContentsMargins(0, 0, 0, 0)
        bars_row.setSpacing(24)
        self.consistency_card, self.consistency_bar = self._build_progress_card("Consistenza", ACCENT_RED)
        self.efficiency_card, self.efficiency_bar = self._build_progress_card("Efficienza", ACCENT_RED_HOVER)
        bars_row.addWidget(self.consistency_card, stretch=1)
        bars_row.addWidget(self.efficiency_card, stretch=1)
        layout.addLayout(bars_row)

    def update_metrics(self, metrics: Optional[Dict[str, str]], api_ready: bool) -> None:
        self.set_api_ready(api_ready)
        if not api_ready:
            self.alert_label.setText("API key Google AI Studio mancante: inseriscila nelle impostazioni per attivare il coach.")
        else:
            self.alert_label.setText("")
        if not metrics:
            self._set_card(self.card_session, "-", "Nessuna sessione registrata")
            self._set_card(self.card_laps, "-", "Ultima sessione")
            self._set_card(self.card_best_lap, "-", "Tempo")
            self.consistency_bar.setValue(0)
            self.efficiency_bar.setValue(0)
            self.session_state_preview.setText("Sessione inattiva")
            self.session_state_preview.setStyleSheet(self._pill_style("rgba(255,255,255,0.08)", TEXT_MUTED))
            return
        session_title = f"{metrics.get('track_name', 'Sconosciuto')} - {metrics.get('car_model', '')}"
        started = metrics.get("started_at", "")
        self._set_card(self.card_session, session_title, started or "Nessuna data")
        laps_value = str(metrics.get("laps", 0))
        self._set_card(self.card_laps, laps_value, "Ultima sessione")
        best_lap = format_lap_time(metrics.get("best_lap"))
        self._set_card(self.card_best_lap, best_lap, "Tempo")
        self.consistency_bar.setValue(int(float(metrics.get("consistency", 0.0)) * 100))
        self.efficiency_bar.setValue(int(float(metrics.get("efficiency", 0.0)) * 100))
        self.session_state_preview.setText("Ultima sessione aggiornata")
        self.session_state_preview.setStyleSheet(self._pill_style("rgba(33,186,117,0.18)", SUCCESS_GREEN))
        self.session_log_preview.setText("Monitorando i feedback più recenti...")

    def set_api_ready(self, ready: bool) -> None:
        if ready:
            self.banner_subtitle.hide()
            self.banner_title.setStyleSheet("font-size: 30px; font-weight: 700; letter-spacing: 0.02em;")
            self.session_state_preview.setText("Sessione inattiva")
            self.session_state_preview.setStyleSheet(self._pill_style("rgba(255,255,255,0.08)", TEXT_MUTED))
        else:
            self.banner_subtitle.show()
            self.banner_title.setStyleSheet("font-size: 30px; font-weight: 700; letter-spacing: 0.02em;")
            self.session_state_preview.setText("API mancanti")
            self.session_state_preview.setStyleSheet(self._pill_style("rgba(156,42,35,0.25)", ACCENT_GLOW))

    @staticmethod
    def _set_card(card: QFrame, value: str, subtitle: str, trend: Optional[str] = None) -> None:
        value_label = card.findChild(QLabel, "cardValue")
        if value_label:
            value_label.setText(value)
        trend_label = card.findChild(QLabel, "cardTrend")
        if trend_label:
            if trend:
                trend_label.setText(trend)
                trend_label.show()
            else:
                trend_label.hide()
        subtitle_labels = [
            child
            for child in card.findChildren(QLabel)
            if child not in (value_label, trend_label) and child.objectName() != "statusDot"
        ]
        if subtitle_labels:
            subtitle_labels[-1].setText(subtitle)
        dot_label = card.findChild(QLabel, "statusDot")
        if dot_label:
            color = card.property("statusColor")
            if color:
                dot_label.setStyleSheet(f"color: {color}; font-size: 16px;")
            else:
                dot_label.hide()

    def _build_session_preview(self) -> QFrame:
        section = QFrame()
        section.setObjectName("sectionCard")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)
        title = QLabel("Sessione in tempo reale")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)
        self.session_state_preview = QLabel("Sessione inattiva")
        self.session_state_preview.setStyleSheet(self._pill_style("rgba(255,255,255,0.08)", TEXT_MUTED))
        self.session_state_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.session_state_preview, alignment=Qt.AlignmentFlag.AlignLeft)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(12)
        buttons_row.addWidget(self._preview_button("Live da ACC", ACCENT_RED))
        buttons_row.addWidget(self._preview_button("Carica JSONL", SURFACE_BORDER))
        buttons_row.addWidget(self._preview_button("Avvia", SUCCESS_GREEN))
        buttons_row.addWidget(self._preview_button("Termina", SURFACE_BORDER, TEXT_PRIMARY))
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        log_frame = QFrame()
        log_frame.setObjectName("sectionInner")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(18, 16, 18, 16)
        log_label = QLabel("I feedback del coach appariranno qui…")
        log_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")
        self.session_log_preview = log_label
        log_layout.addWidget(log_label)
        layout.addWidget(log_frame, stretch=1)
        return section

    def _build_analysis_preview(self) -> QFrame:
        section = QFrame()
        section.setObjectName("sectionCard")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)
        title = QLabel("Lap Analysis — Delta per settore")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(title)

        chart_frame = QFrame()
        chart_frame.setObjectName("sectionInner")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(18, 18, 18, 18)
        chart_layout.setSpacing(12)
        bar_layout = QHBoxLayout()
        bar_layout.setSpacing(12)
        for height in (40, 68, 100, 55, 80, 45, 90):
            bar = QFrame()
            bar.setMinimumWidth(16)
            bar.setMaximumWidth(20)
            bar.setFixedHeight(height)
            bar.setStyleSheet(
                f"background-color: {ACCENT_RED}; border-radius: 6px; min-height: 40px;"
            )
            bar_layout.addWidget(bar, alignment=Qt.AlignmentFlag.AlignBottom)
        bar_layout.addStretch(1)
        chart_layout.addLayout(bar_layout)
        footer = QLabel("Trend stimato sui settori principali (mockup)")
        footer.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")
        chart_layout.addWidget(footer, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(chart_frame, stretch=1)
        return section

    def _build_progress_card(self, title: str, chunk_color: str) -> tuple[QFrame, QProgressBar]:
        wrapper = QFrame()
        wrapper.setObjectName("sectionCard")
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(24, 18, 24, 18)
        wrapper_layout.setSpacing(12)
        label = QLabel(title)
        label.setStyleSheet("font-size: 16px; font-weight: 600;")
        wrapper_layout.addWidget(label)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setFormat("%p%")
        bar.setTextVisible(False)
        bar.setStyleSheet(
            "QProgressBar {background-color: rgba(255,255,255,0.07); border: 0px; height: 16px; border-radius: 10px;}"
            f"QProgressBar::chunk {{background: {chunk_color}; border-radius: 10px;}}"
        )
        wrapper_layout.addWidget(bar)
        return wrapper, bar

    @staticmethod
    def _pill_style(bg: str, fg: str) -> str:
        return f"background-color: {bg}; color: {fg}; padding: 6px 12px; border-radius: 12px; font-weight: 600;"

    def _preview_button(self, text: str, background: str, foreground: str = TEXT_PRIMARY) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            f"background-color: {background}; color: {foreground}; padding: 6px 16px; border-radius: 10px; font-weight: 600;"
        )
        return label
class DownloadPage(QWidget):
    download_requested = Signal(str, str)

    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        card = QFrame()
        card.setObjectName("card")
        card.setFrameShape(QFrame.Shape.NoFrame)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)

        title = QLabel("Download e aggiornamento")
        title.setStyleSheet("font-size: 24px; font-weight: 700; letter-spacing: 0.05em;")
        card_layout.addWidget(title)

        self.repo_input = QLineEdit(self)
        self.repo_input.setPlaceholderText("https://github.com/<user>/acc_coach_ai/archive/refs/heads/main.zip")

        self.install_input = QLineEdit(self)
        self.install_input.setReadOnly(True)
        browse_btn = QPushButton("Sfoglia")
        browse_btn.clicked.connect(self._select_install_dir)

        form = QFormLayout()
        form.addRow("URL pacchetto ZIP:", self.repo_input)
        install_row = QHBoxLayout()
        install_row.addWidget(self.install_input)
        install_row.addWidget(browse_btn)
        form.addRow("Cartella installazione:", install_row)
        card_layout.addLayout(form)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {ACCENT_GLOW}; font-size: 14px;")
        card_layout.addWidget(self.status_label)

        action_btn = QPushButton("Scarica / Aggiorna")
        action_btn.setObjectName("accentButton")
        action_btn.clicked.connect(self._handle_download)
        card_layout.addWidget(action_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(card)
        layout.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        self.repo_input.setText(self.env_manager.values.get("REPO_ZIP_URL", DEFAULT_VALUES["REPO_ZIP_URL"]))
        self.install_input.setText(str(self.env_manager.get_install_dir()))
        self.status_label.setText("")

    def _select_install_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Seleziona cartella installazione", self.install_input.text())
        if chosen:
            self.install_input.setText(chosen)

    def _handle_download(self) -> None:
        url = self.repo_input.text().strip()
        target = self.install_input.text().strip()
        if not url or "<user>" in url:
            QMessageBox.warning(self, "URL mancante", "Inserisci un URL ZIP valido del repository.")
            return
        if not target:
            QMessageBox.warning(self, "Cartella mancante", "Seleziona la cartella di installazione.")
            return
        self.status_label.setText("Download in corso...")
        self.download_requested.emit(url, target)


class ConfigPage(QWidget):
    values_saved = Signal(dict)

    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        self.api_inputs: Dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        card = QFrame()
        card.setObjectName("card")
        card.setFrameShape(QFrame.Shape.NoFrame)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)

        title = QLabel("Impostazioni e integrazioni")
        title.setStyleSheet("font-size: 24px; font-weight: 700; letter-spacing: 0.05em;")
        card_layout.addWidget(title)

        form = QFormLayout()
        for key, label in ENV_FIELDS:
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.EchoMode.Password if "KEY" in key else QLineEdit.EchoMode.Normal)
            self.api_inputs[key] = edit
            form.addRow(label + ":", edit)
        card_layout.addLayout(form)

        self.acc_checkbox = QCheckBox("Connettiti ad Assetto Corsa Competizione in tempo reale (UDP)")
        card_layout.addWidget(self.acc_checkbox)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(12)
        self.acc_status_pill = QLabel("Inattiva")
        self.acc_status_pill.setObjectName("statusPill")
        self.acc_status_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_row.addWidget(self.acc_status_pill, alignment=Qt.AlignmentFlag.AlignLeft)
        self.acc_test_btn = QPushButton("Test connessione")
        self.acc_test_btn.clicked.connect(self._test_connection)
        status_row.addWidget(self.acc_test_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        status_row.addStretch(1)
        card_layout.addLayout(status_row)

        acc_grid = QGridLayout()
        acc_grid.setHorizontalSpacing(16)
        host_label = QLabel("Host UDP")
        host_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        self.acc_host_input = QLineEdit()
        port_label = QLabel("Porta UDP")
        port_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        self.acc_port_input = QLineEdit()
        self.acc_port_input.setValidator(QIntValidator(1, 65535, self))
        acc_grid.addWidget(host_label, 0, 0)
        acc_grid.addWidget(self.acc_host_input, 1, 0)
        acc_grid.addWidget(port_label, 0, 1)
        acc_grid.addWidget(self.acc_port_input, 1, 1)
        card_layout.addLayout(acc_grid)

        self.acc_hint_base = "Apri Assetto Corsa Competizione > Opzioni > Telemetria e abilita UDP con l'host indicato sopra."
        self.acc_hint = QLabel(self.acc_hint_base)
        self.acc_hint.setWordWrap(True)
        self.acc_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        card_layout.addWidget(self.acc_hint)

        save_btn = QPushButton("Salva configurazione")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self._save)
        card_layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(card)
        layout.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        for key, edit in self.api_inputs.items():
            edit.setText(self.env_manager.values.get(key, DEFAULT_VALUES.get(key, "")))
        self.acc_checkbox.setChecked(self.env_manager.values.get("ACC_USE_LIVE", "1") == "1")
        self.acc_host_input.setText(self.env_manager.values.get("ACC_UDP_HOST", DEFAULT_VALUES["ACC_UDP_HOST"]))
        self.acc_port_input.setText(self.env_manager.values.get("ACC_UDP_PORT", DEFAULT_VALUES["ACC_UDP_PORT"]))
        self._update_acc_controls()
        self._update_connection_status("inactive", "In attesa di test connessione.")

    def _save(self) -> None:
        updates: Dict[str, str] = {}
        for key, edit in self.api_inputs.items():
            updates[key] = edit.text().strip()
        if self.acc_checkbox.isChecked():
            port_val = self.acc_port_input.text().strip()
            if not port_val.isdigit():
                QMessageBox.warning(self, "Porta non valida", "Inserisci una porta UDP numerica valida.")
                return
        else:
            port_val = self.acc_port_input.text().strip() or DEFAULT_VALUES["ACC_UDP_PORT"]
        updates["ACC_USE_LIVE"] = "1" if self.acc_checkbox.isChecked() else "0"
        updates["ACC_UDP_HOST"] = self.acc_host_input.text().strip() or DEFAULT_VALUES["ACC_UDP_HOST"]
        updates["ACC_UDP_PORT"] = port_val
        self.env_manager.update(updates)
        self.values_saved.emit(updates)
        QMessageBox.information(self, "Salvato", "Impostazioni aggiornate correttamente.")
        self._update_acc_controls()
        self._update_connection_status("inactive" if self.acc_checkbox.isChecked() else "disabled", "Configurazione salvata.")

    def _update_acc_controls(self) -> None:
        enabled = self.acc_checkbox.isChecked()
        self.acc_host_input.setEnabled(enabled)
        self.acc_port_input.setEnabled(enabled)
        if enabled:
            self.acc_hint.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
            self.acc_test_btn.setEnabled(True)
        else:
            self.acc_hint.setStyleSheet(f"color: {ACCENT_GLOW}; font-size: 13px;")
            self.acc_test_btn.setEnabled(False)
            self._update_connection_status("disabled", "Connessione disattivata.")

    def _update_connection_status(self, state: str, message: str) -> None:
        colors = {
            "inactive": ("Inattiva", "rgba(255,255,255,0.08)", TEXT_MUTED),
            "pending": ("In attesa", "rgba(240,180,41,0.18)", WARNING_AMBER),
            "active": ("Live", "rgba(33,186,117,0.2)", SUCCESS_GREEN),
            "error": ("Errore", "rgba(156,42,35,0.25)", ACCENT_GLOW),
            "disabled": ("Disattiva", "rgba(255,255,255,0.05)", TEXT_MUTED),
        }
        label, bg, fg = colors.get(state, colors["inactive"])
        self.acc_status_pill.setText(label)
        self.acc_status_pill.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 6px 12px; border-radius: 12px; font-weight: 600;"
        )
        hint_text = self.acc_hint_base
        if message:
            hint_text = f"{self.acc_hint_base}\n{message}"
        self.acc_hint.setText(hint_text)

    def _test_connection(self) -> None:
        if not self.acc_checkbox.isChecked():
            self._update_connection_status("disabled", "Abilita prima la connessione live.")
            return
        host = self.acc_host_input.text().strip() or DEFAULT_VALUES["ACC_UDP_HOST"]
        port_text = self.acc_port_input.text().strip() or DEFAULT_VALUES["ACC_UDP_PORT"]
        if not port_text.isdigit():
            self._update_connection_status("error", "Porta non valida.")
            return
        port = int(port_text)
        self._update_connection_status("pending", "Test connessione in corso...")
        QApplication.processEvents()
        try:
            info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_DGRAM)
            family, socktype, proto, _canon, sockaddr = info[0]
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(1.0)
            start = time.perf_counter()
            sock.sendto(b"ping", sockaddr)
            elapsed = (time.perf_counter() - start) * 1000
            sock.close()
            self._update_connection_status("active", f"UDP pronto (≈{elapsed:.0f} ms)")
        except Exception as exc:  # pragma: no cover - fallback
            self._update_connection_status("error", f"Errore connessione: {exc}")


class CoachPage(QWidget):
    def __init__(self, env_manager: EnvManager, controller: ServiceController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        self.controller = controller
        self._updating_tts = False
        self.session_active = False
        self.controls_enabled = True
        self._log_entries: list[tuple[str, str, str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        card = QFrame()
        card.setObjectName("card")
        card.setFrameShape(QFrame.Shape.NoFrame)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)

        title_row = QHBoxLayout()
        title = QLabel("Sessione in tempo reale")
        title.setStyleSheet("font-size: 24px; font-weight: 700; letter-spacing: 0.05em;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.tts_toggle = QCheckBox("Feedback vocale (ElevenLabs)")
        self.tts_toggle.setAccessibleName("Attiva feedback vocale ElevenLabs")
        self.tts_toggle.stateChanged.connect(self._on_tts_toggle)
        title_row.addWidget(self.tts_toggle)
        card_layout.addLayout(title_row)

        self.info_label = QLabel("Configura le API e scegli la sorgente dati per iniziare.")
        self.info_label.setStyleSheet(f"color: {ACCENT_GLOW}; font-size: 14px;")
        card_layout.addWidget(self.info_label)

        self.session_indicator = QLabel("Sessione inattiva")
        self.session_indicator.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.session_indicator.setStyleSheet(self._indicator_style("rgba(255,255,255,0.05)", TEXT_MUTED))
        self.session_indicator.setAccessibleName("Stato della sessione coach")
        card_layout.addWidget(self.session_indicator)

        source_row = QHBoxLayout()
        source_row.setSpacing(12)
        source_column = QVBoxLayout()
        source_column.setSpacing(6)
        self.source_group = QButtonGroup(self)
        self.live_radio = QRadioButton("Live da Assetto Corsa Competizione")
        self.live_radio.setProperty("mode", "live")
        self.live_radio.setAccessibleName("Sorgente live da ACC")
        self.file_radio = QRadioButton("File JSONL locale")
        self.file_radio.setProperty("mode", "simulation")
        self.file_radio.setAccessibleName("Sorgente da file JSONL")
        self.source_group.addButton(self.live_radio)
        self.source_group.addButton(self.file_radio)
        source_column.addWidget(self.live_radio)
        source_column.addWidget(self.file_radio)
        source_row.addLayout(source_column, stretch=0)
        self.simulation_input = QLineEdit(self)
        self.simulation_input.setPlaceholderText("Percorso file JSONL per simulazione")
        self.simulation_input.setEnabled(False)
        self._browse_btn = QPushButton("Sfoglia file")
        self._browse_btn.clicked.connect(self._select_simulation)
        self._browse_btn.setEnabled(False)
        input_column = QHBoxLayout()
        input_column.setContentsMargins(0, 0, 0, 0)
        input_column.setSpacing(12)
        input_column.addWidget(self.simulation_input)
        input_column.addWidget(self._browse_btn)
        source_row.addLayout(input_column, stretch=1)
        card_layout.addLayout(source_row)
        self.source_group.buttonToggled.connect(self._on_source_changed)

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.start_btn = QPushButton("Avvia sessione")
        self.start_btn.clicked.connect(self._start_session)
        self.stop_btn = QPushButton("Termina sessione")
        self.stop_btn.clicked.connect(self._stop_session)
        self.export_btn = QPushButton("Esporta suggerimenti")
        self.export_btn.clicked.connect(self._export_feedback)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.export_btn)
        card_layout.addLayout(buttons)

        self._start_btn_base_style = self._neutral_button_style()
        self._stop_btn_base_style = self._neutral_button_style()
        self.start_btn.setStyleSheet(self._start_btn_base_style)
        self.stop_btn.setStyleSheet(self._stop_btn_base_style)

        self.status_label = QLabel("Sessione non attiva.")
        self.status_label.setAccessibleName("Descrizione stato sessione")
        self._set_session_state(False)
        card_layout.addWidget(self.status_label)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(12)
        filter_label = QLabel("Filtro log")
        filter_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        filter_row.addWidget(filter_label)
        self.log_filter = QComboBox()
        self.log_filter.addItems(["Tutti", "Info", "Warning", "Errore"])
        self.log_filter.setAccessibleName("Filtro severità log coach")
        filter_row.addWidget(self.log_filter, stretch=0)
        filter_row.addStretch(1)
        card_layout.addLayout(filter_row)
        self.log_filter.currentTextChanged.connect(self._refresh_log)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("I feedback del coach appariranno qui.")
        self.log_view.setAccessibleName("Log feedback coach AI")
        self.log_view.setAccessibleDescription("Aggiornamenti in tempo reale dal coach AI.")
        self.log_view.setProperty("aria-live", "polite")
        card_layout.addWidget(self.log_view, stretch=1)

        layout.addWidget(card)
        layout.addStretch(1)

        controller.feedback_received.connect(self._on_feedback)
        controller.status_changed.connect(self._on_status)

    def refresh(self) -> None:
        use_live = self.env_manager.values.get("ACC_USE_LIVE", "1") == "1"
        self.source_group.blockSignals(True)
        self.live_radio.setChecked(use_live)
        self.file_radio.setChecked(not use_live)
        self.source_group.blockSignals(False)
        self._apply_source_mode("live" if use_live else "simulation")
        self._updating_tts = True
        self.tts_toggle.setChecked(self.env_manager.values.get("ENABLE_TTS", "0") == "1")
        self._updating_tts = False
        self._set_session_state(False)
        self.set_enabled(self.controls_enabled)

    def set_enabled(self, enabled: bool) -> None:
        self.controls_enabled = enabled
        if not enabled:
            self._set_session_state(False)
        self.tts_toggle.setEnabled(enabled)
        self.live_radio.setEnabled(enabled)
        self.file_radio.setEnabled(enabled)
        self.simulation_input.setEnabled(enabled and self.file_radio.isChecked())
        self._browse_btn.setEnabled(enabled and self.file_radio.isChecked())
        self.start_btn.setEnabled(enabled and not self.session_active)
        self.stop_btn.setEnabled(enabled and self.session_active)
        if not enabled:
            self.info_label.setText("Inserisci la Google AI Studio API key nella sezione Impostazioni per attivare il coach.")
        else:
            self._apply_source_mode(self._current_mode())

    def _current_mode(self) -> str:
        return "simulation" if self.file_radio.isChecked() else "live"

    def _on_source_changed(self, button: QAbstractButton, checked: bool) -> None:
        if not checked:
            return
        mode = button.property("mode") or "live"
        self._apply_source_mode(mode)

    def _apply_source_mode(self, mode: str) -> None:
        is_sim = mode == "simulation"
        self.simulation_input.setEnabled(is_sim and self.controls_enabled)
        self._browse_btn.setEnabled(is_sim and self.controls_enabled)
        if not self.controls_enabled:
            return
        if is_sim:
            self.info_label.setText("Seleziona un file JSONL di telemetria per la simulazione.")
        else:
            self.info_label.setText("Assicurati che ACC abbia la telemetria UDP attiva e avvia la sessione.")
        self.start_btn.setEnabled(not self.session_active)
        self.stop_btn.setEnabled(self.session_active)

    def _on_tts_toggle(self, state: int) -> None:
        if self._updating_tts:
            return
        enabled = state == int(Qt.CheckState.Checked)
        self.env_manager.update({"ENABLE_TTS": "1" if enabled else "0"})

    def _select_simulation(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona file simulazione",
            str(self.env_manager.get_install_dir()),
            "JSON Lines (*.jsonl);;Tutti i file (*)",
        )
        if file_path:
            self.simulation_input.setText(file_path)

    def _start_session(self) -> None:
        if not self.env_manager.has_required_api_keys():
            QMessageBox.warning(
                self,
                "API mancanti",
                "Inserisci la Google AI Studio API key obbligatoria prima di avviare il coach.",
            )
            return
        if self.session_active or self.controller.is_running:
            self._set_session_state(True)
            self.status_label.setText("Sessione già attiva")
            self.status_label.setStyleSheet(f"color: {SUCCESS_GREEN}; font-size: 14px;")
            return
        mode = self._current_mode()
        if mode == "simulation":
            path = Path(self.simulation_input.text().strip())
            if not path.exists():
                QMessageBox.warning(self, "File mancante", "Seleziona un file JSONL valido per la simulazione.")
                return
            config = CollectorConfig(
                mode=CollectorMode.SIMULATION,
                simulation_file=str(path),
                loop_simulation=True,
                playback_rate=1.0,
            )
        else:
            host = self.env_manager.values.get("ACC_UDP_HOST", DEFAULT_VALUES["ACC_UDP_HOST"])
            port_value = self.env_manager.values.get("ACC_UDP_PORT", DEFAULT_VALUES["ACC_UDP_PORT"])
            try:
                port = int(port_value)
            except ValueError:
                QMessageBox.warning(self, "Porta non valida", "Imposta una porta numerica valida per ACC.")
                return
            config = CollectorConfig(mode=CollectorMode.UDP, udp_host=host, udp_port=port)
        self.env_manager.update({"ACC_USE_LIVE": "0" if mode == "simulation" else "1"})
        self._set_session_pending("Avvio sessione...", activating=True)
        QTimer.singleShot(0, lambda cfg=config: self.controller.start(cfg))

    def _stop_session(self) -> None:
        if not self.session_active and not self.controller.is_running:
            self._set_session_state(False)
            return
        self._set_session_pending("Terminazione sessione...", activating=False)
        QTimer.singleShot(0, self.controller.stop)

    def _export_feedback(self) -> None:
        install_dir = self.env_manager.get_install_dir()
        reports_dir = install_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        file_path = reports_dir / "coach_feedback.txt"
        file_path.write_text(self.log_view.toPlainText(), encoding="utf-8")
        QMessageBox.information(self, "Esportato", f"Log salvato in {file_path}")

    def _on_feedback(self, event: dict) -> None:
        message = event.get("message", "")
        section = event.get("section", "Generale")
        severity = event.get("severity", "info").lower()
        entry = (datetime.now().strftime("%H:%M:%S"), severity, f"{section}: {message}")
        self._log_entries.append(entry)
        self._refresh_log()

    def _refresh_log(self) -> None:
        level = self.log_filter.currentText().lower()
        self.log_view.clear()
        for timestamp, severity, message in self._log_entries:
            if level != "tutti" and severity != level:
                continue
            self.log_view.append(f"[{timestamp}] [{severity.upper()}] {message}")
        sb = self.log_view.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _on_status(self, status: str) -> None:
        lower = status.lower()
        if "avviat" in lower or "attiva" in lower:
            self._set_session_state(True)
        elif "terminata" in lower or "ferm" in lower or "non attiva" in lower or "errore" in lower:
            self._set_session_state(False)
        else:
            self.status_label.setText(status)
            self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")

    def _set_session_pending(self, text: str, activating: bool) -> None:
        self.session_indicator.setText(text)
        self.session_indicator.setStyleSheet(self._indicator_style("rgba(240,180,41,0.25)", TEXT_PRIMARY))
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {WARNING_AMBER}; font-size: 14px;")
        if activating:
            self.start_btn.setStyleSheet(self._accent_button_style())
        else:
            self.start_btn.setStyleSheet(self._start_btn_base_style)
        self.stop_btn.setStyleSheet(self._stop_btn_base_style)

    def _set_session_state(self, active: bool) -> None:
        self.session_active = active
        if active:
            self.session_indicator.setText("Sessione attiva")
            self.session_indicator.setStyleSheet(self._indicator_style("rgba(33,186,117,0.2)", TEXT_PRIMARY))
            self.start_btn.setStyleSheet(self._accent_button_style())
            self.stop_btn.setStyleSheet(self._stop_btn_base_style)
            self.status_label.setText("Sessione attiva")
            self.status_label.setStyleSheet(f"color: {SUCCESS_GREEN}; font-size: 14px;")
        else:
            self.session_indicator.setText("Sessione inattiva")
            self.session_indicator.setStyleSheet(self._indicator_style("rgba(255,255,255,0.05)", TEXT_MUTED))
            self.start_btn.setStyleSheet(self._start_btn_base_style)
            self.stop_btn.setStyleSheet(self._stop_btn_base_style)
            self.status_label.setText("Sessione non attiva.")
            self.status_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")
        self.start_btn.setEnabled(self.controls_enabled and not active)
        self.stop_btn.setEnabled(self.controls_enabled and active)

    def _indicator_style(self, bg: str, fg: str) -> str:
        return (
            f"background-color: {bg}; color: {fg}; padding: 6px 12px; border-radius: 12px;"
            " font-size: 13px; letter-spacing: 0.04em;"
        )

    def _neutral_button_style(self) -> str:
        return (
            f"background-color: {SECONDARY_BG}; color: {TEXT_PRIMARY}; border: 1px solid transparent;"
            " padding: 10px 16px; border-radius: 12px; font-size: 15px;"
        )

    def _accent_button_style(self) -> str:
        return (
            f"background-color: {ACCENT_RED}; color: {TEXT_PRIMARY}; border: none;"
            " padding: 10px 16px; border-radius: 12px; font-size: 15px;"
        )
class LapAnalysisPage(QWidget):
    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        self.analysis_data: dict[str, dict[str, dict]] = {}
        self.current_track: Optional[str] = None
        self.current_car: Optional[str] = None
        self.current_analysis: Optional[dict[str, object]] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        filter_card = QFrame()
        filter_card.setObjectName("sectionCard")
        filter_card.setFrameShape(QFrame.Shape.NoFrame)
        filter_layout = QVBoxLayout(filter_card)
        filter_layout.setContentsMargins(28, 28, 28, 28)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        title = QLabel("Lap Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: 700; letter-spacing: 0.05em;")
        header_row.addWidget(title)
        header_row.addStretch(1)
        refresh_btn = QPushButton("Aggiorna dati")
        refresh_btn.setObjectName("accentButton")
        refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(refresh_btn)
        filter_layout.addLayout(header_row)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(16)

        self.track_combo = QComboBox()
        self.track_combo.setPlaceholderText("Seleziona pista")
        track_col = QVBoxLayout()
        track_label = QLabel("Pista")
        track_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        track_col.addWidget(track_label)
        track_col.addWidget(self.track_combo)
        filter_row.addLayout(track_col, stretch=1)

        self.car_combo = QComboBox()
        self.car_combo.setPlaceholderText("Seleziona vettura")
        car_col = QVBoxLayout()
        car_label = QLabel("Vettura")
        car_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        car_col.addWidget(car_label)
        car_col.addWidget(self.car_combo)
        filter_row.addLayout(car_col, stretch=1)

        filter_layout.addLayout(filter_row)

        self.ideal_label = QLabel("Giro ideale non disponibile")
        self.ideal_label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")
        filter_layout.addWidget(self.ideal_label)

        content_row = QHBoxLayout()
        content_row.setSpacing(18)

        self.lap_list = QListWidget()
        self.lap_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.lap_list.setMinimumWidth(220)
        content_row.addWidget(self.lap_list, stretch=0)

        sector_column = QVBoxLayout()
        sector_title = QLabel("Delta settori (vs giro ideale)")
        sector_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        sector_column.addWidget(sector_title)

        sectors_row = QHBoxLayout()
        sectors_row.setSpacing(12)
        self.sector_labels: dict[str, QLabel] = {}
        for sector in ("S1", "S2", "S3"):
            sector_frame = QFrame()
            sector_frame.setObjectName("card")
            sector_frame.setFrameShape(QFrame.Shape.NoFrame)
            sector_layout = QVBoxLayout(sector_frame)
            sector_layout.setContentsMargins(18, 14, 18, 14)
            sector_caption = QLabel(sector)
            sector_caption.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px; letter-spacing: 0.08em;")
            sector_value = QLabel("-")
            sector_value.setStyleSheet("font-size: 20px; font-weight: 600;")
            sector_layout.addWidget(sector_caption)
            sector_layout.addWidget(sector_value)
            sector_layout.addStretch(1)
            sectors_row.addWidget(sector_frame)
            self.sector_labels[sector] = sector_value
        sector_column.addLayout(sectors_row)
        sector_column.addStretch(1)
        content_row.addLayout(sector_column, stretch=1)

        filter_layout.addLayout(content_row)

        self.state_label = QLabel("Seleziona pista e vettura per iniziare.")
        self.state_label.setStyleSheet(f"color: {ACCENT_GLOW}; font-size: 13px;")
        filter_layout.addWidget(self.state_label)

        layout.addWidget(filter_card)

        table_card = QFrame()
        table_card.setObjectName("sectionCard")
        table_card.setFrameShape(QFrame.Shape.NoFrame)
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(28, 28, 28, 28)

        table_title = QLabel("Delta tempo curva per curva")
        table_title.setStyleSheet("font-size: 20px; font-weight: 600;")
        table_layout.addWidget(table_title)

        self.delta_table = QTableWidget(0, 4)
        self.delta_table.setHorizontalHeaderLabels(["Curva", "Settore", "Delta (ms)", "Consiglio"])
        self.delta_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.delta_table.verticalHeader().setVisible(False)
        self.delta_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.delta_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table_layout.addWidget(self.delta_table)
        self.empty_state = QLabel("Seleziona pista e vettura, poi avvia una sessione per popolare i dati.")
        self.empty_state.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px;")
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        table_layout.addWidget(self.empty_state)
        self.empty_state.hide()
        self.delta_table.hide()

        layout.addWidget(table_card)
        layout.addStretch(1)

        self.track_combo.setEnabled(False)
        self.car_combo.setEnabled(False)
        self._clear_display()

        self.track_combo.currentTextChanged.connect(self._on_track_changed)
        self.car_combo.currentTextChanged.connect(self._on_car_changed)
        self.lap_list.currentRowChanged.connect(self._on_lap_selected)

    def refresh(self) -> None:
        self.analysis_data = load_lap_analysis_data(self.env_manager)
        self._populate_filters()

    def _populate_filters(self) -> None:
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        if not self.analysis_data:
            self.track_combo.setEnabled(False)
            self.car_combo.setEnabled(False)
            self._clear_display()
            self._set_message("Nessuna telemetria disponibile.")
            self.track_combo.blockSignals(False)
            return
        for track in sorted(self.analysis_data.keys()):
            self.track_combo.addItem(track)
        self.track_combo.setEnabled(True)
        self.track_combo.blockSignals(False)
        self.track_combo.setCurrentIndex(0)
        self._on_track_changed(self.track_combo.currentText())

    def _on_track_changed(self, track: str) -> None:
        self.current_track = track
        cars = sorted(self.analysis_data.get(track, {}).keys()) if track in self.analysis_data else []
        self.car_combo.blockSignals(True)
        self.car_combo.clear()
        if not cars:
            self.car_combo.setEnabled(False)
            self.car_combo.blockSignals(False)
            self._clear_display()
            self._set_message("Nessun giro disponibile per la pista selezionata.")
            return
        for car in cars:
            self.car_combo.addItem(car)
        self.car_combo.setEnabled(True)
        self.car_combo.blockSignals(False)
        self.car_combo.setCurrentIndex(0)
        self._on_car_changed(self.car_combo.currentText())

    def _on_car_changed(self, car: str) -> None:
        self.current_car = car
        track = self.current_track
        analysis = self.analysis_data.get(track, {}).get(car) if track else None
        self.current_analysis = analysis
        self.lap_list.blockSignals(True)
        self.lap_list.clear()
        if not analysis or not analysis.get("laps"):
            self.lap_list.blockSignals(False)
            self._clear_table()
            self._set_message("Nessun giro registrato per la combinazione selezionata.")
            self.ideal_label.setText("Giro ideale non disponibile")
            self.delta_table.hide()
            self.empty_state.show()
            return
        self.empty_state.hide()
        self.delta_table.show()
        laps = sorted(analysis["laps"], key=lambda lap: lap.get("lap_number") or 0)
        for lap in laps:
            item_text = f"Lap {lap.get('lap_number', '?')} - {format_lap_time(lap.get('lap_time_ms'))}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, lap)
            self.lap_list.addItem(item)
        self.lap_list.blockSignals(False)
        ideal = analysis.get("ideal")
        if ideal:
            self.ideal_label.setText(
                f"Giro ideale Lap {ideal.get('lap_number', '?')} - {format_lap_time(ideal.get('lap_time_ms'))}"
            )
        else:
            self.ideal_label.setText("Giro ideale non disponibile")
        self._set_message("Seleziona un giro da confrontare con il giro ideale.")
        if self.lap_list.count():
            self.lap_list.setCurrentRow(0)

    def _on_lap_selected(self, row: int) -> None:
        if self.current_analysis is None or row < 0:
            self._clear_table()
            return
        item = self.lap_list.item(row)
        if item is None:
            self._clear_table()
            return
        lap_data = item.data(Qt.ItemDataRole.UserRole)
        ideal = self.current_analysis.get("ideal") if self.current_analysis else None
        self._update_sector_summary(lap_data, ideal)
        self._update_delta_table(lap_data, ideal)

    def _update_delta_table(self, lap: dict, ideal: Optional[dict]) -> None:
        sections = lap.get("sections") or []
        ideal_map = {}
        if ideal and ideal.get("sections"):
            ideal_map = {sec.get("section_id"): sec for sec in ideal["sections"]}
        self.delta_table.setRowCount(len(sections))
        for idx, section in enumerate(sections):
            name = section.get("name") or section.get("section_id") or f"Curva {idx + 1}"
            sector = self._sector_from_index(idx, len(sections))
            current_delta = section.get("delta_time_ms")
            baseline = None
            if ideal_map:
                ref = ideal_map.get(section.get("section_id"))
                if ref is not None:
                    baseline = ref.get("delta_time_ms")
            delta_value = None
            if current_delta is not None:
                delta_value = float(current_delta)
                if baseline is not None:
                    delta_value -= float(baseline)
            self.delta_table.setItem(idx, 0, QTableWidgetItem(name))
            self.delta_table.setItem(idx, 1, QTableWidgetItem(sector))
            self.delta_table.setItem(idx, 2, QTableWidgetItem(self._format_delta(delta_value)))
        if not sections:
            self.delta_table.setRowCount(0)

    def _update_sector_summary(self, lap: dict, ideal: Optional[dict]) -> None:
        sections = lap.get("sections") or []
        total_sections = len(sections)
        ideal_map = {}
        if ideal and ideal.get("sections"):
            ideal_map = {sec.get("section_id"): sec for sec in ideal["sections"]}
        chunk = max(1, math.ceil(total_sections / 3)) if total_sections else 1
        for index, sector in enumerate(("S1", "S2", "S3")):
            start = index * chunk
            end = total_sections if index == 2 else min(total_sections, (index + 1) * chunk)
            subset = sections[start:end]
            if not subset:
                self.sector_labels[sector].setText("-")
                continue
            value = 0.0
            for sec in subset:
                current = sec.get("delta_time_ms") or 0.0
                baseline = 0.0
                if ideal_map:
                    ref = ideal_map.get(sec.get("section_id"))
                    if ref and ref.get("delta_time_ms") is not None:
                        baseline = float(ref["delta_time_ms"])
                value += float(current) - baseline
            self.sector_labels[sector].setText(self._format_delta(value))

    def _sector_from_index(self, index: int, total: int) -> str:
        if total <= 0:
            return "S1"
        threshold = math.ceil(total / 3)
        if index < threshold:
            return "S1"
        if index < threshold * 2:
            return "S2"
        return "S3"

    def _format_delta(self, value: Optional[float]) -> str:
        if value is None:
            return "-"
        return f"{value:+.0f} ms"

    def _set_message(self, text: str) -> None:
        self.state_label.setText(text)

    def _clear_display(self) -> None:
        self._clear_table()
        for label in self.sector_labels.values():
            label.setText("-")
        self.lap_list.clear()
        self.ideal_label.setText("Giro ideale non disponibile")
        self.delta_table.hide()
        self.empty_state.show()

    def _clear_table(self) -> None:
        self.delta_table.setRowCount(0)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ACC Coach AI - Desktop")
        self.resize(1080, 720)
        window_icon = load_app_icon()
        if not window_icon.isNull():
            self.setWindowIcon(window_icon)

        self.env_manager = EnvManager()
        self.service_controller = ServiceController()
        self.download_workers: list[DownloadWorker] = []

        container = QWidget()
        root_layout = QHBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = Sidebar(
            {
                "home": "Dashboard",
                "coach": "Coach AI",
                "analysis": "Lap Analysis",
                "download": "Download",
                "settings": "Impostazioni",
            }
        )
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(240)
        self.sidebar.selection_changed.connect(self._on_menu_selected)
        root_layout.addWidget(self.sidebar)

        self.divider = QFrame()
        self.divider.setObjectName("sidebarDivider")
        self.divider.setFrameShape(QFrame.Shape.NoFrame)
        self.divider.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.divider.setFixedWidth(2)
        root_layout.addWidget(self.divider)

        self.main_panel = QWidget()
        main_layout = QVBoxLayout(self.main_panel)
        main_layout.setContentsMargins(32, 24, 32, 24)
        main_layout.setSpacing(24)

        self.header_label = QLabel("Dashboard")
        self.header_label.setStyleSheet("font-size: 30px; font-weight: 700; letter-spacing: 0.04em;")
        main_layout.addWidget(self.header_label)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        root_layout.addWidget(self.main_panel, stretch=1)
        self.setCentralWidget(container)

        self.home_page = HomePage(self.env_manager)
        self.coach_page = CoachPage(self.env_manager, self.service_controller)
        self.lap_analysis_page = LapAnalysisPage(self.env_manager)
        self.download_page = DownloadPage(self.env_manager)
        self.config_page = ConfigPage(self.env_manager)
        self._animations: list[QPropertyAnimation] = []

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.coach_page)
        self.stack.addWidget(self.lap_analysis_page)
        self.stack.addWidget(self.download_page)
        self.stack.addWidget(self.config_page)

        self.download_page.download_requested.connect(self._start_download)
        self.config_page.values_saved.connect(self._apply_updates)

        self.apply_theme()
        self.lap_analysis_page.refresh()
        self.sidebar.select("home")
        self.stack.setCurrentIndex(0)
        self.refresh_home()
        self.update_availability()

    def apply_theme(self) -> None:
        style = f"""
        QMainWindow {{
            background-color: {PRIMARY_BG};
            color: {TEXT_PRIMARY};
        }}
        QWidget {{
            background-color: {PRIMARY_BG};
            color: {TEXT_PRIMARY};
            font-family: 'Segoe UI', 'Inter', sans-serif;
            font-size: 15px;
            line-height: 1.48;
        }}
        QFrame#card {{
            background-color: {CARD_BG};
            background: qlineargradient(0, 0, 0, 1, stop: 0 rgba(156,42,35,0.12), stop: 1 rgba(24,26,30,1));
            border-radius: 20px;
            border: 1px solid {SURFACE_BORDER};
            box-shadow: none;
        }}
        QFrame#card:hover {{
            background: qlineargradient(0, 0, 0, 1, stop: 0 rgba(210,59,59,0.16), stop: 1 rgba(30,33,39,1));
        }}
        QFrame#sectionCard {{
            background-color: {SURFACE_MEDIUM};
            border-radius: 26px;
            border: 1px solid {SURFACE_BORDER};
            padding: 28px;
        }}
        QFrame#sectionInner {{
            background-color: {CARD_BG};
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.04);
        }}
        QFrame#card QLabel {{
            background-color: transparent;
            border: none;
            text-shadow: none;
        }}
        QLabel#statusPill {{
            padding: 6px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        QLabel#cardValue {{
            color: {TEXT_PRIMARY};
            font-size: 32px;
            font-weight: 600;
            text-shadow: none;
            padding-left: 12px;
        }}
        QLabel#cardUnit {{
            color: {TEXT_MUTED};
            font-size: 14px;
            font-weight: 500;
        }}
        QLabel#cardTrend {{
            font-size: 13px;
            font-weight: 600;
        }}
        QLabel#statusDot {{
            font-size: 16px;
        }}
        QLabel {{
            text-shadow: none;
        }}
        QPushButton {{
            background-color: {SECONDARY_BG};
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 12px;
            padding: 10px 16px;
            color: {TEXT_PRIMARY};
            font-size: 15px;
            box-shadow: none;
        }}
        QPushButton:hover {{
            background-color: {ACCENT_RED_HOVER};
        }}
        QPushButton:disabled {{
            background-color: #2a2d36;
            color: {TEXT_MUTED};
        }}
        QPushButton#accentButton {{
            background-color: {ACCENT_RED};
            border: 1px solid {ACCENT_RED};
            font-weight: bold;
            font-size: 16px;
        }}
        QPushButton#accentButton:hover {{
            background-color: {ACCENT_RED_HOVER};
        }}
        QPushButton#sidebarButton {{
            text-align: left;
            padding: 12px 18px;
            border-radius: 14px;
            background-color: transparent;
            border: none;
            color: {TEXT_PRIMARY};
            font-size: 17px;
            letter-spacing: 0.05em;
        }}
        QPushButton#sidebarButton:hover {{
            background-color: rgba(78, 0, 0, 0.45);
        }}
        QPushButton#sidebarButton:checked {{
            background-color: rgba(78, 0, 0, 0.8);
            color: {TEXT_PRIMARY};
            border: 1px solid rgba(78, 0, 0, 0.9);
        }}
        QLineEdit, QComboBox {{
            background-color: {SECONDARY_BG};
            border-radius: 10px;
            padding: 8px 12px;
            border: 1px solid rgba(255,255,255,0.06);
            color: {TEXT_PRIMARY};
            box-shadow: none;
            selection-background-color: {ACCENT_RED};
            selection-color: {TEXT_PRIMARY};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QTextEdit {{
            background-color: {SECONDARY_BG};
            border-radius: 12px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: none;
        }}
        QPushButton:focus, QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{
            border: 1px solid #D23B3B;
            box-shadow: 0 0 0 2px rgba(210, 59, 59, 0.28);
        }}
        QWidget#Sidebar {{
            background-color: #121216;
        }}
        QFrame#sidebarDivider {{
            background-color: {SURFACE_BORDER};
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(139,31,43,0.6);
            border-radius: 5px;
        }}
        """
        self.setStyleSheet(style)

    def _on_menu_selected(self, key: str) -> None:
        index_map = {"home": 0, "coach": 1, "analysis": 2, "download": 3, "settings": 4}
        titles = {
            "home": "Dashboard",
            "coach": "Coach AI",
            "analysis": "Lap Analysis",
            "download": "Download e aggiornamento",
            "settings": "Impostazioni",
        }
        self.header_label.setText(titles.get(key, ""))
        if key == "home":
            self.refresh_home()
        elif key == "coach":
            self.coach_page.refresh()
        elif key == "analysis":
            self.lap_analysis_page.refresh()
        elif key == "download":
            self.download_page.refresh()
        elif key == "settings":
            self.config_page.refresh()
        if key in index_map:
            self.stack.setCurrentIndex(index_map[key])
            target = self.stack.widget(index_map[key])
            self._animate_widget(target)
            self._animate_widget(self.header_label)

    def refresh_home(self) -> None:
        metrics = get_last_session_summary(self.env_manager)
        api_ready = self.env_manager.has_required_api_keys()
        self.home_page.update_metrics(metrics, api_ready)

    def update_availability(self) -> None:
        api_ready = self.env_manager.has_required_api_keys()
        self.coach_page.set_enabled(api_ready)
        self.home_page.set_api_ready(api_ready)

    def _apply_updates(self, updates: Dict[str, str]) -> None:
        self.env_manager.update(updates)
        self.statusBar().showMessage("Configurazione salvata", 3000)
        self.update_availability()
        self.refresh_home()
        self.lap_analysis_page.refresh()

    def _start_download(self, url: str, target: str) -> None:
        self.env_manager.update({"REPO_ZIP_URL": url, "INSTALL_DIR": target})
        target_dir = Path(target).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        worker = DownloadWorker(url, target_dir)
        self.download_workers.append(worker)
        worker.finished.connect(self._on_download_finished)
        worker.start()
        self.statusBar().showMessage("Download in corso...")

    def _on_download_finished(self, success: bool, message: str) -> None:  # pragma: no cover
        sender = self.sender()
        if sender in self.download_workers:
            self.download_workers.remove(sender)  # type: ignore[arg-type]
        if success:
            QMessageBox.information(self, "Download completato", message)
            self.statusBar().showMessage("Download completato", 3000)
            self.refresh_home()
            self.lap_analysis_page.refresh()
        else:
            QMessageBox.critical(self, "Errore download", message)
            self.statusBar().showMessage("Errore download", 3000)

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover
        self.service_controller.stop()
        super().closeEvent(event)

    def _animate_widget(self, widget: QWidget) -> None:
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        animation = QPropertyAnimation(effect, b"opacity", widget)
        animation.setDuration(280)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        def _cleanup() -> None:
            if animation in self._animations:
                self._animations.remove(animation)

        animation.finished.connect(_cleanup)
        self._animations.append(animation)
        animation.start()


def main() -> None:
    app = QApplication(sys.argv)
    app_icon = load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
