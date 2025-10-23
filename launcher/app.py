from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional

import importlib.util


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


from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtGui import QCloseEvent, QPixmap  # noqa: E402  pylint: disable=wrong-import-position
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
    "OPENAI_API_KEY": "",
    "OPENAI_MODEL": "gpt-4o-mini",
    "ENABLE_TTS": "0",
    "TTS_PROVIDER": "gtts",
    "ELEVENLABS_API_KEY": "",
    "TTS_OUTPUT_DIR": "data/audio",
    "REPO_ZIP_URL": "https://github.com/<user>/acc_coach_ai/archive/refs/heads/main.zip",
    "INSTALL_DIR": str((Path.home() / "ACC_Coach_AI").resolve()),
}

ENV_FIELDS = [
    ("OPENAI_API_KEY", "OpenAI API Key"),
    ("OPENAI_MODEL", "Modello OpenAI"),
    ("ENABLE_TTS", "Abilita TTS (0/1)"),
    ("TTS_PROVIDER", "Provider TTS"),
    ("ELEVENLABS_API_KEY", "ElevenLabs API Key"),
    ("TTS_OUTPUT_DIR", "Cartella output TTS"),
    ("APP_ENV", "Ambiente"),
    ("LOG_LEVEL", "Log level"),
    ("REDIS_URL", "Redis URL"),
    ("DATABASE_URL", "Database URL"),
]

PRIMARY_BG = "#050505"
SECONDARY_BG = "#090909"
CARD_BG = "#101010"
ACCENT_RED = "#4E0000"
ACCENT_RED_HOVER = "#660000"
ACCENT_GLOW = "#8c1a1a"
TEXT_PRIMARY = "#f2f2f2"
TEXT_MUTED = "#9b9b9b"


def resource_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


class EnvManager:
    def __init__(self) -> None:
        self.values: Dict[str, str] = DEFAULT_VALUES.copy()
        self.load()

    def load(self) -> None:
        env_file = resource_path() / ".env"
        if not env_file.exists():
            return
        for line in env_file.read_text(encoding="utf-8").splitlines():
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
        required = ["OPENAI_API_KEY"]
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

    def start(self, simulation_file: Path) -> None:
        if self.running:
            self.status_changed.emit("Servizi gia attivi")
            return
        if not simulation_file.exists():
            self.status_changed.emit(f"File simulazione non trovato: {simulation_file}")
            return

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        future = asyncio.run_coroutine_threadsafe(self._async_start(simulation_file), self.loop)
        future.add_done_callback(self._handle_future)

    def stop(self) -> None:
        if not self.running or not self.loop:
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
        self.status_changed.emit("Servizi fermati")

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

    async def _async_start(self, simulation_file: Path) -> None:
        self.analytics = RealtimeAnalyticsEngine(on_feedback=self._on_feedback)
        await self.analytics.start()
        config = CollectorConfig(
            mode=CollectorMode.SIMULATION,
            simulation_file=str(simulation_file),
            loop_simulation=True,
            playback_rate=1.0,
        )
        self.collector = TelemetryCollector()
        await self.collector.start(config)
        self.running = True
        self.status_changed.emit("Servizi avviati (simulazione)")

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
        icon_path = resource_path() / "resources" / "acc_icon.ico"
        pix = QPixmap(str(icon_path))
        if not pix.isNull():
            icon_label.setPixmap(
                pix.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            icon_label.setText("ACC Coach")
            icon_label.setStyleSheet(f"color: {ACCENT_RED}; font-size: 24px; font-weight: 700; letter-spacing: 0.08em;")
        layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignLeft)

        subtitle = QLabel("Virtual racing coach")
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 13px;")
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        for key, label in items.items():
            btn = SidebarButton(label)
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


def build_card(title: str, value: str, subtitle: str = "") -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    frame.setFrameShape(QFrame.Shape.NoFrame)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    title_label = QLabel(title)
    title_label.setStyleSheet(
        f"color: {TEXT_MUTED}; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em;"
    )
    layout.addWidget(title_label)
    value_label = QLabel(value)
    value_label.setObjectName("cardValue")
    value_label.setStyleSheet(f"color: {ACCENT_RED};")
    layout.addWidget(value_label)
    subtitle_label = QLabel(subtitle)
    subtitle_label.setStyleSheet(f"font-size: 14px; color: {TEXT_MUTED};")
    layout.addWidget(subtitle_label)
    layout.addStretch(1)
    return frame


class HomePage(QWidget):
    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        self.banner = QFrame()
        self.banner.setObjectName("banner")
        self.banner.setFrameShape(QFrame.Shape.NoFrame)
        banner_layout = QVBoxLayout(self.banner)
        banner_layout.setContentsMargins(24, 24, 24, 24)
        self.banner_title = QLabel("Benvenuto in ACC Coach")
        self.banner_title.setObjectName("bannerTitle")
        self.banner_title.setStyleSheet("font-size: 28px; font-weight: 700;")
        banner_layout.addWidget(self.banner_title)
        self.banner_subtitle = QLabel("Configura le API key per sbloccare le funzionalita avanzate.")
        self.banner_subtitle.setStyleSheet(f"font-size: 15px; color: {TEXT_MUTED};")
        banner_layout.addWidget(self.banner_subtitle)
        self.alert_label = QLabel("")
        self.alert_label.setStyleSheet(f"font-size: 13px; color: {ACCENT_GLOW};")
        banner_layout.addWidget(self.alert_label)
        layout.addWidget(self.banner)

        cards_widget = QWidget()
        cards_layout = QGridLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(18)

        self.card_session = build_card("Ultima sessione", "-", "Nessuna sessione registrata")
        self.card_laps = build_card("Giri completati", "-", "Giri")
        self.card_best_lap = build_card("Miglior giro", "-", "Tempo")

        self.consistency_card = QFrame()
        self.consistency_card.setObjectName("card")
        self.consistency_card.setFrameShape(QFrame.Shape.NoFrame)
        cons_layout = QVBoxLayout(self.consistency_card)
        cons_layout.setContentsMargins(18, 16, 18, 16)
        cons_title = QLabel("Consistenza")
        cons_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em;")
        cons_layout.addWidget(cons_title)
        self.consistency_bar = QProgressBar()
        self.consistency_bar.setRange(0, 100)
        self.consistency_bar.setFormat("%p%")
        self.consistency_bar.setTextVisible(True)
        self.consistency_bar.setStyleSheet(
            "QProgressBar {background-color: #141414; border: 0px; height: 16px; border-radius: 8px;}"
            f"QProgressBar::chunk {{background: {ACCENT_RED}; border-radius: 8px;}}"
        )
        cons_layout.addWidget(self.consistency_bar)

        self.efficiency_card = QFrame()
        self.efficiency_card.setObjectName("card")
        self.efficiency_card.setFrameShape(QFrame.Shape.NoFrame)
        eff_layout = QVBoxLayout(self.efficiency_card)
        eff_layout.setContentsMargins(18, 16, 18, 16)
        eff_title = QLabel("Efficienza")
        eff_title.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em;")
        eff_layout.addWidget(eff_title)
        self.efficiency_bar = QProgressBar()
        self.efficiency_bar.setRange(0, 100)
        self.efficiency_bar.setFormat("%p%")
        self.efficiency_bar.setTextVisible(True)
        self.efficiency_bar.setStyleSheet(
            "QProgressBar {background-color: #141414; border: 0px; height: 16px; border-radius: 8px;}"
            f"QProgressBar::chunk {{background: {ACCENT_RED_HOVER}; border-radius: 8px;}}"
        )
        eff_layout.addWidget(self.efficiency_bar)

        cards_layout.addWidget(self.card_session, 0, 0, 1, 2)
        cards_layout.addWidget(self.card_laps, 0, 2)
        cards_layout.addWidget(self.card_best_lap, 0, 3)
        cards_layout.addWidget(self.consistency_card, 1, 0, 1, 2)
        cards_layout.addWidget(self.efficiency_card, 1, 2, 1, 2)

        layout.addWidget(cards_widget)

    def update_metrics(self, metrics: Optional[Dict[str, str]], api_ready: bool) -> None:
        if not api_ready:
            self.alert_label.setText("API key mancanti: inseriscile nelle impostazioni per attivare il coach.")
        else:
            self.alert_label.setText("")
        if not metrics:
            self._set_card(self.card_session, "-", "Nessuna sessione registrata")
            self._set_card(self.card_laps, "-", "Giri")
            self._set_card(self.card_best_lap, "-", "Tempo")
            self.consistency_bar.setValue(0)
            self.efficiency_bar.setValue(0)
            return
        session_title = f"{metrics.get('track_name', 'Sconosciuto')} - {metrics.get('car_model', '')}"
        self._set_card(self.card_session, session_title, metrics.get("started_at", ""))
        self._set_card(self.card_laps, str(metrics.get("laps", 0)), "Giri completati")
        self._set_card(self.card_best_lap, format_lap_time(metrics.get("best_lap")), "Miglior tempo")
        self.consistency_bar.setValue(int(float(metrics.get("consistency", 0.0)) * 100))
        self.efficiency_bar.setValue(int(float(metrics.get("efficiency", 0.0)) * 100))

    @staticmethod
    def _set_card(card: QFrame, value: str, subtitle: str) -> None:
        value_label = card.findChild(QLabel, "cardValue")
        if value_label:
            value_label.setText(value)
        labels = card.findChildren(QLabel)
        if len(labels) >= 3:
            labels[2].setText(subtitle)


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
        self.inputs: Dict[str, QLineEdit | QComboBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        card = QFrame()
        card.setObjectName("card")
        card.setFrameShape(QFrame.Shape.NoFrame)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)

        title = QLabel("Impostazioni e API key")
        title.setStyleSheet("font-size: 24px; font-weight: 700; letter-spacing: 0.05em;")
        card_layout.addWidget(title)

        form = QFormLayout()
        for key, label in ENV_FIELDS:
            if key == "ENABLE_TTS":
                combo = QComboBox()
                combo.addItems(["0", "1"])
                self.inputs[key] = combo
                form.addRow(label + ":", combo)
            elif key == "TTS_PROVIDER":
                combo = QComboBox()
                combo.addItems(["gtts", "elevenlabs"])
                self.inputs[key] = combo
                form.addRow(label + ":", combo)
            else:
                edit = QLineEdit()
                self.inputs[key] = edit
                form.addRow(label + ":", edit)
        card_layout.addLayout(form)

        save_btn = QPushButton("Salva configurazione")
        save_btn.setObjectName("accentButton")
        save_btn.clicked.connect(self._save)
        card_layout.addWidget(save_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addWidget(card)
        layout.addStretch(1)
        self.refresh()

    def refresh(self) -> None:
        for key, widget in self.inputs.items():
            value = self.env_manager.values.get(key, DEFAULT_VALUES.get(key, ""))
            if isinstance(widget, QComboBox):
                idx = widget.findText(value)
                widget.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                widget.setText(value)

    def _save(self) -> None:
        updates: Dict[str, str] = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QComboBox):
                updates[key] = widget.currentText()
            else:
                updates[key] = widget.text().strip()
        self.env_manager.update(updates)
        self.values_saved.emit(updates)
        QMessageBox.information(self, "Salvato", "Impostazioni aggiornate correttamente.")


class CoachPage(QWidget):
    def __init__(self, env_manager: EnvManager, controller: ServiceController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        self.controller = controller

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        card = QFrame()
        card.setObjectName("card")
        card.setFrameShape(QFrame.Shape.NoFrame)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)

        title = QLabel("Coach AI - Telemetria locale")
        title.setStyleSheet("font-size: 24px; font-weight: 700; letter-spacing: 0.05em;")
        card_layout.addWidget(title)

        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"color: {ACCENT_GLOW}; font-size: 14px;")
        card_layout.addWidget(self.info_label)

        row = QHBoxLayout()
        self.simulation_input = QLineEdit(self)
        browse_btn = QPushButton("Sfoglia simulazione")
        browse_btn.clicked.connect(self._select_simulation)
        row.addWidget(self.simulation_input)
        row.addWidget(browse_btn)
        card_layout.addLayout(row)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Avvia servizi")
        self.start_btn.setObjectName("accentButton")
        self.start_btn.clicked.connect(self._start_services)
        self.stop_btn = QPushButton("Ferma servizi")
        self.stop_btn.clicked.connect(self._stop_services)
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        card_layout.addLayout(buttons)

        self.status_label = QLabel("Servizi non avviati.")
        self.status_label.setStyleSheet(f"color: {ACCENT_GLOW}; font-size: 14px;")
        card_layout.addWidget(self.status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("I feedback del coach appariranno qui.")
        card_layout.addWidget(self.log_view, stretch=1)

        layout.addWidget(card)
        layout.addStretch(1)

        default_sim = env_manager.get_install_dir() / "data" / "simulations" / "sample_lap.jsonl"
        self.simulation_input.setText(str(default_sim))
        controller.feedback_received.connect(self._on_feedback)
        controller.status_changed.connect(self._on_status)

    def refresh(self) -> None:
        default_sim = self.env_manager.get_install_dir() / "data" / "simulations" / "sample_lap.jsonl"
        if not self.simulation_input.text().strip():
            self.simulation_input.setText(str(default_sim))

    def set_enabled(self, enabled: bool) -> None:
        self.start_btn.setEnabled(enabled)
        if not enabled:
            self.info_label.setText("Inserisci le API key nella sezione Impostazioni per attivare il coach.")
        else:
            self.info_label.setText("Seleziona un file di simulazione o collega ACC e avvia i servizi.")

    def _select_simulation(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona file simulazione",
            str(self.env_manager.get_install_dir()),
            "JSON Lines (*.jsonl);;Tutti i file (*)",
        )
        if file_path:
            self.simulation_input.setText(file_path)

    def _start_services(self) -> None:
        if not self.env_manager.has_required_api_keys():
            QMessageBox.warning(self, "API mancanti", "Inserisci le API key obbligatorie prima di avviare il coach.")
            return
        path = Path(self.simulation_input.text().strip())
        self.controller.start(path)

    def _stop_services(self) -> None:
        self.controller.stop()

    def _on_feedback(self, event: dict) -> None:
        message = event.get("message", "")
        section = event.get("section", "Generale")
        severity = event.get("severity", "info")
        self.log_view.append(f"[{severity.upper()}] {section}: {message}")

    def _on_status(self, status: str) -> None:
        self.status_label.setText(status)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ACC Coach AI - Desktop")
        self.resize(1080, 720)

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
                "download": "Download",
                "settings": "Impostazioni",
                "coach": "Coach AI",
            }
        )
        self.sidebar.selection_changed.connect(self._on_menu_selected)
        root_layout.addWidget(self.sidebar)

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
        self.download_page = DownloadPage(self.env_manager)
        self.config_page = ConfigPage(self.env_manager)
        self.coach_page = CoachPage(self.env_manager, self.service_controller)

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.download_page)
        self.stack.addWidget(self.config_page)
        self.stack.addWidget(self.coach_page)

        self.download_page.download_requested.connect(self._start_download)
        self.config_page.values_saved.connect(self._apply_updates)

        self.apply_theme()
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
        }}
        QFrame#card {{
            background-color: {CARD_BG};
            border-radius: 20px;
            border: 1px solid #1b1b1b;
        }}
        QLabel#cardValue {{
            font-size: 32px;
            font-weight: 600;
        }}
            padding-left: 12px;
        }}
        QLabel#cardValue {{
            color: {TEXT_PRIMARY};
            font-size: 32px;
            font-weight: 600;
        }}
        QPushButton {{
            background-color: {SECONDARY_BG};
            border: 1px solid transparent;
            border-radius: 12px;
            padding: 10px 16px;
            color: {TEXT_PRIMARY};
            font-size: 15px;
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
            border: none;
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
            border: 1px solid rgba(255,255,255,0.05);
            color: {TEXT_PRIMARY};
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QTextEdit {{
            background-color: {SECONDARY_BG};
            border-radius: 12px;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.08);
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
        index_map = {"home": 0, "download": 1, "settings": 2, "coach": 3}
        titles = {
            "home": "Dashboard",
            "download": "Download e aggiornamento",
            "settings": "Impostazioni",
            "coach": "Coach AI",
        }
        self.header_label.setText(titles.get(key, ""))
        if key == "home":
            self.refresh_home()
        elif key == "download":
            self.download_page.refresh()
        elif key == "settings":
            self.config_page.refresh()
        elif key == "coach":
            self.coach_page.refresh()
        self.stack.setCurrentIndex(index_map[key])

    def refresh_home(self) -> None:
        metrics = get_last_session_summary(self.env_manager)
        api_ready = self.env_manager.has_required_api_keys()
        self.home_page.update_metrics(metrics, api_ready)

    def update_availability(self) -> None:
        api_ready = self.env_manager.has_required_api_keys()
        self.coach_page.set_enabled(api_ready)

    def _apply_updates(self, updates: Dict[str, str]) -> None:
        self.env_manager.update(updates)
        self.statusBar().showMessage("Configurazione salvata", 3000)
        self.update_availability()
        self.refresh_home()

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
        else:
            QMessageBox.critical(self, "Errore download", message)
            self.statusBar().showMessage("Errore download", 3000)

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover
        self.service_controller.stop()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()








