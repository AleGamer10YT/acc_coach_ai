
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
from typing import Callable, Dict, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from services.telemetry.collector import CollectorConfig, CollectorMode, TelemetryCollector
from services.analytics.engine import RealtimeAnalyticsEngine


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


def resource_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


class EnvManager:
    def __init__(self) -> None:
        self.default_values = DEFAULT_VALUES.copy()
        self.values = self.default_values.copy()
        self.load()

    def load(self) -> None:
        env_file = resource_path() / ".env"
        if env_file.exists():
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
        path_str = self.values.get("INSTALL_DIR", DEFAULT_VALUES["INSTALL_DIR"])
        return Path(path_str).expanduser().resolve()

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
            self.status_changed.emit("Servizi già attivi")
            return
        if not simulation_file.exists():
            self.status_changed.emit(f"File simulazione non trovato: {simulation_file}")
            return

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        future = asyncio.run_coroutine_threadsafe(
            self._async_start(simulation_file), self.loop
        )
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


class HomePage(QWidget):
    def __init__(
        self,
        go_download: Callable[[], None],
        go_config: Callable[[], None],
        go_coach: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        title = QLabel("ACC Coach AI - Launcher")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Benvenuto! Scegli un'azione per iniziare:\n"
            "1. Scarica o aggiorna i file dal repository.\n"
            "2. Configura le API key e i parametri.\n"
            "3. Avvia il Coach direttamente dall'applicazione desktop."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        buttons_layout = QHBoxLayout()
        download_btn = QPushButton("Download / Aggiornamento")
        download_btn.clicked.connect(go_download)
        config_btn = QPushButton("Configurazione API")
        config_btn.clicked.connect(go_config)
        coach_btn = QPushButton("Coach AI")
        coach_btn.clicked.connect(go_coach)
        for btn in (download_btn, config_btn, coach_btn):
            btn.setMinimumHeight(80)
            btn.setStyleSheet("font-size: 16px;")
        buttons_layout.addWidget(download_btn)
        buttons_layout.addWidget(config_btn)
        buttons_layout.addWidget(coach_btn)
        layout.addLayout(buttons_layout)

        tips = QLabel(
            "Suggerimento: puoi compilare questo launcher in un unico .exe e condividerlo.\n"
            "L'applicazione gestirà in autonomia download, configurazione e avvio del coach."
        )
        tips.setWordWrap(True)
        tips.setStyleSheet("color: #666666; margin-top: 20px;")
        layout.addWidget(tips)


class DownloadPage(QWidget):
    back_requested = Signal()
    download_requested = Signal(str, str)

    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Scarica o aggiorna il progetto da GitHub")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self.repo_input = QLineEdit(self)
        self.repo_input.setPlaceholderText("https://github.com/<user>/acc_coach_ai/archive/refs/heads/main.zip")

        self.install_input = QLineEdit(self)
        self.install_input.setReadOnly(True)
        browse_btn = QPushButton("Scegli cartella...")
        browse_btn.clicked.connect(self._select_install_dir)

        form = QFormLayout()
        form.addRow("URL pacchetto ZIP:", self.repo_input)
        install_row = QHBoxLayout()
        install_row.addWidget(self.install_input)
        install_row.addWidget(browse_btn)
        form.addRow("Cartella installazione:", install_row)
        layout.addLayout(form)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        back_btn = QPushButton("← Indietro")
        back_btn.clicked.connect(self.back_requested.emit)
        download_btn = QPushButton("Scarica / Aggiorna")
        download_btn.clicked.connect(self._handle_download)
        buttons.addWidget(back_btn)
        buttons.addStretch()
        buttons.addWidget(download_btn)
        layout.addLayout(buttons)
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
            QMessageBox.warning(self, "URL mancante", "Inserisci l'URL ZIP del repository (release o branch).")
            return
        if not target:
            QMessageBox.warning(self, "Cartella mancante", "Scegli la cartella di installazione.")
            return
        self.download_requested.emit(url, target)


class ConfigPage(QWidget):
    back_requested = Signal()
    values_saved = Signal(dict)

    def __init__(self, env_manager: EnvManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        self.inputs: Dict[str, QLineEdit | QComboBox] = {}

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        title = QLabel("Configurazione API e parametri")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        form = QFormLayout()
        for key, label in ENV_FIELDS:
            if key == "ENABLE_TTS":
                combo = QComboBox()
                combo.addItems(["0", "1"])
                form.addRow(label + ":", combo)
                self.inputs[key] = combo
            elif key == "TTS_PROVIDER":
                combo = QComboBox()
                combo.addItems(["gtts", "elevenlabs"])
                form.addRow(label + ":", combo)
                self.inputs[key] = combo
            else:
                edit = QLineEdit()
                form.addRow(label + ":", edit)
                self.inputs[key] = edit
        layout.addLayout(form)

        buttons = QHBoxLayout()
        back_btn = QPushButton("← Indietro")
        back_btn.clicked.connect(self.back_requested.emit)
        save_btn = QPushButton("Salva impostazioni")
        save_btn.clicked.connect(self._save)
        buttons.addWidget(back_btn)
        buttons.addStretch()
        buttons.addWidget(save_btn)
        layout.addLayout(buttons)
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
        QMessageBox.information(self, "Salvato", "Variabili aggiornate correttamente.")


class CoachPage(QWidget):
    back_requested = Signal()

    def __init__(self, env_manager: EnvManager, controller: ServiceController, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.env_manager = env_manager
        self.controller = controller

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Coach AI - Telemetria in tempo reale")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self.simulation_input = QLineEdit(self)
        browse_btn = QPushButton("Scegli file simulazione...")
        browse_btn.clicked.connect(self._select_simulation)

        row = QHBoxLayout()
        row.addWidget(self.simulation_input)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        back_btn = QPushButton("← Indietro")
        back_btn.clicked.connect(self.back_requested.emit)
        start_btn = QPushButton("Avvia servizi")
        start_btn.clicked.connect(self._start_services)
        stop_btn = QPushButton("Ferma servizi")
        stop_btn.clicked.connect(self._stop_services)
        buttons.addWidget(back_btn)
        buttons.addStretch()
        buttons.addWidget(start_btn)
        buttons.addWidget(stop_btn)
        layout.addLayout(buttons)

        self.status_label = QLabel("Servizi non avviati.")
        layout.addWidget(self.status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Feedback del coach AI appariranno qui...")
        layout.addWidget(self.log_view, stretch=1)

        default_sim = env_manager.get_install_dir() / "data" / "simulations" / "sample_lap.jsonl"
        self.simulation_input.setText(str(default_sim))
        controller.feedback_received.connect(self._on_feedback)
        controller.status_changed.connect(self._on_status)

    def refresh(self) -> None:
        default_sim = self.env_manager.get_install_dir() / "data" / "simulations" / "sample_lap.jsonl"
        if not self.simulation_input.text().strip():
            self.simulation_input.setText(str(default_sim))

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
        self.resize(960, 640)

        self.env_manager = EnvManager()
        self.service_controller = ServiceController()
        self._workers: list[DownloadWorker] = []

        self.stack = QStackedWidget()
        self.home_page = HomePage(
            go_download=lambda: self._go_page(1),
            go_config=lambda: self._go_page(2),
            go_coach=lambda: self._go_page(3),
        )
        self.download_page = DownloadPage(self.env_manager)
        self.config_page = ConfigPage(self.env_manager)
        self.coach_page = CoachPage(self.env_manager, self.service_controller)

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.download_page)
        self.stack.addWidget(self.config_page)
        self.stack.addWidget(self.coach_page)

        self.setCentralWidget(self.stack)

        self.download_page.back_requested.connect(lambda: self._go_page(0))
        self.download_page.download_requested.connect(self._start_download)
        self.config_page.back_requested.connect(lambda: self._go_page(0))
        self.config_page.values_saved.connect(self._apply_updates)
        self.coach_page.back_requested.connect(lambda: self._go_page(0))

        self.statusBar().showMessage("Pronto")

    def _go_page(self, index: int) -> None:
        if index == 1:
            self.download_page.refresh()
        elif index == 2:
            self.config_page.refresh()
        elif index == 3:
            self.coach_page.refresh()
        self.stack.setCurrentIndex(index)

    def _apply_updates(self, updates: Dict[str, str]) -> None:
        self.env_manager.update(updates)
        self.statusBar().showMessage("Configurazione salvata", 3000)

    def _start_download(self, url: str, target: str) -> None:
        target_dir = Path(target).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        worker = DownloadWorker(url, target_dir)
        self._workers.append(worker)
        worker.finished.connect(self._on_download_finished)
        worker.start()
        self.statusBar().showMessage("Download in corso...")

    def _on_download_finished(self, success: bool, message: str) -> None:  # pragma: no cover
        sender = self.sender()
        if sender in self._workers:
            self._workers.remove(sender)  # type: ignore[arg-type]
        if success:
            self.env_manager.save()
            QMessageBox.information(self, "Completato", message)
            self.statusBar().showMessage("Download completato", 3000)
        else:
            QMessageBox.critical(self, "Errore download", message)
            self.statusBar().showMessage("Errore download", 3000)

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover
        self.service_controller.stop()
        super().closeEvent(event)


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


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
