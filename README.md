# ACC Coach AI

ACC Coach AI is a desktop assistant for Assetto Corsa Competizione that ingests telemetry, analyses driving patterns, and delivers coaching tips in real time. Everything runs locally inside a single PySide6 application—no browser windows or localhost tabs required.

## Highlights

- **One-click desktop experience** – a dark OLED‑style UI with sidebar navigation (Dashboard, Download, Settings, Coach).
- **Automatic dependency bootstrap** – when you launch `python launcher/app.py`, any missing Python packages (PySide6, FastAPI, etc.) are installed automatically.
- **Self-service coach** – download or update the project files from GitHub, enter your Google AI Studio API key once, and start the telemetry simulator or connect ACC for live feedback.
- **Data-driven dashboard** – cards and progress bars summarise the most recent session (track, car, laps, consistency, efficiency, best lap).

## Getting Started

### Requirements

- Windows 10/11 (tested)  
- Python 3.11+ (only if you run from source or build the executable)  
- Docker Desktop (optional, for a full microservice stack or analytics via containers)

### Run from Source

```bash
git clone https://github.com/<your-account>/acc_coach_ai.git
cd acc_coach_ai
python launcher/app.py
```

The first launch triggers the automatic installation of all required Python packages. The app opens with the Dashboard view; use the sidebar to download assets, configure API keys, and start the coach.

### Build the Windows Executable

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_launcher.ps1 -Clean
# Output: dist\ACCCoachLauncher.exe
```

Share the generated `.exe` directly—PyInstaller bundles PySide6 and every runtime dependency, so the end user only needs to double-click the executable.

## Typical Workflow

1. **Download** – open the “Download” section and provide the GitHub ZIP URL (branch or release). The launcher will fetch or update the local install directory.
2. **Settings** – enter the required API keys (e.g. `GOOGLE_API_KEY` from Google AI Studio) and save. Until the keys are present, the Coach section remains disabled to avoid accidental runs.
3. **Coach** – choose a telemetry source:
   - play back the sample session stored in `data/simulations/sample_lap.jsonl`, or  
   - connect to live ACC telemetry.
   Real-time feedback appears directly in the Coach page log.
4. **Dashboard** – monitor the latest session metrics (laps, best lap, consistency, efficiency) on the home page.

## Repository Layout

```
services/      backend services (telemetry collector, analytics, coach AI)
shared/        shared schemas and utilities
launcher/      PySide6 desktop application
infrastructure docker-compose and infrastructure assets
data/          sample telemetry data
scripts/       helper scripts (e.g. telemetry playback)
```

## Need Help in Italian?

A full Italian version of this README is available in **README.it.md**, covering the same topics (overview, usage, and build steps) per gli utenti preferibilmente in lingua italiana.

---

Happy racing! If you have ideas for new coaching features or UI tweaks, feel free to open an issue or submit a pull request.
