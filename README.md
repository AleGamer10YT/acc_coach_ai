# ACC Coach AI

Coach virtuale intelligente per Assetto Corsa Competizione con analisi telemetrica in tempo reale, feedback live e reportistica post-sessione potenziata da AI.

## Monorepo
```
services/
  telemetry/      # Ingestione telemetria ACC e simulatore
  analytics/      # Stream processor, API realtime, persistenza
  coach_ai/       # Suggerimenti naturali via LLM/TTS
shared/           # Schemi Pydantic, utilita comuni
infrastructure/   # Docker Compose, provisioning locali
data/             # Dataset e playback
scripts/          # Utility CLI (es. replay telemetria)
launcher/         # Applicazione desktop (PySide6) per setup e coaching
```

Per dettagli architetturali vedi il documento `Base`.

## Setup rapido
1. Copia `.env.example` in `.env` e personalizza credenziali (es. `OPENAI_API_KEY`).
2. Avvia lo stack completo con Docker Compose **oppure** usa l'app desktop (`python launcher/app.py`):
   ```bash
   cd infrastructure
   docker compose up --build
   ```
   Servizi esposti:
   - Telemetry API -> `http://localhost:8081`
   - Analytics + WebSocket -> `http://localhost:8080`
   - Coach AI -> `http://localhost:8082`
   - Dashboard Streamlit -> `http://localhost:8501`
   - Overlay statico -> `http://localhost:8090`
3. (Facoltativo) riproduci la telemetria demo:
   ```bash
   python scripts/playback_simulation.py --loop
   ```

## Servizi backend
- **Telemetry Collector** (`services/telemetry`): ascolta ACC (UDP/shared memory) o file simulati e pubblica frame normalizzati su Redis o coda in-memory.
- **Analytics API** (`services/analytics`): calcola KPI in streaming, salva sessioni su SQLite/PostgreSQL e diffonde `FeedbackEvent` per il coach.
- **Coach AI** (`services/coach_ai`): coordina modelli ML/LLM e TTS, con fallback rule-based se manca la chiave API.

Avvio stand-alone:
```bash
pip install -e .
uvicorn services.<nome>.main:app --reload
```
Ricorda di impostare `PYTHONPATH=.` se lanci senza installazione.

## Interfaccia desktop
- **ACC Coach Desktop** (`launcher/app.py`): applicazione PySide6 con homepage, gestione download dal repository, configurazione API e sezione Coach AI per avviare i servizi locali senza usare pagine web.
- Legacy: le directory `frontend/overlay` e `frontend/dashboard` contengono gli asset web storici.

## Script utili
- `scripts/playback_simulation.py`: replay di file JSONL (`data/simulations/sample_lap.jsonl`) verso il bus telemetria.
- `data/simulations/sample_lap.jsonl`: giro demo Monza GT3 per test rapidi.

## Launcher Windows
- Lancia `python launcher/app.py` per usare l'app desktop.
- Per creare un `.exe` one-file: `powershell -ExecutionPolicy Bypass -File tools/build_launcher.ps1`. Consulta `SETUP_GUIDE.md` per la distribuzione automatica (download GitHub, configurazione e avvio servizi).

## Flusso di test rapido
1. Avvia l'app desktop (`python launcher/app.py` o l'eseguibile compilato).
2. Dalla home scegli "Download/Aggiornamento" per scaricare la release da GitHub (se necessario).
3. Configura le API key nella sezione dedicata.
4. Apri il menu "Coach AI", seleziona il file di telemetria (es. `data/simulations/sample_lap.jsonl`) e premi "Avvia servizi". I feedback appaiono direttamente nell'interfaccia.
