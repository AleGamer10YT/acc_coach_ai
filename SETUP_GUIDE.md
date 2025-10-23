# ACC Coach AI - Guida di Avvio

Questa guida spiega come preparare l'ambiente, configurare le variabili e avviare tutti i servizi del coach virtuale.

---

## 1. Prerequisiti

- **Docker Desktop** (consigliato per lanciare l'intera stack rapidamente).
- **Python 3.11+** (opzionale, se vuoi avviare i servizi manualmente o compilare il launcher).
- Redis e PostgreSQL sono gia inclusi nel `docker-compose.yml`.

---

## 2. Variabili d'ambiente principali

| Variabile            | Valore da impostare                                         | Dove ottenerla / istruzioni                                                                                          |
|----------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `APP_ENV`            | `development` oppure `production`                           | Imposta `production` per deploy live.                                                                                  |
| `LOG_LEVEL`          | `INFO` / `DEBUG`                                            | Usa `DEBUG` per logging dettagliato.                                                                                   |
| `REDIS_URL`          | `redis://localhost:6379/0`                                  | Mantieni il default con Docker oppure punta al tuo Redis esterno.                                                     |
| `DATABASE_URL`       | `sqlite+aiosqlite:///./data/app.db`                         | Per PostgreSQL usa `postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}`.                           |
| `COACH_API_URL`      | `http://localhost:8082`                                     | Endpoint del servizio Coach AI (per riferimenti legacy o API esterne).                                                |
| `OVERLAY_WS_URL`     | `ws://localhost:8080/ws/feedback`                           | Canale WebSocket legacy (overlay web).                                                                                |
| `DASHBOARD_URL`      | `http://localhost:8501`                                     | URL della dashboard Streamlit (legacy).                                                                               |
| `OPENAI_API_KEY`     | Inserisci qui la tua API key: `{OPENAI_API_KEY}`            | Ottienila da [platform.openai.com](https://platform.openai.com) ? Dashboard API Keys.                                  |
| `OPENAI_MODEL`       | Inserisci qui il modello preferito: `{OPENAI_MODEL}`        | Usa un modello supportato (es. `gpt-4o-mini`).                                                                         |
| `ENABLE_TTS`         | `0` oppure `1`                                              | Imposta `1` per attivare la voce.                                                                                      |
| `TTS_PROVIDER`       | Inserisci qui il provider TTS: `{TTS_PROVIDER}`             | Default `gtts`; per ElevenLabs usa l'identificativo relativo.                                                          |
| `ELEVENLABS_API_KEY` | Inserisci qui la tua API key: `{ELEVENLABS_API_KEY}`        | Recuperala da [https://elevenlabs.io](https://elevenlabs.io) se utilizzi quel provider.                                |
| `TTS_OUTPUT_DIR`     | Inserisci qui la cartella output audio: `{TTS_OUTPUT_DIR}`  | Esempio: `data/audio`. Assicurati che esista o venga creata.                                                           |
| `REPO_ZIP_URL`       | Inserisci qui l'URL ZIP del progetto: `{REPO_ZIP_URL}`      | Link diretto alla release/branch su GitHub che verra scaricata dal launcher.                                          |
| `INSTALL_DIR`        | Inserisci qui la cartella di installazione: `{INSTALL_DIR}` | Destinazione locale dei file scaricati (default: `~/ACC_Coach_AI`).                                                   |

> Suggerimento: usa `.env.example` come riferimento. Copialo in `.env` e aggiorna i valori.

---

## 3. Avvio con Docker Compose

1. Assicurati che Docker Desktop sia in esecuzione.
2. Copia il file `.env` di esempio e personalizza i valori:
   ```bash
   cp .env.example .env
   # modifica .env con le tue chiavi/API
   ```
3. Avvia la stack:
   ```bash
   cd infrastructure
   docker compose up --build
   ```
4. Servizi disponibili localmente:
   - Telemetry API ? `http://localhost:8081`
   - Analytics API + WebSocket ? `http://localhost:8080`
   - Coach AI ? `http://localhost:8082`
   - Dashboard Streamlit ? `http://localhost:8501`
   - Overlay statico ? `http://localhost:8090`
   - Redis ? `localhost:6379`
   - PostgreSQL ? `localhost:5432` (user/pass `acc/acc`)
5. Arresta con `CTRL+C` o `docker compose down`.

---

## 4. Avvio manuale (senza Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install -e .
```

Avvia i servizi in terminali separati:

```bash
# Telemetry Collector (porta 8081)
pip install -r services/telemetry/requirements.txt
uvicorn services.telemetry.main:app --port 8081 --reload

# Analytics + WebSocket (porta 8080)
pip install -r services/analytics/requirements.txt
uvicorn services.analytics.main:app --port 8080 --reload

# Coach AI (porta 8082)
pip install -r services/coach_ai/requirements.txt
uvicorn services.coach_ai.main:app --port 8082 --reload

# Dashboard Streamlit (porta 8501)
pip install -r frontend/dashboard/requirements.txt
streamlit run frontend/dashboard/streamlit_app.py

# Overlay statico (opzionale)
python -m http.server 8090 --directory frontend/overlay
```

Se scegli questa modalita ricorda di fornire Redis/PostgreSQL o aggiornare `DATABASE_URL`/`REDIS_URL` di conseguenza.

---

## 5. Telemetria di test

- Dataset demo: `data/simulations/sample_lap.jsonl`
- Replay:
  ```bash
  python scripts/playback_simulation.py --loop
  ```
  Pubblica i frame sul bus telemetria (in-memory o Redis a seconda della configurazione).

---

## 6. Flusso di verifica

1. Avvia i servizi (Docker o manuale).
2. Lancia il playback demo.
3. Apri il menu Coach dell'app desktop per visualizzare feedback realtime.
4. Analizza i dati registrati direttamente dall'interfaccia.

---

## 7. Risorse

- Documento architetturale: `Base`
- Docker Compose: `infrastructure/docker-compose.yml`
- Script e simulazioni: `scripts/` e `data/`

---

## 8. Compilare e distribuire il launcher

Il launcher (`launcher/app.py`) gestisce download del progetto, configurazione `.env` e avvio del coach in locale.

### 8.1 Requisiti
- Python 3.11+
- PyInstaller (`pip install pyinstaller`)

### 8.2 Compilazione
```powershell
powershell -ExecutionPolicy Bypass -File tools/build_launcher.ps1
```
Opzioni utili:
- `-OutputName <NomeExe>` per personalizzare il nome dell'eseguibile.
- `-Clean` per rimuovere le cartelle `build/` e `dist/` prima della compilazione.

### 8.3 Distribuzione
1. Pubblica il pacchetto ZIP (branch o release) su GitHub e aggiorna `REPO_ZIP_URL`.
2. Condividi l'eseguibile generato (`dist/<NomeExe>.exe`).
3. Al primo avvio l'app:
   - chiede URL e cartella installazione;
   - scarica/aggiorna i file del progetto;
   - salva `.env` sia accanto all'eseguibile sia nella cartella di installazione;
   - consente di avviare immediatamente il coach senza passare dal browser.

> Nota: Docker Desktop deve essere installato e in esecuzione sul PC di destinazione. L'app non installa automaticamente Docker o altri prerequisiti di sistema.

---

Con questi passaggi puoi distribuire facilmente il coach e garantire un'esperienza completamente desktop.
