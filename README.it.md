# ACC Coach AI (Italiano)

ACC Coach AI è un assistente da scrivania per Assetto Corsa Competizione. Raccoglie la telemetria, analizza lo stile di guida e fornisce consigli in tempo reale all’interno di un’unica applicazione PySide6, senza aprire browser o pagine local host.

## Caratteristiche principali

- **Interfaccia OLED stile dashboard** – sidebar con le sezioni Dashboard, Download, Impostazioni e Coach, grafica nero/rosso/giallo.
- **Bootstrap automatico delle dipendenze** – se avvii `python launcher/app.py`, lo script installa in autonomia i pacchetti Python mancanti (PySide6, FastAPI, ecc.).
- **Coach self-service** – scarica/aggiorna i file dal tuo repository GitHub, salva la API key di Google AI Studio e avvia la simulazione o collegati ad ACC per il coaching live.
- **Dashboard dati** – card e barre mostrano l’ultima sessione (tracciato, vettura, giri, consistenza, efficienza, miglior giro).

## Prerequisiti

- Windows 10/11 (testato)
- Python 3.11+ (solo per esecuzione da sorgente o compilazione)
- Docker Desktop (opzionale, per usare l’intera stack a microservizi)

## Avvio da sorgente

```bash
git clone https://github.com/<tuo-account>/acc_coach_ai.git
cd acc_coach_ai
python launcher/app.py
```

Al primo avvio il launcher installa automaticamente le dipendenze e apre la finestra principale. Usa la sidebar per scaricare gli asset, salvare le API key e passare alla sezione Coach.

## Creazione dell’eseguibile Windows

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_launcher.ps1 -Clean
# Risultato: dist\ACCCoachLauncher.exe
```

L’eseguibile contiene già tutte le librerie: è sufficiente distribuirlo e farlo partire con un doppio click.

## Flusso consigliato

1. **Download** – nella sezione omonima inserisci l’URL ZIP del tuo repository GitHub (branch o release). Il launcher scarica/aggiorna i file nella cartella indicata.
2. **Impostazioni** - compila le API key obbligatorie (es. `GOOGLE_API_KEY` di Google AI Studio) e salva. Finche le chiavi non sono presenti, la sezione Coach resta disabilitata.
3. **Coach** – scegli un file di telemetria (es. `data/simulations/sample_lap.jsonl`) oppure collega ACC. I suggerimenti compaiono direttamente nel log della pagina.
4. **Dashboard** – la home riepiloga l’ultima sessione: giri, consistenza, efficienza, miglior tempo.

## Struttura della repo

```
services/      servizi backend (telemetria, analytics, coach AI)
shared/        schemi e utilità condivisi
launcher/      applicazione desktop PySide6
infrastructure docker-compose e asset infrastrutturali
data/          dati di telemetria di esempio
scripts/       script di supporto (es. playback telemetria)
```

Per dettagli storici sull’overlay web o sull’infrastruttura, consulta le directory legacy (`frontend/overlay`, `frontend/dashboard`, `infrastructure`).

---

Buon coaching virtuale! Per richieste o suggerimenti, apri una issue o invia una pull request.
