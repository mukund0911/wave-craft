# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WaveCraft is a web app for editing speech audio via speaker-wise transcripts. Users upload multi-speaker audio, get diarized transcripts, edit the text, and generate new audio preserving each speaker's voice (via voice cloning).

## Architecture

Three-tier cloud-native setup:

- **Frontend (React)** → deployed to GitHub Pages via GitHub Actions
- **Backend (Flask)** → deployed to Heroku (CPU-only, orchestration layer)
- **GPU Service (Modal)** → WhisperX transcription + Chatterbox TTS on A10G GPUs

The backend never runs ML models directly. It proxies to Modal endpoints for all inference. Audio is passed as base64-encoded WAV throughout.

### Data Flow

1. User uploads audio → Flask saves to `backend/uploads/`, starts background thread
2. Backend sends base64 audio to Modal's `/transcribe` endpoint (WhisperX + diarization)
3. Modal returns conversations with word-level timestamps + per-segment audio
4. User edits transcript in frontend
5. Frontend sends edits to `/conversations_modified` → backend calls Modal's `/synthesize` (Chatterbox TTS with voice cloning) for changed segments only
6. Segments assembled with crossfading → returned (optionally uploaded to S3)

### Agent Pattern

Backend uses an MCP-style agent pattern in `backend/mcp_agents/`:

- `base_agent.py` — abstract base with request validation and response formatting
- `speech_processing_agent.py` — main orchestrator, calls the two agents below
- `whisperx_agent.py` — remote (Modal) or local transcription; dual-mode via `MODAL_TRANSCRIBE_URL`
- `chatterbox_agent.py` — remote (Modal) or local TTS; dual-mode via `MODAL_TTS_URL`

If Modal env vars are unset, agents fall back to local inference (requires GPU dependencies).

### Modal Service (`modal_service/modal_app.py`)

Two separate Docker images to avoid dependency conflicts:
- `whisperx_image`: torch 2.5.1, numpy<2.0, whisperx, pyannote.audio, pydub
- `chatterbox_image`: torch 2.5.1, numpy<1.26, chatterbox-tts

Models are cached in a persistent Modal volume (`wavecraft-models`). Container startup loads models via `@modal.enter()`. Includes a `torch.load` monkey-patch for PyTorch 2.6 `weights_only` compatibility.

## Common Commands

### Frontend
```bash
cd frontend && npm install    # install dependencies
cd frontend && npm start      # dev server on :3000
cd frontend && npm run build  # production build
```

### Backend
```bash
cd backend && pip install -r requirements.txt   # install (CPU-only deps)
cd backend && python -m flask --app backend run  # dev server
# Production (Heroku): gunicorn "backend:create_app()" --timeout 120 --workers 1 --threads 2 --worker-class gthread
```

### Modal GPU Service
```bash
modal serve modal_service/modal_app.py    # local dev (hot-reload)
modal deploy modal_service/modal_app.py   # deploy to Modal cloud
```

## Environment Variables

### Backend (.env)
```
MODAL_TRANSCRIBE_URL=<Modal transcribe endpoint URL>
MODAL_TTS_URL=<Modal synthesize endpoint URL>
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_S3_BUCKET / AWS_REGION  # optional S3
S3_ENABLED=true/false
```

### Modal (set as `wavecraft-secrets` in Modal dashboard)
```
HF_TOKEN=<HuggingFace token for pyannote diarization>
```

### Frontend (.env)
```
REACT_APP_API_URL=<backend URL, e.g. https://wave-crafter-587074aad3d2.herokuapp.com>
```

## Deployment

- **Frontend**: Auto-deploys to GitHub Pages on push to `main` when `frontend/**` changes (`.github/workflows/frontend_deploy.yml`). Custom domain: `wave-crafter.com`
- **Backend**: Heroku GitHub integration (auto-deploy from `main`). Runtime: Python 3.11.5
- **Modal**: Auto-deploys on push when `modal_service/**` changes (`.github/workflows/modal_deploy.yml`). Requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` GitHub secrets

## Key API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload audio file, returns `job_id` for polling |
| `/status/<job_id>` | GET | Poll transcription progress |
| `/conversations_modified` | POST | Send edited transcripts, get regenerated audio |
| `/health` | GET | Health check |

## Known Issues & Gotchas

- WhisperX and Chatterbox have **conflicting numpy requirements** — this is why Modal uses two separate images
- PyTorch 2.6 changed `torch.load` defaults (`weights_only=True`) — there's a monkey-patch in `modal_app.py:93-97`
- Heroku dynos sleep after 30min inactivity; first request after sleep returns 503 with CORS headers
- The `backend/requirements.txt` is intentionally lightweight (no torch/whisperx) — GPU deps are in `requirements-gpu.txt` for local dev only
- Audio cache and job store are in-memory; they reset on Heroku dyno restart
