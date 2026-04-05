# WaveCrafter

WaveCrafter is a web application for editing speech audio via speaker-wise transcripts. Upload multi-speaker audio, get diarized transcripts, edit the text, and generate new audio preserving each speaker's voice through zero-shot voice cloning.

## Features

- Upload multi-speaker audio files (podcasts, interviews, conversations)
- Automatic speaker diarization and word-level transcription (WhisperX)
- Speaker-wise transcript editing — click to delete words, double-click to insert
- Emotion tagging and paralinguistic sounds (`[laugh]`, `[sigh]`, `[gasp]`)
- Drag-to-reorder conversation segments
- Text-to-speech voice cloning maintaining original speaker characteristics (Chatterbox)
- **AI Speaker generation** — generate dialogue via GPT and synthesize with TTS
- Real-time waveform visualization per segment
- Responsive web design with dark/light theme support

## Architecture

Three-tier cloud-native setup:

- **Frontend (React)** — deployed to GitHub Pages
- **Backend (Flask)** — deployed to Heroku (CPU-only orchestration layer)
- **GPU Service (Modal)** — WhisperX transcription + Chatterbox TTS on A10G GPUs

The backend never runs ML models directly. It proxies to Modal endpoints for all inference. Audio is passed as base64-encoded WAV throughout.

### Agent System (LangGraph)

The backend uses LangGraph `StateGraph` for workflow orchestration:

| Graph | Workflow |
|---|---|
| `TranscriptionGraph` | Optimize audio → WhisperX transcribe + diarize → segments |
| `ModificationGraph` | Parse changes → parallel voice cloning → assemble → quality check → S3 |
| `ArtificialSpeakerGraph` | GPT dialogue generation → TTS engine selection → synthesize |

Agents live in `backend/langgraph_agents/agents/`:
- `WhisperXAgent` — remote (Modal) or local transcription
- `ChatterboxAgent` — remote (Modal) or local voice cloning TTS
- `DialogueGeneratorAgent` — OpenAI GPT dialogue generation
- `TextToSpeechAgent` — OpenAI TTS fast fallback

## Tech Stack

### Frontend
- React 18
- Axios (API communication)
- React Router (navigation)
- dnd-kit (drag-to-reorder)
- wavesurfer.js (waveform visualization)
- CSS3 for styling

### Backend
- Flask (Python web framework)
- LangGraph (agent workflow orchestration)
- WhisperX (speech-to-text + diarization)
- Chatterbox TTS (zero-shot voice cloning)
- Pydub (audio processing)
- OpenAI API (dialogue generation + TTS fallback)

### Deployment
- Frontend: GitHub Pages (via GitHub Actions)
- Backend: Heroku
- GPU Service: Modal Cloud (serverless A10G)
- CI/CD: GitHub Actions

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.11.5)
- npm
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wave-crafter.git
cd wave-crafter
```

2. Set up the frontend:
```bash
cd frontend
npm install
```

3. Set up the backend:
```bash
cd ../backend
pip install -r requirements.txt
```

## Running Locally

1. Start the backend server:
```bash
cd backend
python -m flask --app backend run
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Environment Variables

### Frontend
Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=http://localhost:5000
```

### Backend
Create a `.env` file in the backend directory:
```
MODAL_TRANSCRIBE_URL=<Modal transcribe endpoint URL>
MODAL_TTS_URL=<Modal synthesize endpoint URL>
OPENAI_API_KEY=<OpenAI API key for dialogue generation and TTS fallback>
HF_TOKEN=<HuggingFace token for pyannote diarization (local dev only)>
AWS_ACCESS_KEY_ID=<optional>
AWS_SECRET_ACCESS_KEY=<optional>
AWS_S3_BUCKET=<optional>
S3_ENABLED=false
```

## Project Structure

```
wave-crafter/
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── styles/           # CSS files
│   │   └── App.js
│   ├── package.json
│   └── build/
├── backend/
│   ├── langgraph_agents/     # LangGraph workflow agents
│   │   ├── agents/           # WhisperX, Chatterbox, Dialogue, TTS
│   │   ├── state.py          # TypedDict state schemas
│   │   ├── nodes.py          # Graph node implementations
│   │   └── graphs.py         # Graph builders
│   ├── utils/                # Audio quality, S3 storage
│   ├── uploads/              # Temporary upload directory
│   ├── routes.py             # Flask API routes
│   ├── __init__.py           # Flask app factory
│   └── requirements.txt
├── modal_service/
│   └── modal_app.py          # Modal GPU service (WhisperX + Chatterbox)
├── .github/
│   └── workflows/            # CI/CD for frontend, backend, Modal
├── requirements-gpu.txt      # Local GPU dev dependencies
└── README.md
```

## Deployment

### Frontend (GitHub Pages)

Auto-deploys on push to `main` when `frontend/**` changes. See `.github/workflows/frontend_deploy.yml`.

### Backend (Heroku)

Auto-deploys via Heroku GitHub integration. Runtime: Python 3.11.5. Gunicorn command:
```bash
gunicorn "backend:create_app()" --timeout 120 --workers 1 --threads 2 --worker-class gthread
```

### Modal GPU Service

Auto-deploys on push when `modal_service/**` changes. See `.github/workflows/modal_deploy.yml`.
Requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` GitHub secrets.

## License

This project is licensed under the MIT License.
