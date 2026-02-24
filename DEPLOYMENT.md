# WaveCraft Deployment Guide

## Quick Start

### 1. Deploy Modal GPU Service
```bash
cd modal_service
modal deploy modal_app.py
```

**Your new API endpoints:**
```
https://<your-username>--wavecraft-gpu-transcribeservice-transcribe.modal.run
https://<your-username>--wavecraft-gpu-ttsservice-synthesize.modal.run
```
*(Copy the base URL from the deployment output)*

### 2. Configure Backend

**Heroku Setup (API Proxy):**
1. Create a new app in Heroku Dashboard
2. Go to Deploy tab -> Connect GitHub repository
3. Enable Automatic Deploys from `main`
4. Go to Settings -> Reveal Config Vars and add:
   - `MODAL_TRANSCRIBE_URL` = (your Modal transcribe URL)
   - `MODAL_TTS_URL` = (your Modal synthesize URL)
   - `HF_TOKEN` = (your Hugging Face token)

**Local Development (.env):**
```bash
HF_TOKEN=hf_your_token_here
# Leave empty to run inference locally, OR test remote by setting:
# MODAL_TRANSCRIBE_URL=https://<your-modal-app-url>
# MODAL_TTS_URL=https://<your-modal-app-url>
```

### 3. Test

**Run backend locally:**
```bash
python run.py
```

**Open frontend:**
```bash
cd frontend
npm start
```

## Architecture

```
Frontend (React, GitHub Pages)
       ↓
Backend API (Flask, Heroku)
       ↓
GPU Inference (Modal A10G)
```

1. **Frontend**: Hosted on GitHub pages (`wave-crafter.com`). Makes calls to Heroku API.
2. **Backend**: Lightweight API proxy on Heroku. No GPUs, no heavy ML deps.
3. **Modal**: Serverless endpoints for WhisperX transcription and Chatterbox TTS. Runs purely on A10G GPUs.

## Cost Estimate

- Modal GPU (A10G): ~$0.0008/sec active time
- No API costs for transcription/TTS (all inference happens on Modal)
- S3 Storage (optional): ~$0.001/request

For 1000 requests/month: ~$5-10/month (Modal usage only)

## Support

- Modal Dashboard: https://modal.com/apps
- Heroku Logs: `heroku logs --tail -a your-app-name`
