# WaveCraft Deployment Guide

## Quick Start

### 1. Deploy Modal Service
```bash
cd modal_service
modal deploy voicecraft_text_modification.py
```

**Your API endpoint:**
```
https://mukund0911--wavecraft-voicecraft-textmod-web.modal.run
```

### 2. Configure Backend

**Heroku:**
```bash
heroku config:set MODAL_VOICECRAFT_URL="https://mukund0911--wavecraft-voicecraft-textmod-web.modal.run" -a your-app-name
```

**Local (.env):**
```bash
MODAL_VOICECRAFT_URL=https://mukund0911--wavecraft-voicecraft-textmod-web.modal.run
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_S3_BUCKET=wavecraft-audio
AWS_REGION=us-east-1
ASSEMBLY_AI_KEY=your_key
```

### 3. Test

**Health check:**
```bash
curl https://mukund0911--wavecraft-voicecraft-textmod-web.modal.run/health
```

**Run backend:**
```bash
python backend/run.py
```

**Open frontend:**
```
http://localhost:3000
```

## Features

✅ Voice cloning with text modification
✅ Multi-speaker support
✅ S3 storage for audio files
✅ Real-time transcription
✅ Quality: 85-90% voice similarity
✅ Latency: 8-12s per segment

## Architecture

```
Frontend (React) → Backend (Flask) → Modal (VoiceCraft GPU) → S3 (Storage)
                                   ↓
                            AssemblyAI (Transcription)
```

## Cost Estimate

- Modal GPU: ~$0.004/request
- S3 Storage: ~$0.001/request
- AssemblyAI: ~$0.04/request
- **Total: ~$0.045 per voice cloning request**

For 1000 requests/month: ~$45/month

## Support

- Modal Dashboard: https://modal.com/apps
- Issues: Check logs with `modal app logs wavecraft-voicecraft-textmod`
