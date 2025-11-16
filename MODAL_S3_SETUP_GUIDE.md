# WaveCraft Production Setup Guide
## Modal.com + AWS S3 Integration

**Complete step-by-step setup for production voice cloning infrastructure**

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Modal Setup](#modal-setup)
3. [AWS S3 Setup](#aws-s3-setup)
4. [Backend Configuration](#backend-configuration)
5. [Deployment](#deployment)
6. [Testing](#testing)
7. [Monitoring](#monitoring)
8. [Cost Management](#cost-management)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- [x] Modal.com account (free tier available)
- [x] AWS account (free tier for S3)
- [x] Python 3.9+ installed

### Required Tools
```bash
# Install Modal CLI
pip install modal

# Install AWS CLI (optional but recommended)
pip install awscli

# Install boto3 for S3
pip install boto3
```

---

## Modal Setup

### Step 1: Create Modal Account

1. Go to https://modal.com
2. Sign up with GitHub (recommended) or email
3. Verify your email

### Step 2: Install and Authenticate Modal

```bash
# Install Modal
pip install modal

# Authenticate (opens browser)
modal token new

# Verify authentication
modal token list
```

You should see:
```
✓ Successfully authenticated with Modal
```

### Step 3: Deploy VoiceCraft Service

```bash
# Navigate to project
cd /path/to/wave-craft

# Deploy Modal service
modal deploy modal_service/voicecraft_service.py
```

**Expected output:**
```
✓ Initialized. View run at https://modal.com/...
✓ Created web function voicecraft-voice-cloning/web
🎉 Deployed app voicecraft-voice-cloning

View endpoints at: https://your-workspace--voicecraft-voice-cloning-web.modal.run
```

### Step 4: Get Modal Endpoint URL

After deployment, Modal will provide a URL like:
```
https://your-workspace--voicecraft-voice-cloning-web.modal.run
```

**Save this URL** - you'll need it for backend configuration!

### Step 5: Test Modal Endpoint

```bash
# Test health endpoint
curl https://your-workspace--voicecraft-voice-cloning-web.modal.run/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "voicecraft-voice-cloning",
  "version": "1.0.0"
}
```

### Step 6: Configure Modal Settings (Optional)

Edit `modal_service/voicecraft_service.py` to adjust:

```python
# GPU type (cost vs speed tradeoff)
gpu="A10G"  # Recommended: $0.0006/s
# gpu="T4"  # Cheaper: $0.0003/s (slower)
# gpu="A100" # Faster: $0.0014/s (expensive)

# Container idle timeout
container_idle_timeout=300  # Keep warm for 5 minutes

# Concurrent requests per container
allow_concurrent_inputs=10  # Handle 10 parallel requests
```

**Recommendation:** Start with A10G, adjust based on performance needs.

---

## AWS S3 Setup

### Step 1: Create AWS Account

1. Go to https://aws.amazon.com
2. Create account (requires credit card, but free tier available)
3. Complete verification

### Step 2: Create S3 Bucket

#### Option A: AWS Console (Easiest)

1. Go to AWS Console → S3
2. Click "Create bucket"
3. Settings:
   - **Bucket name:** `wavecraft-audio` (must be globally unique)
   - **Region:** `us-east-1` (or closest to your users)
   - **Block all public access:** ✅ Enabled (we'll use presigned URLs)
   - **Versioning:** Disabled
   - **Encryption:** AES-256 (default)
4. Click "Create bucket"

#### Option B: AWS CLI (Advanced)

```bash
# Create bucket
aws s3 mb s3://wavecraft-audio --region us-east-1

# Configure CORS (if needed for direct browser uploads)
aws s3api put-bucket-cors --bucket wavecraft-audio --cors-configuration file://cors.json
```

**cors.json:**
```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "MaxAgeSeconds": 3000
    }
  ]
}
```

### Step 3: Create IAM User for S3 Access

1. Go to AWS Console → IAM → Users
2. Click "Add users"
3. User details:
   - **Username:** `wavecraft-s3-user`
   - **Access type:** Programmatic access ✅
4. Permissions:
   - Attach policy: **AmazonS3FullAccess** (or create custom policy below)
5. Review and create
6. **IMPORTANT:** Download credentials (Access Key ID + Secret Access Key)
   - You won't be able to see the secret key again!

#### Custom S3 Policy (More Secure)

Instead of `AmazonS3FullAccess`, use this restrictive policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::wavecraft-audio",
        "arn:aws:s3:::wavecraft-audio/*"
      ]
    }
  ]
}
```

### Step 4: Configure S3 Lifecycle Policy (Optional - Cost Savings)

Auto-delete old audio files after 30 days to save storage costs:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket wavecraft-audio \
  --lifecycle-configuration '{
    "Rules": [{
      "Id": "auto-delete-old-audio",
      "Status": "Enabled",
      "Prefix": "audio/",
      "Expiration": {
        "Days": 30
      }
    }]
  }'
```

**Cost savings:** ~87% reduction (files deleted after 30 days instead of stored forever)

---

## Backend Configuration

### Step 1: Update Environment Variables

Create or update `.env` file in `backend/` directory:

```bash
# backend/.env

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================
MODAL_VOICECRAFT_URL=https://your-workspace--voicecraft-voice-cloning-web.modal.run

# ============================================================================
# AWS S3 CONFIGURATION
# ============================================================================
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_S3_BUCKET=wavecraft-audio
AWS_REGION=us-east-1

# Optional: Disable S3 if you want to use base64 only
# S3_ENABLED=false

# ============================================================================
# EXISTING CONFIGURATION (keep these)
# ============================================================================
ASSEMBLY_AI_KEY=your_assembly_ai_key
OPENAI_API_KEY=your_openai_key
```

### Step 2: Install Dependencies

```bash
cd backend
pip install -r requirements.txt

# Ensure these are in requirements.txt:
# modal>=0.57.0
# boto3>=1.28.0
# requests>=2.31.0
```

### Step 3: Update requirements.txt

Add to `backend/requirements.txt` if not present:

```txt
modal>=0.57.0
boto3>=1.28.0
requests>=2.31.0
```

---

## Deployment

### Option 1: Heroku (Existing Setup)

```bash
# Set environment variables
heroku config:set MODAL_VOICECRAFT_URL="https://your-workspace--voicecraft-voice-cloning-web.modal.run"
heroku config:set AWS_ACCESS_KEY_ID="AKIA..."
heroku config:set AWS_SECRET_ACCESS_KEY="wJal..."
heroku config:set AWS_S3_BUCKET="wavecraft-audio"
heroku config:set AWS_REGION="us-east-1"

# Deploy
git add .
git commit -m "Add Modal + S3 integration"
git push heroku main
```

### Option 2: Local Development

```bash
# Load environment variables
export $(cat backend/.env | xargs)

# Or use python-dotenv
pip install python-dotenv

# Run Flask
cd backend
python run.py
```

### Option 3: Docker (Advanced)

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY backend/ /app/backend/
COPY modal_service/ /app/modal_service/

RUN pip install -r backend/requirements.txt

# Set environment variables
ENV MODAL_VOICECRAFT_URL=${MODAL_VOICECRAFT_URL}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

CMD ["python", "backend/run.py"]
```

```bash
# Build and run
docker build -t wavecraft .
docker run -p 5000:5000 --env-file backend/.env wavecraft
```

---

## Testing

### Test 1: Modal API Health Check

```bash
curl https://your-modal-url.modal.run/health
```

Expected:
```json
{"status": "healthy", "service": "voicecraft-voice-cloning", "version": "1.0.0"}
```

### Test 2: Voice Cloning Endpoint

```python
import requests
import base64

# Create test audio (1 second of silence)
import numpy as np
import wave
import io

sample_rate = 24000
duration = 1.0
samples = np.zeros(int(sample_rate * duration), dtype=np.int16)

# Encode to WAV
buffer = io.BytesIO()
with wave.open(buffer, 'wb') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(samples.tobytes())

audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Test Modal API
response = requests.post(
    "https://your-modal-url.modal.run/clone",
    json={
        "reference_audio_b64": audio_b64,
        "original_text": "Hello world",
        "modified_text": "Hi there",
        "sample_rate": 24000
    },
    timeout=120
)

print(f"Status: {response.status_code}")
print(f"Result: {response.json()}")
```

### Test 3: S3 Upload

```python
from backend.utils.s3_storage import S3AudioStorage

# Initialize
storage = S3AudioStorage()

print(f"S3 enabled: {storage.is_enabled()}")
print(f"Bucket: {storage.bucket_name}")

# Test upload
test_data = b"test audio data"
s3_key = storage.upload_audio(test_data, filename="test.wav")
print(f"Uploaded: {s3_key}")

# Test presigned URL
url = storage.get_presigned_url(s3_key)
print(f"Download URL: {url}")
```

### Test 4: End-to-End Test

```bash
# Upload audio file
curl -X POST http://localhost:5000/api/upload \
  -F "audio_file=@test_audio.wav"

# Transcribe
# (get transcription_id from upload response)

# Modify text in frontend

# Submit modifications
curl -X POST http://localhost:5000/api/modified_transcript \
  -F "conversation_mod=@modified_conversations.json"

# Check response for:
# - "audio_url": "https://s3..." (S3 URL)
# - "storage_method": "s3"
# - "segments_cloned": N
```

---

## Monitoring

### Modal Dashboard

1. Go to https://modal.com/apps
2. Select `voicecraft-voice-cloning`
3. View metrics:
   - Request count
   - Average latency
   - GPU utilization
   - Cost breakdown

### AWS S3 Console

1. Go to AWS Console → S3 → wavecraft-audio
2. Monitor:
   - Storage size (GB)
   - Number of objects
   - Request metrics

### Backend Logs

```python
# Add to routes.py for monitoring
from backend.mcp_agents.voice_cloning_agent import VoiceCloningAgent

voice_agent = VoiceCloningAgent()
stats = voice_agent.get_stats()

print(f"""
Voice Cloning Stats:
- Total requests: {stats['total_requests']}
- Successful: {stats['successful_clones']}
- Success rate: {stats['success_rate']}
- Avg time: {stats['average_inference_time']}
- Modal configured: {stats['modal_url_configured']}
""")
```

---

## Cost Management

### Monthly Cost Estimate

**Scenario: 1000 voice cloning requests/month**

| Service | Usage | Unit Cost | Monthly Cost |
|---------|-------|-----------|--------------|
| **Modal (A10G)** | 1000 req × 8s | $0.0006/s | $4.80 |
| **S3 Storage** | 5 GB average | $0.023/GB | $0.12 |
| **S3 Transfer** | 100 GB download | $0.09/GB | $9.00 |
| **Heroku Dyno** | 1× Hobby | $7.00/mo | $7.00 |
| **AssemblyAI** | 50 hrs transcription | $0.25/hr | $12.50 |
| **Total** | | | **$33.42/mo** |

**Comparison to alternatives:**
- AWS EC2 (24/7 GPU): ~$350/month
- Google Colab Pro+: $49.99/month (limited GPU time)
- Our approach: **$33/month** ✅ 90% savings

### Cost Optimization Tips

1. **Reduce Modal cold starts**
   ```python
   # In voicecraft_service.py
   container_idle_timeout=600  # Keep warm for 10 min (costs ~$0.36/hour idle)
   ```

2. **Enable S3 lifecycle**
   - Auto-delete files after 30 days
   - Saves ~87% storage costs

3. **Use CloudFront (CDN) for S3**
   - Faster downloads
   - Cheaper transfer costs ($0.085/GB vs $0.09/GB)

4. **Batch processing**
   - Process multiple segments in parallel (already implemented)
   - Reduces total GPU time

5. **Monitor usage**
   ```python
   # Track daily costs
   import modal

   # Get usage stats
   modal.usage()  # Shows current month's spending
   ```

---

## Troubleshooting

### Issue 1: Modal Deployment Fails

**Error:** `ModuleNotFoundError: No module named 'voicecraft'`

**Solution:**
```bash
# Ensure VoiceCraft is cloned in image build
# Check voicecraft_service.py lines 40-48
```

**Error:** `GPU not available`

**Solution:**
- Modal account needs GPU access (request if needed)
- Check GPU quota in Modal dashboard

### Issue 2: S3 Upload Fails

**Error:** `NoCredentialsError: Unable to locate credentials`

**Solution:**
```bash
# Verify environment variables
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Or check .env file
cat backend/.env | grep AWS
```

**Error:** `Access Denied (403)`

**Solution:**
- Check IAM user has S3 permissions
- Verify bucket name is correct
- Ensure bucket is in same region as configured

### Issue 3: Voice Cloning Returns Original Audio

**Check logs:**
```bash
heroku logs --tail | grep "Voice cloning"
```

**Common causes:**
1. Modal URL not configured → Check `MODAL_VOICECRAFT_URL`
2. Modal endpoint unreachable → Test with `curl`
3. Audio format mismatch → Check sample rate (24000 Hz)
4. Text unchanged → Expected behavior (optimization)

### Issue 4: Slow Performance

**Symptoms:** Voice cloning takes >30s per segment

**Solutions:**

1. **Check Modal GPU**
   ```python
   # Upgrade to faster GPU
   gpu="A10G"  # from "T4"
   ```

2. **Reduce reference duration**
   ```python
   # In voice_cloning_agent.py
   'reference_duration_target': 8.0,  # from 10.0
   ```

3. **Enable parallel processing**
   - Already enabled in speech_processing_agent.py
   - Check logs for "parallel voice cloning"

### Issue 5: High Costs

**Check usage:**
```bash
# Modal usage
modal usage

# S3 usage
aws s3api list-objects --bucket wavecraft-audio --output json --query "[sum(Contents[].Size)]"
```

**Reduce costs:**
1. Decrease `container_idle_timeout` (more cold starts, but cheaper idle)
2. Enable S3 lifecycle policy (auto-delete old files)
3. Use T4 GPU instead of A10G (50% cheaper, but slower)

---

## Advanced Configuration

### Smart Warmup (Eliminate Cold Starts)

When user uploads audio, warm Modal during transcription:

```python
# In routes.py upload endpoint
import requests
import asyncio

async def warm_modal_endpoint():
    """Warm Modal endpoint during transcription"""
    try:
        response = requests.get(
            f"{os.environ.get('MODAL_VOICECRAFT_URL')}/health",
            timeout=10
        )
        print(f"[DEBUG] Modal warmed: {response.status_code}")
    except:
        pass

# During upload
async def process_upload():
    # Start transcription + warmup in parallel
    transcription_task = transcribe_audio(...)
    warmup_task = warm_modal_endpoint()

    await asyncio.gather(transcription_task, warmup_task)
```

### CloudFront CDN for S3 (Faster Downloads)

1. Go to AWS Console → CloudFront
2. Create distribution:
   - Origin: `wavecraft-audio.s3.amazonaws.com`
   - Viewer protocol: Redirect HTTP to HTTPS
3. Update presigned URL generation to use CloudFront domain

### Monitoring Dashboard

```python
# monitoring.py
from flask import jsonify

@app.route('/api/stats')
def get_stats():
    from backend.mcp_agents.voice_cloning_agent import VoiceCloningAgent
    from backend.utils.s3_storage import S3AudioStorage

    voice_agent = VoiceCloningAgent()
    s3_storage = S3AudioStorage()

    return jsonify({
        'voice_cloning': voice_agent.get_stats(),
        's3_storage': s3_storage.get_stats(),
        'modal_url': os.environ.get('MODAL_VOICECRAFT_URL'),
        'timestamp': datetime.utcnow().isoformat()
    })
```

---

## Security Best Practices

1. **Never commit credentials**
   ```bash
   # Add to .gitignore
   .env
   *.pem
   *.key
   ```

2. **Use environment variables**
   - Never hardcode API keys
   - Use Heroku config vars or .env files

3. **Rotate credentials regularly**
   - AWS access keys: every 90 days
   - Modal tokens: every 180 days

4. **Enable S3 bucket encryption**
   - Server-side encryption (AES-256)
   - Already enabled by default

5. **Monitor access logs**
   ```bash
   # Enable S3 access logging
   aws s3api put-bucket-logging --bucket wavecraft-audio \
     --bucket-logging-status '{
       "LoggingEnabled": {
         "TargetBucket": "wavecraft-logs",
         "TargetPrefix": "s3-access/"
       }
     }'
   ```

---

## Next Steps

✅ Modal deployed and tested
✅ S3 bucket configured
✅ Backend updated with credentials
✅ End-to-end testing completed

**Ready for production!**

**Recommended next steps:**
1. Set up monitoring dashboard
2. Configure CloudFront CDN
3. Enable S3 lifecycle policy
4. Implement smart warmup
5. Test with real users

**Questions or issues?**
- Modal docs: https://modal.com/docs
- AWS S3 docs: https://docs.aws.amazon.com/s3/
- GitHub Issues: [your-repo]/issues

---

**Last updated:** 2024
**Version:** 1.0
**Estimated setup time:** 30-60 minutes
