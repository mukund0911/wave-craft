"""
WaveCraft Modal GPU Service
Hosts WhisperX (transcription + diarization) and Chatterbox (TTS/voice cloning)
on A10G GPU as serverless endpoints.

Deploy: modal deploy modal_service/modal_app.py
Test:   modal serve modal_service/modal_app.py
"""

import modal
import os
import io
import base64
import tempfile
import logging

logger = logging.getLogger(__name__)

# ─── Modal App Setup ───

app = modal.App("wavecraft-gpu")

# Persistent volume for model caching
model_volume = modal.Volume.from_name("wavecraft-models", create_if_missing=True)
MODEL_CACHE_DIR = "/models"

# ─── GPU Images ───

# WhisperX image: Python 3.11, Torch 2.5.1, Numpy <2.0 (for compatibility)
whisperx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy>=1.24.0,<2.0.0",
        "fastapi[standard]",
        "pydub",
        gpu="a10g",
    )
    .pip_install(
        "whisperx @ git+https://github.com/m-bain/whisperx.git",
        "pyannote.audio>=3.1.0",
        "huggingface_hub",
    )
    .env({
        "HF_HOME": MODEL_CACHE_DIR,
        "TORCH_HOME": MODEL_CACHE_DIR,
        "TRANSFORMERS_CACHE": f"{MODEL_CACHE_DIR}/transformers",
    })
)

# Chatterbox TTS image: Python 3.11, Torch 2.5.1, Numpy <1.26
chatterbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "numpy>=1.24.0,<1.26.0",
        "fastapi[standard]",
        "peft",
        "huggingface_hub[hf_xet]",
        gpu="a10g",
    )
    .pip_install(
        "chatterbox-tts>=0.1.0",
    )
    .env({
        "HF_HOME": MODEL_CACHE_DIR,
        "TORCH_HOME": MODEL_CACHE_DIR,
        "TRANSFORMERS_CACHE": f"{MODEL_CACHE_DIR}/transformers",
    })
)


# ─── Transcription Endpoint (WhisperX) ───

@app.cls(
    image=whisperx_image,
    gpu="A10G",
    timeout=600,
    volumes={MODEL_CACHE_DIR: model_volume},
    secrets=[modal.Secret.from_name("wavecraft-secrets")],
    scaledown_window=120,
)
class TranscribeService:
    @modal.enter()
    def setup(self):
        """Load models on container start (cached in volume)."""
        import torch

        # Monkey-patch torch.load for PyTorch 2.6 compatibility
        _original_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _patched_load

        import whisperx

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"

        logger.info(f"Loading WhisperX model on {self.device}...")
        self.model = whisperx.load_model(
            "large-v2", self.device, compute_type=compute_type
        )
        logger.info("✓ WhisperX model loaded")

        # Load alignment model
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )
        logger.info("✓ Alignment model loaded")

        # Load diarization pipeline
        hf_token = os.environ.get("HF_TOKEN", "")
        self.diarize_pipeline = None
        if hf_token:
            from whisperx.diarize import DiarizationPipeline
            self.diarize_pipeline = DiarizationPipeline(
                use_auth_token=hf_token, device=self.device
            )
            logger.info("✓ Diarization pipeline loaded")

        self.whisperx = whisperx
        model_volume.commit()

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, request: dict):
        """Transcribe audio with diarization and word-level timestamps."""
        import torch
        from pydub import AudioSegment

        try:
            audio_b64 = request.get("audio")
            num_speakers = request.get("num_speakers")

            if not audio_b64:
                return {"status": "error", "error": "No audio data provided"}

            # Decode base64 audio to temp file
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                audio_path = f.name

            # Step 1: Transcribe
            audio = self.whisperx.load_audio(audio_path)
            result = self.model.transcribe(audio, batch_size=16)
            language = result.get("language", "en")

            # Step 2: Align words
            result = self.whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )

            # Step 3: Diarize speakers
            if self.diarize_pipeline is not None:
                diarize_args = {}
                if num_speakers:
                    diarize_args["num_speakers"] = num_speakers
                diarize_segments = self.diarize_pipeline(audio_path, **diarize_args)
                result = self.whisperx.assign_word_speakers(diarize_segments, result)
            else:
                for segment in result["segments"]:
                    segment["speaker"] = "SPEAKER_00"

            # Step 4: Build conversation structure
            conversations = self._build_conversations(result, audio_path)

            # Step 5: Full audio base64
            full_audio_b64 = self._audio_to_base64(audio_path)

            # Cleanup
            os.unlink(audio_path)

            return {
                "status": "completed",
                "conversations": conversations,
                "full_audio": full_audio_b64,
                "language": language,
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _build_conversations(self, result, audio_path):
        from pydub import AudioSegment

        full_audio = AudioSegment.from_file(audio_path)
        speaker_map = {}
        speaker_counter = 0
        conversations = []

        for segment in result.get("segments", []):
            speaker_id = segment.get("speaker", "SPEAKER_00")
            if speaker_id not in speaker_map:
                speaker_map[speaker_id] = chr(ord("A") + speaker_counter)
                speaker_counter += 1

            speaker_label = speaker_map[speaker_id]
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)

            segment_audio = full_audio[start_ms:end_ms]
            buffer = io.BytesIO()
            segment_audio.export(buffer, format="wav")
            segment_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            words_with_timestamps = []
            for word_info in segment.get("words", []):
                words_with_timestamps.append({
                    "word": word_info.get("word", ""),
                    "start": word_info.get("start", segment["start"]),
                    "end": word_info.get("end", segment["end"]),
                    "score": word_info.get("score", 1.0),
                })

            conv_key = f"conversation_{len(conversations)}"
            conversations.append({
                conv_key: {
                    "speaker": speaker_label,
                    "original": {
                        "text": segment["text"].strip(),
                        "speaker_audio": segment_b64,
                        "start": segment["start"],
                        "end": segment["end"],
                        "words": words_with_timestamps,
                    },
                    "modified": {
                        "text": segment["text"].strip(),
                        "emotions": [],
                    },
                }
            })

        return conversations

    def _audio_to_base64(self, audio_path):
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─── TTS Endpoint (Chatterbox) ───

@app.cls(
    image=chatterbox_image,
    gpu="A10G",
    timeout=300,
    volumes={MODEL_CACHE_DIR: model_volume},
    scaledown_window=120,
)
class TTSService:
    @modal.enter()
    def setup(self):
        """Load Chatterbox model on container start."""
        import torch
        from chatterbox.tts import ChatterboxTTS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Chatterbox on {self.device}...")
        self.model = ChatterboxTTS.from_pretrained(self.device)
        logger.info("✓ Chatterbox loaded")

        self.torch = torch
        model_volume.commit()

    @modal.fastapi_endpoint(method="POST")
    def synthesize(self, request: dict):
        """Synthesize speech with voice cloning."""
        import torchaudio
        import tempfile

        try:
            text = request.get("text", "")
            reference_audio_b64 = request.get("reference_audio", "")
            exaggeration = request.get("exaggeration", 0.5)

            if not text:
                return {"status": "error", "error": "No text provided"}

            # Save reference audio to temp file if provided
            ref_path = None
            if reference_audio_b64:
                audio_bytes = base64.b64decode(reference_audio_b64)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_bytes)
                    ref_path = f.name

            # Generate speech
            wav = self.model.generate(
                text,
                audio_prompt_path=ref_path,
                exaggeration=exaggeration,
            )

            # Convert tensor to base64 WAV
            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                wav.cpu(),
                self.model.sr,
                format="wav",
            )
            audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Cleanup
            if ref_path:
                os.unlink(ref_path)

            return {
                "status": "completed",
                "audio": audio_b64,
                "sample_rate": self.model.sr,
            }

        except Exception as e:
            logger.error(f"TTS failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


# ─── Health Check ───

@app.function(image=modal.Image.debian_slim().pip_install("fastapi[standard]"))
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "service": "wavecraft-gpu", "version": "1.0.0"}
