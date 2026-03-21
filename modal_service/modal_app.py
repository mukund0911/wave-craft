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

# WhisperX image: Python 3.11 — let whisperx drive torch/pyannote versions
whisperx_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "fastapi[standard]",
        "pydub",
        "soundfile",
        "huggingface_hub",
    )
    .pip_install(
        "whisperx @ git+https://github.com/m-bain/whisperx.git",
        "pyannote.audio>=3.3.0",
        gpu="a10g",
    )
    .env({
        "HF_HOME": MODEL_CACHE_DIR,
        "TORCH_HOME": MODEL_CACHE_DIR,
        "PYANNOTE_AUDIO_USE_SOUNDFILE": "1",
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
                model_name="pyannote/speaker-diarization-community-1",
                token=hf_token, device=self.device
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
                diarize_args = {"return_embeddings": True}
                if num_speakers:
                    diarize_args["num_speakers"] = num_speakers
                else:
                    diarize_args["min_speakers"] = 2
                    diarize_args["max_speakers"] = 10
                diarize_result = self.diarize_pipeline(audio_path, **diarize_args)

                # Unpack embeddings if returned
                if isinstance(diarize_result, tuple):
                    diarize_segments, speaker_embeddings = diarize_result
                else:
                    diarize_segments = diarize_result
                    speaker_embeddings = None

                result = self.whisperx.assign_word_speakers(diarize_segments, result)

                # Auto-merge similar speakers based on embedding cosine similarity
                if speaker_embeddings and not num_speakers:
                    result = self._merge_similar_speakers(result, speaker_embeddings)
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

    def _merge_similar_speakers(self, result, speaker_embeddings, threshold=0.65):
        """
        Merge speakers whose voice embeddings are too similar.
        Uses Union-Find for transitive merges — if A~B and B~C, all three merge.
        The most-frequent speaker in each group becomes the canonical label.
        """
        import numpy as np

        speakers = list(speaker_embeddings.keys())
        if len(speakers) <= 1:
            return result

        embeddings = {s: np.array(e) for s, e in speaker_embeddings.items()}

        # Union-Find
        parent = {s: s for s in speakers}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        # Merge all pairs above threshold
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                s1, s2 = speakers[i], speakers[j]
                e1, e2 = embeddings[s1], embeddings[s2]
                norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
                if norm1 == 0 or norm2 == 0:
                    continue
                similarity = np.dot(e1, e2) / (norm1 * norm2)
                if similarity >= threshold:
                    union(s1, s2)
                    logger.info(f"Linking speakers {s1} <-> {s2} (similarity={similarity:.3f})")

        # Build groups and pick canonical (most segments) per group
        from collections import defaultdict
        groups = defaultdict(list)
        for s in speakers:
            groups[find(s)].append(s)

        merge_map = {}
        for members in groups.values():
            if len(members) <= 1:
                continue
            counts = {s: sum(1 for seg in result["segments"] if seg.get("speaker") == s) for s in members}
            canonical = max(members, key=lambda s: counts[s])
            for s in members:
                if s != canonical:
                    merge_map[s] = canonical
                    logger.info(f"Merging speaker {s} into {canonical}")

        if not merge_map:
            return result

        for segment in result["segments"]:
            speaker = segment.get("speaker", "")
            if speaker in merge_map:
                segment["speaker"] = merge_map[speaker]
            for word in segment.get("words", []):
                if word.get("speaker", "") in merge_map:
                    word["speaker"] = merge_map[word["speaker"]]

        return result

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
    scaledown_window=300,
    concurrency_limit=4,
)
class TTSService:
    @modal.enter()
    def setup(self):
        """Load Chatterbox Turbo model on container start."""
        import torch
        from chatterbox.tts import ChatterboxTTS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Chatterbox Turbo on {self.device}...")

        # Try Turbo first, fall back to base if unavailable
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self.model = ChatterboxTurboTTS.from_pretrained(self.device)
            logger.info("✓ Chatterbox Turbo loaded")
        except (ImportError, Exception) as e:
            logger.warning(f"Turbo unavailable ({e}), falling back to base model")
            self.model = ChatterboxTTS.from_pretrained(self.device)
            logger.info("✓ Chatterbox base loaded")

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
            exaggeration = max(0.0, min(2.0, float(request.get("exaggeration", 0.5))))
            cfg_weight = max(0.0, min(1.0, float(request.get("cfg_weight", 0.5))))

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
                cfg_weight=cfg_weight,
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
