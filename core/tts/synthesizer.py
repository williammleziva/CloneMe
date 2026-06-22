import time
import torch
import torchaudio
import numpy as np
from pathlib import Path

from chatterbox.tts import ChatterboxTTS

VOICE_REF_DIR = Path("data/media/voice_ref")
DEFAULT_REF_AUDIO = str(VOICE_REF_DIR / "reference.wav")


class VoiceSynthesizer:
    """Chatterbox voice synthesizer. Clones your voice from a reference WAV."""

    def __init__(self, ref_audio_path: str = DEFAULT_REF_AUDIO):
        if not Path(ref_audio_path).exists():
            raise FileNotFoundError(
                f"Reference audio not found at '{ref_audio_path}'. "
                "Record 10-30 seconds of clear speech and save it there."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[tts] Loading Chatterbox on {device}...")
        t0 = time.perf_counter()
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.ref_audio = ref_audio_path
        self.sample_rate = self.model.sr
        print(f"[tts] Model loaded in {time.perf_counter() - t0:.1f}s")

        # Pre-compute voice conditioning once so every synthesis skips librosa
        # load, resample, VoiceEncoder forward, and S3Gen ref embedding.
        print(f"[tts] Pre-computing voice conditioning from '{ref_audio_path}'...")
        t0 = time.perf_counter()
        self.model.prepare_conditionals(ref_audio_path)
        print(f"[tts] Voice conditioning ready in {time.perf_counter() - t0:.1f}s")

        # Warmup: pre-compile CUDA kernels so the first real request isn't slow.
        print("[tts] Warming up CUDA kernels (short synthesis)...")
        t0 = time.perf_counter()
        self._warmup()
        print(f"[tts] Warmup done in {time.perf_counter() - t0:.1f}s — synthesizer ready.")

    def _warmup(self) -> None:
        self.model.generate("Hello.")

    def synthesize(self, text: str, output_path: str) -> str:
        print(f"[tts] Synthesizing {len(text)} chars: '{text[:60]}{'...' if len(text) > 60 else ''}'")
        t0 = time.perf_counter()
        # No audio_prompt_path — conditioning was pre-computed at init.
        wav = self.model.generate(text)
        print(f"[tts] Synthesis done in {time.perf_counter() - t0:.1f}s → {output_path}")
        torchaudio.save(output_path, wav, self.sample_rate)
        return output_path

    def synthesize_to_array(self, text: str) -> tuple[np.ndarray, int]:
        wav = self.model.generate(text)
        return wav.squeeze().numpy(), self.sample_rate
