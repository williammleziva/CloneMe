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
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.ref_audio = ref_audio_path
        self.sample_rate = self.model.sr

    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize text to a WAV file. Returns the output path."""
        wav = self.model.generate(text, audio_prompt_path=self.ref_audio)
        torchaudio.save(output_path, wav, self.sample_rate)
        return output_path

    def synthesize_to_array(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to a numpy float array for in-memory use."""
        wav = self.model.generate(text, audio_prompt_path=self.ref_audio)
        return wav.squeeze().numpy(), self.sample_rate
