import subprocess
import shutil
import uuid
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

SADTALKER_DIR = os.getenv("SADTALKER_DIR", "third_party/SadTalker")
REFERENCE_IMAGE = os.getenv("REFERENCE_IMAGE", "data/media/reference_photo.jpg")
OUTPUT_DIR = Path("output/video")

_HF_REPO = "vinthony/SadTalker"
_ROOT_MODELS = [
    "auido2exp_00300-model.pth",
    "auido2pose_00140-model.pth",
    "epoch_20.pth",
    "facevid2vid_00189-model.pth.tar",
    "mapping_00109-model.pth.tar",
    "mapping_00229-model.pth.tar",
    "shape_predictor_68_face_landmarks.dat",
    "wav2lip.pth",
]


def _ensure_models(checkpoints_dir: Path) -> None:
    missing = [f for f in _ROOT_MODELS if not (checkpoints_dir / f).exists()]
    bfm_missing = not (checkpoints_dir / "BFM_Fitting" / "01_MorphableModel.mat").exists()

    if not missing and not bfm_missing:
        return

    print("[video] Downloading SadTalker model weights (~4 GB)...")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for filename in missing:
        print(f"[video]   {filename}")
        hf_hub_download(
            repo_id=_HF_REPO,
            filename=filename,
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False,
        )

    if bfm_missing:
        print("[video]   BFM_Fitting/")
        snapshot_download(
            repo_id=_HF_REPO,
            local_dir=str(checkpoints_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["BFM_Fitting/*"],
        )

    print("[video] Model download complete.")


class TalkingHeadGenerator:
    def __init__(
        self,
        sadtalker_dir: str = SADTALKER_DIR,
        ref_image: str = REFERENCE_IMAGE,
        enhancer: str = "gfpgan",
    ):
        self.sadtalker_dir = Path(sadtalker_dir)
        self.ref_image = Path(ref_image)
        self.enhancer = enhancer
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not self.sadtalker_dir.exists():
            raise RuntimeError(
                f"SadTalker not found at '{sadtalker_dir}'. "
                "Run: git clone https://github.com/OpenTalker/SadTalker third_party/SadTalker"
            )
        if not self.ref_image.exists():
            raise FileNotFoundError(
                f"Reference photo not found at '{ref_image}'. "
                "Add a clear frontal photo of your face."
            )

        _ensure_models(self.sadtalker_dir / "checkpoints")

    def generate(self, audio_path: str, output_filename: str | None = None) -> str:
        if output_filename is None:
            output_filename = f"response_{uuid.uuid4().hex[:8]}.mp4"

        output_path = OUTPUT_DIR / output_filename
        tmp_out_dir = OUTPUT_DIR / f"tmp_{uuid.uuid4().hex[:6]}"
        tmp_out_dir.mkdir(parents=True)

        cmd = [
            "python", "inference.py",
            "--driven_audio", str(Path(audio_path).resolve()),
            "--source_image", str(self.ref_image.resolve()),
            "--result_dir", str(tmp_out_dir.resolve()),
            "--still",
            "--preprocess", "full",
            "--enhancer", self.enhancer,
        ]

        result = subprocess.run(
            cmd,
            cwd=str(self.sadtalker_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)
            raise RuntimeError(f"SadTalker failed:\n{result.stderr}")

        videos = sorted(tmp_out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not videos:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)
            raise RuntimeError("SadTalker ran successfully but produced no video.")

        shutil.move(str(videos[-1]), str(output_path))
        shutil.rmtree(tmp_out_dir, ignore_errors=True)

        return str(output_path)
