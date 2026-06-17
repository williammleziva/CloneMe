# CloneMe Setup

## System Requirements

- **Python:** 3.11 (see `.python-version`)
- **OS:** Linux (WSL2 on Windows, native Linux, or macOS)
- **GPU:** NVIDIA RTX 3060+ (CUDA 12.1)
- **RAM:** 16GB+ recommended

## WSL Setup (Windows)

```bash
# Install Python dev headers
sudo apt update
sudo apt install python3.11-dev build-essential

# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build vector store
python scripts/ingest_facts.py

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Linux/macOS Setup

```bash
# Install Python 3.11 (if not present)
# macOS: brew install python@3.11
# Ubuntu: apt install python3.11 python3.11-dev

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python scripts/ingest_facts.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

Create `.env` in project root:

```env
CLONE_NAME=Will
LLM_MODEL_ID=microsoft/Phi-3-mini-4k-instruct
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
TTS_REF_AUDIO=data/media/voice_ref/reference.wav
REFERENCE_IMAGE=data/media/reference_photo.jpg
HF_TOKEN=your_hf_token_here
```

## Data Setup

1. **Voice Reference:** Record 10-30 seconds of clear speech, save as `data/media/voice_ref/reference.wav`
2. **Face Photo:** Frontal face photo, save as `data/media/reference_photo.jpg`
3. **Profile Facts:** Edit `data/facts/profile.yaml` with your experience, skills, projects

## API Endpoints

- **Chat:** `POST /chat/` — RAG-backed conversational AI
- **TTS:** `POST /tts/synthesize` — Voice synthesis
- **Video:** `POST /video/generate` — Full pipeline: chat → voice → talking-head video
- **Health:** `GET /health` — API status

## UI

```bash
python ui/app.py
```

Opens Gradio interface at `http://localhost:7860`
