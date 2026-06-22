# CloneMe — AI Digital Twin

An end-to-end AI system that creates a personalized, interactive digital twin capable of responding via **text**, **cloned voice**, and **talking-head video**. Built on a RAG pipeline for context-aware answers, neural voice cloning, and generative video synthesis — all served through a REST API with a Gradio web UI.

---

## Demo

| Mode | Description |
|------|-------------|
| **Text Chat** | Multi-turn RAG conversation with persistent session history |
| **Voice Response** | RAG answer synthesized in a cloned voice (Chatterbox TTS) |
| **Video Response** | Full pipeline: RAG → voice → talking-head video (SadTalker) |

---

## Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌───────────────────────────────────────────┐
│  Gradio UI  │────▶│              FastAPI Backend               │
│  (port 7860)│     │                                           │
└─────────────┘     │  /chat/        /tts/synthesize  /video/   │
                    └──────┬──────────────┬──────────┬──────────┘
                           │              │          │
                    ┌──────▼──────┐       │          │
                    │  RAG Chain  │       │          │
                    │             │       │          │
                    │  Retriever  │       │          │
                    │  (Qdrant)   │       │          │
                    │      +      │       │          │
                    │  LLM        │       │          │
                    │  (Phi-3)    │       │          │
                    └──────┬──────┘       │          │
                           │ Text         │          │
                    ┌──────▼──────────────▼──┐       │
                    │   Chatterbox TTS       │       │
                    │   (voice cloning)      │       │
                    └──────┬─────────────────┘       │
                           │ WAV                      │
                    ┌──────▼─────────────────────────▼──┐
                    │         SadTalker                  │
                    │   (talking-head video synthesis)   │
                    └───────────────────────────────────┘
                                    │ MP4
                                    ▼
                              User Response
```

---

## Features

- **RAG Chatbot** — Retrieves personal facts from a Qdrant vector store to answer questions accurately in-character, with full session-based conversation history
- **Voice Cloning** — Synthesizes responses using a real reference voice recording via Chatterbox TTS
- **Talking-Head Video** — Animates a reference photo with the generated voice using SadTalker
- **REST API** — Clean FastAPI backend with lifespan model loading, CORS support, and a `/health` endpoint
- **Gradio UI** — Three-tab web interface for text, voice, and video interaction modes
- **Configurable Identity** — Swap name, facts, voice, and photo via `.env` and YAML to clone anyone

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Microsoft Phi-3-mini-4k-instruct (HuggingFace Transformers) |
| **RAG** | LangChain, `RunnableWithMessageHistory` |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (Sentence Transformers) |
| **Vector DB** | Qdrant (local on-disk) |
| **Voice Cloning** | Chatterbox TTS |
| **Talking-Head Video** | SadTalker, GFPGAN, face_alignment |
| **Deep Learning** | PyTorch 2.4+ (CUDA 12.1), torchaudio, torchvision |
| **API** | FastAPI, Uvicorn |
| **UI** | Gradio |
| **Audio** | librosa, soundfile, scipy |

---

## Project Structure

```
CloneMe/
├── api/
│   ├── main.py               # FastAPI app with lifespan model loading
│   └── routes/
│       ├── chat.py           # RAG chat endpoint + history management
│       ├── tts.py            # Voice synthesis endpoint
│       └── video.py          # Full video generation pipeline
├── core/
│   ├── rag/
│   │   ├── chain.py          # LangChain RAG chain + LLM loading
│   │   └── vectorstore.py    # Qdrant vector store + retriever
│   ├── tts/
│   │   └── synthesizer.py    # Chatterbox TTS with voice pre-conditioning
│   └── video/
│       └── generator.py      # SadTalker inference + model weight management
├── data/
│   ├── facts/
│   │   ├── profile.yaml.example  # Template — copy and customize
│   │   └── profile.yaml          # Your personal facts (private, in .gitignore)
│   └── media/
│       ├── voice_ref/
│       │   └── reference.wav     # Your voice recording (private)
│       └── reference_photo.jpg   # Your face photo (private)
├── scripts/
│   └── ingest_facts.py       # YAML → embeddings → Qdrant
├── third_party/
│   └── SadTalker/            # Talking-head model (git submodule)
├── ui/
│   └── app.py                # Gradio web interface
├── .env.example
└── requirements.txt
```

---

## Setup

### Requirements

- Python 3.11
- NVIDIA GPU with 8 GB+ VRAM (tested on RTX 3060)
- CUDA 12.1
- FFmpeg

### Install

```bash
git clone --recurse-submodules https://github.com/williammleziva/CloneMe.git
cd CloneMe
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env with your values
```

| Variable | Default | Description |
|----------|---------|-------------|
| `CLONE_NAME` | `Will` | Name used in system prompt |
| `LLM_MODEL_ID` | `microsoft/Phi-3-mini-4k-instruct` | HuggingFace model ID |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for retrieval |
| `TTS_REF_AUDIO` | `data/media/voice_ref/reference.wav` | 10–30s reference voice clip (private) |
| `REFERENCE_IMAGE` | `data/media/reference_photo.jpg` | Frontal face photo (private) |
| `HF_TOKEN` | — | HuggingFace token (for gated models) |

### Prepare Your Data

Copy and customize the example profile, then add private media:

1. **Profile facts** — Copy the template and fill in your own:
   ```bash
   cp data/facts/profile.yaml.example data/facts/profile.yaml
   # Edit data/facts/profile.yaml with your personal facts
   ```

2. **Voice reference** — Record a 10–30 second clip of your voice:
   ```
   data/media/voice_ref/reference.wav
   ```

3. **Face photo** — Add a frontal-facing photo of yourself:
   ```
   data/media/reference_photo.jpg
   ```

**Privacy note:** `profile.yaml`, voice recordings, and photos are all in `.gitignore` — they stay private and never commit to the repository.

### Ingest Facts

```bash
python scripts/ingest_facts.py
```

Chunks the YAML facts, generates embeddings, and stores them in the local Qdrant database.

---

## Running

```bash
# Terminal 1 — Start the API
uvicorn api.main:app --reload

# Terminal 2 — Start the UI
python ui/app.py
```

- API: `http://localhost:8000` — Swagger docs at `/docs`
- UI: `http://localhost:7860`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/` | RAG chat — `{message, session_id}` → `{response}` |
| `DELETE` | `/chat/history/{session_id}` | Clear session history |
| `POST` | `/tts/synthesize` | Voice synthesis — `{text}` → WAV |
| `POST` | `/video/generate` | Full pipeline — `{message, session_id}` → MP4 |
| `GET` | `/health` | Health check |

---

## How It Works

### Ingestion
Personal facts (work history, skills, personality, etc.) are stored in `data/facts/profile.yaml`. The ingest script splits these into chunks, generates dense vector embeddings using `BAAI/bge-small-en-v1.5`, and stores them in a local Qdrant instance.

### Retrieval-Augmented Generation
On each `/chat/` request, the top-5 most semantically similar facts are retrieved from Qdrant and injected into the LLM context alongside the conversation history. This grounds the LLM in real personal information and prevents hallucination. Microsoft Phi-3-mini is loaded with CUDA auto device mapping to fit within ~4GB VRAM.

### Voice Synthesis
Chatterbox TTS pre-computes a voice conditioning embedding from the reference audio at startup. Each TTS request passes text through the conditioned model to produce a cloned-voice WAV without reloading the reference each time.

### Video Generation
SadTalker takes a static face image and a driving audio file to produce a lip-synced talking-head video. Model weights (~4 GB) are downloaded from HuggingFace Hub on first run and cached locally. The full `/video/generate` pipeline chains RAG → TTS → SadTalker in a single request.

---

## License

MIT
