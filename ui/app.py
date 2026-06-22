"""
Gradio demo UI — runs against the FastAPI backend.
Start the API first: uvicorn api.main:app --reload
Then: python ui/app.py
"""
import os
import requests
import tempfile
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CLONE_NAME = os.getenv("CLONE_NAME", "Will")


# ── helpers ──────────────────────────────────────────────────────────────────

def _post(path: str, payload: dict, timeout: int = 12000) -> requests.Response:
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r


def _save_bytes(content: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.flush()
    return tmp.name


# ── tab handlers ─────────────────────────────────────────────────────────────

def chat_text(message: str, history: list, session_id: str):
    r = _post("/chat/", {"message": message, "session_id": session_id})
    reply = r.json()["response"]
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return history, ""


def chat_voice(message: str, history: list, session_id: str):
    r = _post("/chat/", {"message": message, "session_id": session_id})
    text = r.json()["response"]
    audio_r = _post("/tts/synthesize", {"text": text}, timeout=10000)
    audio_path = _save_bytes(audio_r.content, ".wav")
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
    return history, audio_path, ""


def chat_video(message: str, history: list, session_id: str):
    video_r = _post("/video/generate", {"message": message, "session_id": session_id}, timeout=12000)
    video_path = _save_bytes(video_r.content, ".mp4")
    text = video_r.headers.get("X-Response-Text", "")
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
    return history, video_path, ""


def clear_history(session_id: str):
    requests.delete(f"{API_BASE}/chat/history/{session_id}", timeout=10)
    return [], [], []


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title=f"CloneMe — {CLONE_NAME}") as demo:
    gr.Markdown(f"# CloneMe — Chat with {CLONE_NAME}")
    gr.Markdown(
        "A RAG-powered digital twin: text, voice (Chatterbox), and video (SadTalker) modes."
    )

    session_state = gr.State("session_" + os.urandom(4).hex())

    with gr.Tabs():

        # ── Text Chat ──────────────────────────────────────────────────────
        with gr.Tab("Text Chat"):
            text_chatbot = gr.Chatbot(height=400, label=CLONE_NAME)
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Ask me anything about myself…",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear history", size="sm")

            send_btn.click(
                fn=chat_text,
                inputs=[text_input, text_chatbot, session_state],
                outputs=[text_chatbot, text_input],
            )
            text_input.submit(
                fn=chat_text,
                inputs=[text_input, text_chatbot, session_state],
                outputs=[text_chatbot, text_input],
            )

        # ── Voice Response ─────────────────────────────────────────────────
        with gr.Tab("Voice Response"):
            gr.Markdown("Gets a RAG answer, then speaks it in your cloned voice.")
            voice_chatbot = gr.Chatbot(height=300, label=f"Conversation with {CLONE_NAME}")
            with gr.Row():
                with gr.Column(scale=2):
                    voice_input = gr.Textbox(
                        label="Your question",
                        placeholder="What are your hobbies?",
                        lines=2,
                    )
                    voice_btn = gr.Button("Ask", variant="primary")
                with gr.Column(scale=1):
                    voice_audio_out = gr.Audio(label="Voice response", autoplay=True)

            voice_btn.click(
                fn=chat_voice,
                inputs=[voice_input, voice_chatbot, session_state],
                outputs=[voice_chatbot, voice_audio_out, voice_input],
            )
            voice_input.submit(
                fn=chat_voice,
                inputs=[voice_input, voice_chatbot, session_state],
                outputs=[voice_chatbot, voice_audio_out, voice_input],
            )

        # ── Video Response ─────────────────────────────────────────────────
        with gr.Tab("Video Response"):
            gr.Markdown(
                "Full pipeline: RAG answer → Chatterbox voice → SadTalker talking head.  \n"
                "Expect **1–5 minutes** of processing per response."
            )
            video_chatbot = gr.Chatbot(height=250, label=f"Conversation with {CLONE_NAME}")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Textbox(
                        label="Your question",
                        placeholder="Tell me about your work experience.",
                        lines=2,
                    )
                    video_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column(scale=2):
                    video_out = gr.Video(label="Video response")

            video_btn.click(
                fn=chat_video,
                inputs=[video_input, video_chatbot, session_state],
                outputs=[video_chatbot, video_out, video_input],
            )

    # Wire clear after all chatbots are defined
    clear_btn.click(
        fn=clear_history,
        inputs=[session_state],
        outputs=[text_chatbot, voice_chatbot, video_chatbot],
    )


if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("GRADIO_PORT", "7860")), share=False, theme=gr.themes.Soft())
