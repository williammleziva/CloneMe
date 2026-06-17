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

def _post(path: str, payload: dict) -> requests.Response:
    r = requests.post(f"{API_BASE}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r


def _save_bytes(content: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(content)
    tmp.flush()
    return tmp.name


# ── tab handlers ─────────────────────────────────────────────────────────────

def chat_text(message: str, history: list, session_id: str) -> tuple[list, str]:
    r = _post("/chat/", {"message": message, "session_id": session_id})
    reply = r.json()["response"]
    history = history + [[message, reply]]
    return history, ""


def chat_voice(message: str, session_id: str):
    r = _post("/chat/", {"message": message, "session_id": session_id})
    text = r.json()["response"]

    audio_r = _post("/tts/synthesize", {"text": text})
    audio_path = _save_bytes(audio_r.content, ".wav")

    return text, audio_path


def chat_video(message: str, session_id: str):
    video_r = _post("/video/generate", {"message": message, "session_id": session_id})
    video_path = _save_bytes(video_r.content, ".mp4")
    return video_path


def clear_history(session_id: str):
    requests.delete(f"{API_BASE}/chat/history/{session_id}")
    return [], f"History cleared for session '{session_id}'"


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title=f"CloneMe — {CLONE_NAME}", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# CloneMe — Chat with {CLONE_NAME}")
    gr.Markdown(
        "A RAG-powered digital twin: text, voice (Chatterbox), and video (SadTalker) modes."
    )

    session_state = gr.State("session_" + os.urandom(4).hex())

    with gr.Tabs():

        # ── Text Chat ──────────────────────────────────────────────────────
        with gr.Tab("Text Chat"):
            chatbot = gr.Chatbot(height=400, label=CLONE_NAME)
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Ask me anything about myself…",
                    show_label=False,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear history", size="sm")
            status_box = gr.Textbox(show_label=False, interactive=False, visible=False)

            send_btn.click(
                fn=chat_text,
                inputs=[text_input, chatbot, session_state],
                outputs=[chatbot, text_input],
            )
            text_input.submit(
                fn=chat_text,
                inputs=[text_input, chatbot, session_state],
                outputs=[chatbot, text_input],
            )
            clear_btn.click(
                fn=clear_history,
                inputs=[session_state],
                outputs=[chatbot, status_box],
            )

        # ── Voice Response ─────────────────────────────────────────────────
        with gr.Tab("Voice Response"):
            gr.Markdown("Gets a RAG answer, then speaks it in your cloned voice.")
            with gr.Row():
                with gr.Column():
                    voice_input = gr.Textbox(
                        label="Your question", lines=3, placeholder="What are your hobbies?"
                    )
                    voice_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    voice_text_out = gr.Textbox(
                        label="Response text", lines=3, interactive=False
                    )
                    voice_audio_out = gr.Audio(label="Voice response", autoplay=True)

            voice_btn.click(
                fn=chat_voice,
                inputs=[voice_input, session_state],
                outputs=[voice_text_out, voice_audio_out],
            )

        # ── Video Response ─────────────────────────────────────────────────
        with gr.Tab("Video Response"):
            gr.Markdown(
                "Full pipeline: RAG answer → Chatterbox voice → SadTalker talking head.  \n"
                "Expect **15–45 seconds** of processing per response."
            )
            with gr.Row():
                with gr.Column():
                    video_input = gr.Textbox(
                        label="Your question", lines=3, placeholder="Tell me about your work experience."
                    )
                    video_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column():
                    video_out = gr.Video(label="Video response")

            video_btn.click(
                fn=chat_video,
                inputs=[video_input, session_state],
                outputs=[video_out],
            )


if __name__ == "__main__":
    demo.launch(server_port=int(os.getenv("GRADIO_PORT", "7860")), share=False)
