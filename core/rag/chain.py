from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from .vectorstore import get_retriever

MODEL_ID = os.getenv("LLM_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

SYSTEM_PROMPT = """You are {name}, having a casual chat with someone curious about you.

Reply like you're talking to a friend — short, natural, first person. One or two sentences max.
Don't ask follow-up questions. Don't generate fake dialogue or repeat yourself.
If the facts below don't cover the question, just say you're not sure.

What you know about yourself:
{context}"""

# In-memory session store — swap for Redis in prod
_session_store: dict[str, BaseChatMessageHistory] = {}

# Phi-3 special tokens that mark turn boundaries — used to strip any runaway generation.
_PHI3_STOP_MARKERS = ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"]


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def _format_docs(docs) -> str:
    return "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))


def _build_phi3_prompt(inputs: dict) -> str:
    """Format the prompt using Phi-3's native chat template so the model stops at <|end|>."""
    system = SYSTEM_PROMPT.format(name=inputs["name"], context=inputs["context"])
    parts = [f"<|system|>\n{system}<|end|>\n"]

    for msg in inputs.get("history", []):
        role = getattr(msg, "type", "")
        if role == "human":
            parts.append(f"<|user|>\n{msg.content}<|end|>\n")
        elif role == "ai":
            parts.append(f"<|assistant|>\n{msg.content}<|end|>\n")

    parts.append(f"<|user|>\n{inputs['question']}<|end|>\n<|assistant|>\n")
    return "".join(parts)


def _clean_output(text: str) -> str:
    """Strip any Phi-3 special tokens or continuation the model generates past its turn."""
    for marker in _PHI3_STOP_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


def load_llm() -> tuple[HuggingFacePipeline, AutoTokenizer]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
            attn_implementation="eager",
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Tell the pipeline to also stop at <|end|> (Phi-3 turn separator), not just <|endoftext|>.
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    eos_ids = [tokenizer.eos_token_id]
    if end_token_id and end_token_id != tokenizer.eos_token_id:
        eos_ids.append(end_token_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        eos_token_id=eos_ids,
    )

    return HuggingFacePipeline(pipeline=pipe, pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.5,
        "do_sample": True,
    })


def build_rag_chain(name: str = "Will") -> tuple[RunnableWithMessageHistory, any]:
    llm = load_llm()
    retriever = get_retriever(k=5)

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | _format_docs,
            "question": lambda x: x["question"],
            "name": lambda _: name,
            "history": lambda x: x.get("history", []),
        }
        | RunnableLambda(_build_phi3_prompt)
        | llm
        | StrOutputParser()
        | RunnableLambda(_clean_output)
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, retriever
