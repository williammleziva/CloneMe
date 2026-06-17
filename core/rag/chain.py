from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
from .vectorstore import get_retriever

MODEL_ID = os.getenv("LLM_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

SYSTEM_PROMPT = """You are a digital clone of {name}. Answer questions about yourself \
using ONLY the provided context. Speak in first person, be conversational and accurate. \
If the context doesn't contain enough information, say so rather than guessing.

Relevant facts about you:
{context}"""

# In-memory session store — swap for Redis in prod
_session_store: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def _format_docs(docs) -> str:
    return "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(docs))


def load_llm() -> HuggingFacePipeline:
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
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=pipe, pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
    })


def build_rag_chain(name: str = "Will") -> tuple[RunnableWithMessageHistory, any]:
    llm = load_llm()
    retriever = get_retriever(k=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | _format_docs,
            "question": lambda x: x["question"],
            "name": lambda _: name,
            "history": lambda x: x.get("history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history, retriever
