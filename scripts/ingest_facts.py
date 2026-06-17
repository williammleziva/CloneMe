import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from core.rag.vectorstore import get_embeddings, COLLECTION_NAME, QDRANT_PATH

FACTS_DIR = Path("data/facts")


def load_facts() -> list[Document]:
    docs: list[Document] = []

    for yaml_file in sorted(FACTS_DIR.glob("*.yaml")):
        with open(yaml_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        category = yaml_file.stem
        _walk(data, category, yaml_file.name, docs)

    return docs


def _walk(node, category: str, source: str, docs: list[Document], section: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, category, source, docs, section=key)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _walk(item, category, source, docs, section=section)
            else:
                docs.append(Document(
                    page_content=str(item).strip(),
                    metadata={"category": category, "section": section, "source": source},
                ))
    else:
        text = f"{section}: {node}".strip() if section else str(node).strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"category": category, "section": section, "source": source},
            ))


def main():
    print(f"Loading facts from {FACTS_DIR}...")
    docs = load_facts()
    print(f"  {len(docs)} chunks loaded")

    if not docs:
        print("No documents found. Fill out data/facts/profile.yaml first.")
        return

    print("Building embeddings and vector store...")
    embeddings = get_embeddings()
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)

    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    print(f"Done. {len(docs)} facts indexed in Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
