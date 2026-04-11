"""
ingest.py

MVP ingestion pipeline for a conversation-neural-map project.

Expected input:
[
  {
    "id": "conv_001",
    "title": "Cookie pricing and bakery stress",
    "created_at": "2026-04-09T10:15:00Z",
    "messages": [
      {
        "role": "user",
        "content": "How much should I charge for these cookies?",
        "timestamp": "2026-04-09T10:15:00Z"
      },
      {
        "role": "assistant",
        "content": "Let's break down your costs first...",
        "timestamp": "2026-04-09T10:15:20Z"
      }
    ]
  }
]

Outputs:
- Chroma persistent collection with conversation documents + metadata + embeddings
- graph.json for the frontend
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
from openai import OpenAI


# =========================
# Config
# =========================

DATA_DIR = Path("./data")
INPUT_PATH = DATA_DIR / "conversations.json"
CHROMA_PATH = DATA_DIR / "chroma"
OUTPUT_GRAPH_PATH = DATA_DIR / "graph.json"
COLLECTION_NAME = "conversation_map"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 64

# Graph pruning
TOP_K_NEIGHBORS = 4
MIN_SIMILARITY = 0.72

# Optional truncation guard for embedding text
MAX_EMBEDDING_TEXT_CHARS = 8000


# =========================
# Data models
# =========================

@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[str] = None


@dataclass
class ConversationRecord:
    id: str
    title: str
    created_at: Optional[str]
    messages: List[Message]
    full_text: str
    summary: str
    tags: List[str]
    message_count: int
    char_count: int
    embedding_text: str


# =========================
# Helpers
# =========================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_iso(dt: Optional[str]) -> Optional[datetime]:
    if not dt:
        return None
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except ValueError:
        return None


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u0000", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_tag_slug(tag: str) -> str:
    tag = tag.lower().strip()
    tag = re.sub(r"[^a-z0-9]+", "_", tag)
    return tag.strip("_")


def chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# =========================
# Normalization
# =========================

def normalize_message(raw: Dict[str, Any]) -> Optional[Message]:
    content = clean_text(str(raw.get("content", "")))
    if not content:
        return None

    role = str(raw.get("role", "unknown")).strip().lower() or "unknown"
    timestamp = raw.get("timestamp")
    return Message(role=role, content=content, timestamp=timestamp)


def build_full_text(messages: List[Message]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.role.upper()
        if msg.timestamp:
            parts.append(f"[{msg.timestamp}] {role}: {msg.content}")
        else:
            parts.append(f"{role}: {msg.content}")
    return "\n".join(parts)


def summarize_conversation(title: str, full_text: str) -> str:
    """
    Placeholder summary function.
    Replace later with:
    - LLM summarization
    - heuristic summarization
    - or precomputed summaries from your pipeline
    """
    preview = full_text[:500].replace("\n", " ")
    if len(full_text) > 500:
        preview += " ..."
    return f"{title}: {preview}"


def extract_tags(title: str, full_text: str) -> List[str]:
    """
    Placeholder tag extractor.
    Replace later with a better classifier or LLM call.
    """
    text = f"{title}\n{full_text}".lower()

    keyword_map = {
        "coding": ["javascript", "python", "bug", "code", "debug", "react", "api"],
        "baking": ["cookie", "bread", "sourdough", "loaf", "bake", "starter"],
        "business": ["llc", "pricing", "profit", "customer", "brand", "business"],
        "legal": ["court", "custody", "motion", "filing", "contempt", "lawyer"],
        "family": ["yudy", "son", "daughter", "baby", "family", "pregnant"],
        "health": ["doctor", "hospital", "pain", "migraine", "recovery"],
        "gaming": ["game", "gpu", "steam", "fps", "valheim", "apex", "league"],
    }

    tags: List[str] = []
    for tag, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    if not tags:
        tags.append("general")

    return tags[:6]


def build_embedding_text(
    title: str,
    summary: str,
    tags: List[str],
    full_text: str,
) -> str:
    """
    The embedding text should be semantically dense, not the entire raw export.
    """
    sample = full_text[:MAX_EMBEDDING_TEXT_CHARS]
    text = (
        f"Title: {title}\n"
        f"Summary: {summary}\n"
        f"Tags: {', '.join(tags)}\n"
        f"Conversation:\n{sample}"
    )
    return clean_text(text)


def normalize_conversation(raw: Dict[str, Any]) -> ConversationRecord:
    conv_id = str(raw["id"])
    title = clean_text(str(raw.get("title", ""))) or f"Conversation {conv_id}"
    created_at = raw.get("created_at")

    raw_messages = raw.get("messages", [])
    messages: List[Message] = []

    for raw_msg in raw_messages:
        msg = normalize_message(raw_msg)
        if msg is not None:
            messages.append(msg)

    # Sort if timestamps exist
    messages.sort(key=lambda m: parse_iso(m.timestamp) or datetime.min)

    full_text = build_full_text(messages)
    summary = summarize_conversation(title=title, full_text=full_text)
    tags = extract_tags(title=title, full_text=full_text)
    embedding_text = build_embedding_text(
        title=title,
        summary=summary,
        tags=tags,
        full_text=full_text,
    )

    return ConversationRecord(
        id=conv_id,
        title=title,
        created_at=created_at,
        messages=messages,
        full_text=full_text,
        summary=summary,
        tags=tags,
        message_count=len(messages),
        char_count=len(full_text),
        embedding_text=embedding_text,
    )


# =========================
# Embeddings
# =========================

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def embed_text_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# =========================
# Chroma store
# =========================

def get_chroma_collection():
    """
    Chroma PersistentClient stores data on disk automatically.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def upsert_records(
    collection,
    records: List[ConversationRecord],
    embeddings: List[List[float]],
) -> None:
    ids = [record.id for record in records]
    documents = [record.embedding_text for record in records]
    metadatas = [
        {
            "title": record.title,
            "created_at": record.created_at or "",
            "summary": record.summary,
            "tags": ", ".join(record.tags),
            "message_count": record.message_count,
            "char_count": record.char_count,
        }
        for record in records
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


# =========================
# Graph building
# =========================

def build_nodes(records: List[ConversationRecord]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    for record in records:
        node = {
            "id": record.id,
            "label": record.title,
            "summary": record.summary,
            "tags": record.tags,
            "date": record.created_at,
            "val": max(1, min(30, record.message_count)),
            "message_count": record.message_count,
            "char_count": record.char_count,
        }
        nodes.append(node)

    return nodes


def build_edges_from_embeddings(
    records: List[ConversationRecord],
    embeddings_by_id: Dict[str, List[float]],
    top_k: int = TOP_K_NEIGHBORS,
    min_similarity: float = MIN_SIMILARITY,
) -> List[Dict[str, Any]]:
    """
    Simple local edge generation.
    Later you can swap this for Chroma neighbor queries if you want.
    """
    edges: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str]] = set()

    for source in records:
        source_vec = embeddings_by_id[source.id]
        scored: List[Tuple[str, float]] = []

        for target in records:
            if source.id == target.id:
                continue

            target_vec = embeddings_by_id[target.id]
            score = cosine_similarity(source_vec, target_vec)

            if score >= min_similarity:
                scored.append((target.id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored[:top_k]

        for target_id, score in top_matches:
            pair = tuple(sorted((source.id, target_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            edges.append(
                {
                    "source": source.id,
                    "target": target_id,
                    "weight": round(score, 4),
                    "relation_type": "semantic_match",
                }
            )

    return edges


def export_graph_json(
    records: List[ConversationRecord],
    embeddings: List[List[float]],
    output_path: Path,
) -> None:
    nodes = build_nodes(records)
    embeddings_by_id = {
        record.id: embedding
        for record, embedding in zip(records, embeddings)
    }
    links = build_edges_from_embeddings(records, embeddings_by_id)

    payload = {
        "nodes": nodes,
        "links": links,
    }
    save_json(output_path, payload)


# =========================
# Main pipeline
# =========================

def run_ingest(input_path: Path = INPUT_PATH) -> None:
    raw_items = load_json(input_path)
    if not isinstance(raw_items, list):
        raise ValueError("Input file must be a JSON array of conversations.")

    records = [normalize_conversation(item) for item in raw_items]
    if not records:
        raise ValueError("No valid conversations found.")

    openai_client = get_openai_client()
    collection = get_chroma_collection()

    all_embeddings: List[List[float]] = []

    for batch in chunked(records, EMBED_BATCH_SIZE):
        texts = [record.embedding_text for record in batch]
        batch_embeddings = embed_text_batch(openai_client, texts)
        upsert_records(collection, batch, batch_embeddings)
        all_embeddings.extend(batch_embeddings)

    export_graph_json(records, all_embeddings, OUTPUT_GRAPH_PATH)

    print(f"Ingest complete.")
    print(f"Conversations processed: {len(records)}")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Graph output: {OUTPUT_GRAPH_PATH}")


if __name__ == "__main__":
    run_ingest()