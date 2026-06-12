"""
ingest_local.py — Synapses ingestion, ZERO network dependency at runtime.

Drop-in replacement for ingest.py. Same input schema, same graph.json output,
same Chroma storage — but embeddings are computed locally with
sentence-transformers (all-MiniLM-L6-v2, ~80MB, runs fine on CPU or the 2080).

Why this exists: the original pipeline calls the OpenAI API. Project Ash's
entire premise is no-internet operation. A memory cortex that dies when the
grid does is not an Ash component.

Setup (one-time, online):
    pip install sentence-transformers chromadb numpy

    # Pre-cache the model so runtime is fully offline:
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

Run (offline forever after):
    python ingest_local.py
    python ingest_local.py --no-chroma          # graph.json only
    python ingest_local.py --input my_logs.json

Input schema (identical to ingest.py):
[
  {
    "id": "conv_001",
    "title": "Cookie pricing and bakery stress",
    "created_at": "2026-04-09T10:15:00Z",
    "messages": [
      {"role": "user", "content": "...", "timestamp": "..."},
      {"role": "assistant", "content": "...", "timestamp": "..."}
    ]
  }
]

Output:
- data/graph.json  → {nodes: [...], links: [...]}  (feeds synapses.html)
- data/chroma/     → persistent vector store (unless --no-chroma)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# =========================
# Config
# =========================

DATA_DIR = Path("./data")
INPUT_PATH = DATA_DIR / "conversations.json"
CHROMA_PATH = DATA_DIR / "chroma"
OUTPUT_GRAPH_PATH = DATA_DIR / "graph.json"
COLLECTION_NAME = "conversation_map"

# Local model. all-MiniLM-L6-v2: 384-dim, fast, good enough for clustering.
# Upgrade path: "all-mpnet-base-v2" (768-dim, slower, better) — same code.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

# Graph pruning
TOP_K_NEIGHBORS = 4
MIN_SIMILARITY = 0.45  # MiniLM cosine scores run lower than OpenAI's — 0.72 would prune everything

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


def chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


# =========================
# Normalization
# =========================

def normalize_message(raw: Dict[str, Any]) -> Optional[Message]:
    content = clean_text(str(raw.get("content", "")))
    if not content:
        return None
    role = str(raw.get("role", "unknown")).strip().lower() or "unknown"
    return Message(role=role, content=content, timestamp=raw.get("timestamp"))


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
    """Placeholder. Swap in Verity (LM Studio, port 1234) later for real
    summaries — one local completion per conversation, still offline."""
    preview = full_text[:500].replace("\n", " ")
    if len(full_text) > 500:
        preview += " ..."
    return f"{title}: {preview}"


def extract_tags(title: str, full_text: str) -> List[str]:
    """Keyword tagger. Word-boundary matched — substring matching is how
    'scrape' triggers 'rape' filters and 'api' tags a conversation about
    'therapist'. Same bug class as the Servitor keyword router."""
    text = f"{title}\n{full_text}".lower()

    keyword_map = {
        "coding": ["javascript", "python", "bug", "code", "debug", "react", "api", "websocket"],
        "baking": ["cookie", "bread", "sourdough", "loaf", "bake", "starter"],
        "business": ["llc", "pricing", "profit", "customer", "brand", "business"],
        "legal": ["court", "custody", "motion", "filing", "contempt", "lawyer"],
        "family": ["son", "daughter", "baby", "family", "pregnant", "kids"],
        "health": ["doctor", "hospital", "pain", "migraine", "recovery"],
        "gaming": ["game", "gpu", "steam", "fps", "valheim", "apex"],
        "ash": ["verity", "servitor", "cartographer", "mortality", "rvc", "piper", "mesh", "cogitator"],
    }

    tags: List[str] = []
    for tag, keywords in keyword_map.items():
        pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
        if re.search(pattern, text):
            tags.append(tag)

    return tags[:6] if tags else ["general"]


def build_embedding_text(title: str, summary: str, tags: List[str], full_text: str) -> str:
    sample = full_text[:MAX_EMBEDDING_TEXT_CHARS]
    return clean_text(
        f"Title: {title}\n"
        f"Summary: {summary}\n"
        f"Tags: {', '.join(tags)}\n"
        f"Conversation:\n{sample}"
    )


def normalize_conversation(raw: Dict[str, Any]) -> ConversationRecord:
    conv_id = str(raw["id"])
    title = clean_text(str(raw.get("title", ""))) or f"Conversation {conv_id}"
    created_at = raw.get("created_at")

    messages = [m for m in (normalize_message(r) for r in raw.get("messages", [])) if m]
    messages.sort(key=lambda m: parse_iso(m.timestamp) or datetime.min)

    full_text = build_full_text(messages)
    summary = summarize_conversation(title, full_text)
    tags = extract_tags(title, full_text)

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
        embedding_text=build_embedding_text(title, summary, tags, full_text),
    )


# =========================
# Local embeddings
# =========================

def load_local_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "sentence-transformers not installed.\n"
            "Run once (online): pip install sentence-transformers\n"
            "Then pre-cache the model — see header of this file."
        )
    # local_files_only after first cache → guaranteed no network call
    cached = bool(os.environ.get("SYNAPSES_OFFLINE", ""))
    return SentenceTransformer(EMBEDDING_MODEL, local_files_only=cached)


def embed_all(model, texts: List[str]) -> np.ndarray:
    """Returns L2-normalized float32 matrix (n, dim)."""
    vectors: List[np.ndarray] = []
    for batch in chunked(texts, EMBED_BATCH_SIZE):
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.append(np.asarray(vecs, dtype=np.float32))
    return np.vstack(vectors)


# =========================
# Chroma store (optional)
# =========================

def upsert_chroma(records: List[ConversationRecord], embeddings: np.ndarray) -> None:
    import chromadb
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.upsert(
        ids=[r.id for r in records],
        documents=[r.embedding_text for r in records],
        metadatas=[
            {
                "title": r.title,
                "created_at": r.created_at or "",
                "summary": r.summary,
                "tags": ", ".join(r.tags),
                "message_count": r.message_count,
                "char_count": r.char_count,
            }
            for r in records
        ],
        embeddings=embeddings.tolist(),
    )


# =========================
# Graph building
# =========================

def build_nodes(records: List[ConversationRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "id": r.id,
            "label": r.title,
            "summary": r.summary,
            "tags": r.tags,
            "date": r.created_at,
            "val": max(1, min(30, r.message_count)),
            "message_count": r.message_count,
            "char_count": r.char_count,
        }
        for r in records
    ]


def build_edges(records: List[ConversationRecord], embeddings: np.ndarray) -> List[Dict[str, Any]]:
    """Vectorized: one matmul instead of a pure-Python O(n²·dim) loop.
    Embeddings are already L2-normalized, so dot product == cosine."""
    sim = embeddings @ embeddings.T  # (n, n)
    np.fill_diagonal(sim, -1.0)

    edges: List[Dict[str, Any]] = []
    seen: set = set()

    for i, source in enumerate(records):
        order = np.argsort(sim[i])[::-1][:TOP_K_NEIGHBORS]
        for j in order:
            score = float(sim[i, j])
            if score < MIN_SIMILARITY:
                break
            pair = tuple(sorted((source.id, records[j].id)))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "source": source.id,
                "target": records[j].id,
                "weight": round(score, 4),
                "relation_type": "semantic_match",
            })
    return edges


# =========================
# Main
# =========================

def run_ingest(input_path: Path, use_chroma: bool) -> None:
    raw_items = load_json(input_path)
    if not isinstance(raw_items, list):
        raise ValueError("Input file must be a JSON array of conversations.")

    records = [normalize_conversation(item) for item in raw_items]
    if not records:
        raise ValueError("No valid conversations found.")

    print(f"Normalized {len(records)} conversations. Loading local model...")
    model = load_local_model()

    print("Embedding (local, no network)...")
    embeddings = embed_all(model, [r.embedding_text for r in records])

    if use_chroma:
        print("Upserting to Chroma...")
        upsert_chroma(records, embeddings)

    print("Building graph...")
    payload = {
        "nodes": build_nodes(records),
        "links": build_edges(records, embeddings),
    }
    save_json(OUTPUT_GRAPH_PATH, payload)

    print("Ingest complete.")
    print(f"  Conversations: {len(records)}")
    print(f"  Links:         {len(payload['links'])}")
    print(f"  Graph output:  {OUTPUT_GRAPH_PATH}")
    if use_chroma:
        print(f"  Chroma path:   {CHROMA_PATH}")
    print("Open synapses.html and load the graph (or serve ./ and it auto-loads).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapses offline ingest")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--no-chroma", action="store_true", help="Skip vector store, write graph.json only")
    args = parser.parse_args()
    run_ingest(args.input, use_chroma=not args.no_chroma)
