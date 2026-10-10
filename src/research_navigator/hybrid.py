"""Final capstone hybrid functions, carried forward from the tested course implementation.

Use navigator.py as the entry point. Historical experimental records are in evidence/.
"""
from __future__ import annotations

import argparse

import gzip

import hashlib

import heapq

import json

import math

import os

import re

import statistics

import sys

import time

import urllib.error

import urllib.request

from collections import Counter

from dataclasses import asdict, dataclass

from datetime import datetime

from pathlib import Path

from typing import Iterable, Sequence

import numpy as np

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - friendly setup failure
    raise SystemExit("Missing dependency: install pypdf with 'pip install pypdf'.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent

SCENARIO = "research_papers"

DEFAULT_CORPUS_DIR = SCRIPT_DIR.parents[1] / "data" / "papers"

CACHE_PATH = SCRIPT_DIR.parents[1] / ".cache" / "chunks.json.gz"

EMBEDDING_CACHE_PATH = SCRIPT_DIR.parents[1] / ".cache" / "embeddings.npz"

LOG_PATH = SCRIPT_DIR / "checkpoint_2_1_hybrid_retrieval.log"

RESULTS_PATH = SCRIPT_DIR / "checkpoint_2_1_hybrid_retrieval_results.json"

CHUNK_WORDS = 350

CHUNK_OVERLAP = 70

TOP_K = 5

MAX_RESULTS_PER_PAPER = 2

BM25_K1 = 1.5

BM25_B = 0.75

TITLE_BOOST = 1.35

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = "openai/text-embedding-3-small"

EMBEDDING_DIMENSIONS = 512

EMBEDDING_BATCH_SIZE = 64

LLM_MODEL = "openai/gpt-5.4-mini"

LLM_TEMPERATURE = 0.2

LLM_MAX_TOKENS = 1800

FUSION_CANDIDATES = 80

RRF_K = 60

KEYWORD_WEIGHT = 0.50

SEMANTIC_WEIGHT = 0.50

ANSWER_CONTEXT_CHARS_PER_CHUNK = 6500

REPRESENTATIVE_QUERIES = [
    (
        "What is the main finding of 'The Counterfeit Conundrum' about whether "
        "code language models recognize their own incorrect generations?"
    ),
    (
        "How does the sketching approach to program synthesis use partial programs "
        "with holes and constraints?"
    ),
    (
        "Which papers study program synthesis from input-output examples, and what "
        "learning or search methods do they use?"
    ),
    (
        "What recent directions from 2023 through 2025 connect language models with "
        "program synthesis or code generation?"
    ),
]

@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with enough metadata to verify the source."""

    chunk_id: str
    filename: str
    title: str
    page: int
    part: int
    text: str

@dataclass(frozen=True)
class SearchResult:
    """A scored BM25 hit returned to the user and evidence log."""

    rank: int
    score: float
    chunk_id: str
    filename: str
    title: str
    page: int
    excerpt: str

@dataclass(frozen=True)
class HybridSearchResult:
    """One fused retrieval hit with transparent component scores and ranks."""

    rank: int
    fused_score: float
    keyword_rank: int | None
    keyword_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    chunk_id: str
    filename: str
    title: str
    page: int
    excerpt: str

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.IGNORECASE)

SPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """Normalize PDF extraction whitespace without changing the words."""

    return SPACE_RE.sub(" ", text.replace("\x00", " ")).strip()

def tokenize(text: str) -> list[str]:
    """Lowercase lexical tokens used by the BM25 baseline."""

    return TOKEN_RE.findall(text.lower())

def readable_title(reader: PdfReader, filename: str, first_page_text: str) -> str:
    """Choose a useful title from PDF metadata, first-page text, or filename."""

    metadata_title = ""
    try:
        metadata_title = normalize_text(str((reader.metadata or {}).get("/Title", "")))
    except Exception:
        metadata_title = ""
    metadata_lower = metadata_title.lower()
    if (
        metadata_title
        and metadata_lower not in {"untitled", "none"}
        and not metadata_lower.startswith(("proceedings of", "findings of"))
    ):
        return metadata_title[:300]

    # The first substantial lines of scholarly PDFs usually contain the title.
    raw_lines = [normalize_text(line) for line in first_page_text.splitlines()]
    for index, line in enumerate(raw_lines[:30]):
        lower = line.lower()
        if not (
            4 <= len(line.split()) <= 30
            and len(line) <= 220
            and not re.fullmatch(r"\d+", line)
            and "@" not in line
            and not lower.startswith(
                (
                    "arxiv:",
                    "proceedings of",
                    "findings of",
                    "copyright",
                    "august ",
                    "abstract",
                )
            )
            and " pages " not in f" {lower} "
        ):
            continue
        title_lines = [line]
        # Join a wrapped title line, but stop before typical author/affiliation lines.
        if index + 1 < len(raw_lines):
            following = raw_lines[index + 1]
            following_lower = following.lower()
            if (
                following
                and len(following.split()) <= 20
                and len(line) + len(following) <= 280
                and "@" not in following
                and not re.search(r"[⋆*{}]|\b(university|institute|laboratory|abstract)\b", following_lower)
                and not re.search(r"\d", following)
            ):
                title_lines.append(following)
        return " ".join(title_lines)
    return filename.removesuffix(".pdf").replace("_", " ")

def chunk_page(words: Sequence[str], size: int, overlap: int) -> Iterable[list[str]]:
    """Yield overlapping word windows from one page."""

    if not words:
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = list(words[start : start + size])
        if piece:
            yield piece
        if start + size >= len(words):
            break

def corpus_signature(pdf_paths: Sequence[Path]) -> dict[str, int]:
    """A small signature that invalidates the cache if the corpus changes."""

    return {
        "pdf_count": len(pdf_paths),
        "total_bytes": sum(path.stat().st_size for path in pdf_paths),
        "latest_mtime_ns": max((path.stat().st_mtime_ns for path in pdf_paths), default=0),
    }

def extract_corpus(pdf_paths: Sequence[Path]) -> tuple[list[Chunk], list[dict[str, str]]]:
    """Extract all readable pages and split each page into overlapping chunks."""

    chunks: list[Chunk] = []
    failures: list[dict[str, str]] = []
    total = len(pdf_paths)

    for paper_index, pdf_path in enumerate(pdf_paths, 1):
        print(f"[{paper_index:3d}/{total}] extracting {pdf_path.name}", flush=True)
        try:
            reader = PdfReader(str(pdf_path), strict=False)
            first_page_raw = ""
            if reader.pages:
                try:
                    first_page_raw = reader.pages[0].extract_text() or ""
                except Exception:
                    first_page_raw = ""
            title = readable_title(reader, pdf_path.name, first_page_raw)

            for page_number, page in enumerate(reader.pages, 1):
                try:
                    page_text = normalize_text(page.extract_text() or "")
                except Exception as exc:
                    failures.append(
                        {
                            "filename": pdf_path.name,
                            "location": f"page {page_number}",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                words = page_text.split()
                if len(words) < 20:
                    continue
                for part, piece in enumerate(
                    chunk_page(words, CHUNK_WORDS, CHUNK_OVERLAP), start=1
                ):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{pdf_path.stem}:p{page_number}:c{part}",
                            filename=pdf_path.name,
                            title=title,
                            page=page_number,
                            part=part,
                            text=" ".join(piece),
                        )
                    )
        except Exception as exc:
            failures.append(
                {
                    "filename": pdf_path.name,
                    "location": "document",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return chunks, failures

def save_cache(
    chunks: Sequence[Chunk], failures: Sequence[dict[str, str]], signature: dict[str, int]
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"chunk_words": CHUNK_WORDS, "chunk_overlap": CHUNK_OVERLAP},
        "corpus_signature": signature,
        "failures": list(failures),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    with gzip.open(CACHE_PATH, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

def load_cache(signature: dict[str, int]) -> tuple[list[Chunk], list[dict[str, str]]] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        with gzip.open(CACHE_PATH, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        settings = payload.get("settings", {})
        if (
            payload.get("schema_version") != 1
            or payload.get("corpus_signature") != signature
            or settings.get("chunk_words") != CHUNK_WORDS
            or settings.get("chunk_overlap") != CHUNK_OVERLAP
        ):
            return None
        return (
            [Chunk(**item) for item in payload.get("chunks", [])],
            payload.get("failures", []),
        )
    except (OSError, EOFError, json.JSONDecodeError, TypeError, KeyError):
        return None

def prepare_chunks(
    corpus_dir: Path, rebuild_cache: bool = False, smoke_test: bool = False
) -> tuple[list[Chunk], list[dict[str, str]], int]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Research-paper folder not found: {corpus_dir}")
    pdf_paths = sorted(corpus_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if smoke_test:
        pdf_paths = pdf_paths[:3]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {corpus_dir}")

    signature = corpus_signature(pdf_paths)
    if not rebuild_cache and not smoke_test:
        cached = load_cache(signature)
        if cached is not None:
            chunks, failures = cached
            print(f"Loaded {len(chunks):,} cached chunks from {CACHE_PATH.name}.")
            return chunks, failures, len(pdf_paths)

    chunks, failures = extract_corpus(pdf_paths)
    if not chunks:
        raise RuntimeError("PDF extraction produced no searchable text chunks.")
    if not smoke_test:
        save_cache(chunks, failures, signature)
        print(f"Saved reusable extraction cache: {CACHE_PATH.name}")
    return chunks, failures, len(pdf_paths)

class BM25Retriever:
    """Small dependency-free BM25 implementation with source diversification."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk.")
        self.chunks = list(chunks)
        self.doc_tokens: list[list[str]] = []
        self.term_frequencies: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            self.doc_tokens.append(tokens)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            document_frequency.update(frequencies.keys())

        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = statistics.fmean(self.doc_lengths) if self.doc_lengths else 1.0
        count = len(self.chunks)
        self.idf = {
            term: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }
        self.title_tokens = [set(tokenize(f"{c.title} {c.filename}")) for c in self.chunks]

    def _score(self, index: int, query_tokens: Sequence[str]) -> float:
        frequencies = self.term_frequencies[index]
        doc_length = self.doc_lengths[index]
        score = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * doc_length / self.average_length
            )
            score += self.idf.get(term, 0.0) * frequency * (BM25_K1 + 1.0) / denominator
        title_matches = len(set(query_tokens) & self.title_tokens[index])
        if title_matches:
            score += TITLE_BOOST * title_matches
        return score

    def search(
        self,
        query: str,
        k: int = TOP_K,
        max_per_paper: int = MAX_RESULTS_PER_PAPER,
    ) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        # Keep extra candidates so diversification can suppress duplicate chunks.
        candidates = heapq.nlargest(
            max(k * 10, k),
            ((self._score(i, query_tokens), i) for i in range(len(self.chunks))),
        )
        results: list[SearchResult] = []
        paper_counts: Counter[str] = Counter()
        for score, index in candidates:
            if score <= 0:
                continue
            chunk = self.chunks[index]
            if paper_counts[chunk.filename] >= max_per_paper:
                continue
            paper_counts[chunk.filename] += 1
            excerpt = chunk.text[:420].rstrip()
            if len(chunk.text) > 420:
                excerpt += "..."
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    score=round(score, 3),
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    title=chunk.title,
                    page=chunk.page,
                    excerpt=excerpt,
                )
            )
            if len(results) >= k:
                break
        return results

def load_openrouter_key() -> str:
    """Read only the explicitly configured environment variable."""

    existing = os.getenv("OPENROUTER_API_KEY", "").strip()
    if existing:
        return existing
    raise RuntimeError(
        "Set OPENROUTER_API_KEY in your process environment before using live mode."
    )

def openrouter_post(path: str, body: dict[str, object], retries: int = 6) -> dict[str, object]:
    """POST JSON with bounded retry/backoff for rate limits and transient errors."""

    request_body = json.dumps(body).encode("utf-8")
    url = f"{OPENROUTER_BASE_URL}{path}"
    headers = {
        "Authorization": f"Bearer {load_openrouter_key()}",
        "Content-Type": "application/json",
        "User-Agent": "MIT-xPRO-Capstone-Checkpoint-2.1-Hybrid/1.0",
    }
    for attempt in range(retries):
        request = urllib.request.Request(url, data=request_body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError("OpenRouter returned a non-object JSON response.")
            return payload
        except urllib.error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")[:1200]
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt + 1 >= retries:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {response_text}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        delay = min(30, 2 ** attempt)
        print(f"  transient API error; retrying in {delay}s...", flush=True)
        time.sleep(delay)
    raise RuntimeError("OpenRouter request exhausted its retries.")

def embed_texts(texts: Sequence[str]) -> tuple[np.ndarray, dict[str, float]]:
    """Return 512-dimensional dense vectors and usage for one API batch."""

    payload = openrouter_post(
        "/embeddings",
        {
            "model": EMBEDDING_MODEL,
            "input": list(texts),
            "dimensions": EMBEDDING_DIMENSIONS,
            "encoding_format": "float",
        },
    )
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError("Embedding response count did not match the request count.")
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    vectors = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
    if vectors.shape != (len(texts), EMBEDDING_DIMENSIONS):
        raise RuntimeError(f"Unexpected embedding matrix shape: {vectors.shape}")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return vectors, {
        "tokens": float(usage.get("total_tokens", usage.get("prompt_tokens", 0)) or 0),
        "cost": float(usage.get("cost", 0) or 0),
    }

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)

def semantic_cache_signature(chunks: Sequence[Chunk]) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    digest.update(str(EMBEDDING_DIMENSIONS).encode("ascii"))
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

def embedding_input(chunk: Chunk) -> str:
    return (
        f"Paper title: {chunk.title}\nSource file: {chunk.filename}\n"
        f"Page: {chunk.page}\nPassage: {chunk.text}"
    )

def load_or_build_semantic_index(
    chunks: Sequence[Chunk], rebuild: bool = False, smoke_test: bool = False
) -> tuple[np.ndarray, dict[str, object]]:
    """Load cached document vectors or build them with resumable batching."""

    signature = semantic_cache_signature(chunks)
    if not rebuild and not smoke_test and EMBEDDING_CACHE_PATH.exists():
        try:
            with np.load(EMBEDDING_CACHE_PATH, allow_pickle=False) as cached:
                cached_signature = str(cached["signature"].item())
                cached_model = str(cached["model"].item())
                cached_dimensions = int(cached["dimensions"].item())
                vectors = np.asarray(cached["embeddings"], dtype=np.float32)
            if (
                cached_signature == signature
                and cached_model == EMBEDDING_MODEL
                and cached_dimensions == EMBEDDING_DIMENSIONS
                and vectors.shape == (len(chunks), EMBEDDING_DIMENSIONS)
            ):
                print(f"Loaded {len(chunks):,} dense vectors from {EMBEDDING_CACHE_PATH.name}.")
                return normalize_rows(vectors), {
                    "source": "cache",
                    "embedding_seconds": 0.0,
                    "embedding_tokens": 0,
                    "embedding_cost": 0.0,
                }
        except (OSError, ValueError, KeyError):
            print("Existing semantic cache is invalid; rebuilding it.")

    partial_data = EMBEDDING_CACHE_PATH.with_name(
        EMBEDDING_CACHE_PATH.stem + ".partial.npy"
    )
    partial_meta = EMBEDDING_CACHE_PATH.with_name(
        EMBEDDING_CACHE_PATH.stem + ".partial.json"
    )
    if rebuild:
        for path in (EMBEDDING_CACHE_PATH, partial_data, partial_meta):
            if path.exists():
                path.unlink()

    start_index = 0
    if smoke_test:
        vectors = np.empty((len(chunks), EMBEDDING_DIMENSIONS), dtype=np.float32)
    elif partial_data.exists() and partial_meta.exists():
        try:
            meta = json.loads(partial_meta.read_text(encoding="utf-8"))
            if (
                meta.get("signature") == signature
                and meta.get("shape") == [len(chunks), EMBEDDING_DIMENSIONS]
            ):
                vectors = np.lib.format.open_memmap(partial_data, mode="r+")
                start_index = int(meta.get("next_index", 0))
                print(f"Resuming semantic indexing at chunk {start_index:,}.")
            else:
                raise ValueError("partial cache signature mismatch")
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            partial_data.unlink(missing_ok=True)
            partial_meta.unlink(missing_ok=True)
            vectors = np.lib.format.open_memmap(
                partial_data,
                mode="w+",
                dtype=np.float32,
                shape=(len(chunks), EMBEDDING_DIMENSIONS),
            )
    else:
        vectors = np.lib.format.open_memmap(
            partial_data,
            mode="w+",
            dtype=np.float32,
            shape=(len(chunks), EMBEDDING_DIMENSIONS),
        )

    build_start = time.perf_counter()
    total_tokens = 0.0
    total_cost = 0.0
    for start in range(start_index, len(chunks), EMBEDDING_BATCH_SIZE):
        end = min(start + EMBEDDING_BATCH_SIZE, len(chunks))
        print(f"Embedding chunks {start + 1:,}-{end:,} of {len(chunks):,}...", flush=True)
        batch_vectors, usage = embed_texts([embedding_input(c) for c in chunks[start:end]])
        vectors[start:end] = batch_vectors
        if isinstance(vectors, np.memmap):
            vectors.flush()
        total_tokens += usage["tokens"]
        total_cost += usage["cost"]
        if not smoke_test:
            partial_meta.write_text(
                json.dumps(
                    {
                        "signature": signature,
                        "shape": [len(chunks), EMBEDDING_DIMENSIONS],
                        "next_index": end,
                    }
                ),
                encoding="utf-8",
            )

    normalized = normalize_rows(np.array(vectors, dtype=np.float32, copy=True))
    elapsed = round(time.perf_counter() - build_start, 2)
    if not smoke_test:
        if isinstance(vectors, np.memmap):
            vectors.flush()
            vectors._mmap.close()
        np.savez_compressed(
            EMBEDDING_CACHE_PATH,
            embeddings=normalized,
            signature=np.asarray(signature),
            model=np.asarray(EMBEDDING_MODEL),
            dimensions=np.asarray(EMBEDDING_DIMENSIONS),
        )
        partial_data.unlink(missing_ok=True)
        partial_meta.unlink(missing_ok=True)
        print(f"Saved semantic cache: {EMBEDDING_CACHE_PATH.name}")
    return normalized, {
        "source": "api_build",
        "embedding_seconds": elapsed,
        "embedding_tokens": int(total_tokens),
        "embedding_cost": round(total_cost, 8),
    }

class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        keyword_retriever: BM25Retriever,
        semantic_embeddings: np.ndarray,
    ) -> None:
        self.chunks = list(chunks)
        self.keyword = keyword_retriever
        self.semantic_embeddings = semantic_embeddings
        self.index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}

    def search(
        self, query: str, k: int = TOP_K
    ) -> tuple[list[HybridSearchResult], dict[str, float]]:
        keyword_hits = self.keyword.search(
            query,
            k=FUSION_CANDIDATES,
            max_per_paper=FUSION_CANDIDATES,
        )
        query_vector, usage = embed_texts([query])
        query_vector = normalize_rows(query_vector)[0]
        semantic_scores = self.semantic_embeddings @ query_vector
        candidate_count = min(FUSION_CANDIDATES, len(self.chunks))
        candidate_indices = np.argpartition(semantic_scores, -candidate_count)[-candidate_count:]
        semantic_indices = candidate_indices[
            np.argsort(semantic_scores[candidate_indices])[::-1]
        ].tolist()

        fused: dict[str, dict[str, float | int | None]] = {}
        for rank, hit in enumerate(keyword_hits, 1):
            fused[hit.chunk_id] = {
                "keyword_rank": rank,
                "keyword_score": hit.score,
                "semantic_rank": None,
                "semantic_score": None,
                "fused_score": KEYWORD_WEIGHT / (RRF_K + rank),
            }
        for rank, index in enumerate(semantic_indices, 1):
            chunk_id = self.chunks[index].chunk_id
            record = fused.setdefault(
                chunk_id,
                {
                    "keyword_rank": None,
                    "keyword_score": None,
                    "semantic_rank": None,
                    "semantic_score": None,
                    "fused_score": 0.0,
                },
            )
            record["semantic_rank"] = rank
            record["semantic_score"] = float(semantic_scores[index])
            record["fused_score"] = float(record["fused_score"] or 0.0) + (
                SEMANTIC_WEIGHT / (RRF_K + rank)
            )

        ordered = sorted(
            fused.items(), key=lambda item: float(item[1]["fused_score"] or 0.0), reverse=True
        )
        paper_counts: Counter[str] = Counter()
        results: list[HybridSearchResult] = []
        for chunk_id, scores in ordered:
            chunk = self.chunks[self.index_by_id[chunk_id]]
            if paper_counts[chunk.filename] >= MAX_RESULTS_PER_PAPER:
                continue
            paper_counts[chunk.filename] += 1
            excerpt = chunk.text[:900].rstrip()
            if len(chunk.text) > 900:
                excerpt += "..."
            results.append(
                HybridSearchResult(
                    rank=len(results) + 1,
                    fused_score=round(float(scores["fused_score"] or 0.0), 8),
                    keyword_rank=int(scores["keyword_rank"]) if scores["keyword_rank"] else None,
                    keyword_score=(
                        round(float(scores["keyword_score"]), 3)
                        if scores["keyword_score"] is not None
                        else None
                    ),
                    semantic_rank=(
                        int(scores["semantic_rank"]) if scores["semantic_rank"] else None
                    ),
                    semantic_score=(
                        round(float(scores["semantic_score"]), 6)
                        if scores["semantic_score"] is not None
                        else None
                    ),
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    title=chunk.title,
                    page=chunk.page,
                    excerpt=excerpt,
                )
            )
            if len(results) >= k:
                break
        return results, usage

def grounded_answer(
    query: str, results: Sequence[HybridSearchResult], chunks_by_id: dict[str, Chunk]
) -> tuple[str, dict[str, object]]:
    """Generate a detailed answer using only the retrieved hybrid evidence."""

    contexts: list[str] = []
    for result in results:
        chunk = chunks_by_id[result.chunk_id]
        passage = chunk.text[:ANSWER_CONTEXT_CHARS_PER_CHUNK]
        contexts.append(
            f"[{result.rank}] {chunk.title}\n"
            f"Source: {chunk.filename}, page {chunk.page}\n"
            f"Passage: {passage}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful research-paper navigator. Answer using ONLY the retrieved passages. "
                "Write a detailed, well-organized response that directly addresses every part of the "
                "question. Cite factual claims inline with bracketed passage numbers such as [1] or "
                "[2]. Distinguish what the passages explicitly establish from reasonable synthesis. "
                "Use short direct quotations only when they materially support the answer. If the "
                "retrieved evidence is incomplete, say exactly what cannot be concluded instead of "
                "guessing. End with a 'Sources used' list giving paper title, filename, and page."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nRetrieved passages:\n\n" + "\n\n".join(contexts),
        },
    ]
    payload = openrouter_post(
        "/chat/completions",
        {
            "model": LLM_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "messages": messages,
        },
    )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat completion returned no choices.")
    message = choices[0].get("message", {})
    answer = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Chat completion returned an empty answer.")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return answer.strip(), {
        "model": payload.get("model", LLM_MODEL),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cost": float(usage.get("cost", 0) or 0),
    }

def my_representative_queries() -> list[str]:
    return list(REPRESENTATIVE_QUERIES)

def print_result(result: HybridSearchResult) -> None:
    print(
        f"  {result.rank}. fused={result.fused_score:.8f} | "
        f"keyword_rank={result.keyword_rank} semantic_rank={result.semantic_rank} | "
        f"{result.title} | {result.filename} | page {result.page}"
    )
    print(f"     {result.excerpt}")
