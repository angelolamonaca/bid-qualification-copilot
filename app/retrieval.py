"""Permission-aware hybrid search: meaning (embeddings) + keywords (BM25), fused with RRF.

Access control happens BEFORE ranking. Chunks a persona may not read are never scored, so they
can never reach the model's context. Azure AI Search calls the same idea "security trimming".
"""

import math
import re
from collections import Counter

import numpy as np

from app.llm import LLM
from app.models import Chunk, Hit, LLMUsage, Persona, RetrievalTrace

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "has", "have",
             "in", "is", "it", "of", "on", "or", "our", "the", "this", "to", "we", "what", "which",
             "who", "with"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def can_read(persona: Persona, chunk: Chunk) -> bool:
    return "all" in chunk.access or bool(set(chunk.access) & set(persona.groups))


def embedding_text(chunk: Chunk) -> str:
    # Prefixing title and section gives each chunk context it would lack on its own.
    return f"{chunk.title} — {chunk.section}\n{chunk.text}"


class SearchIndex:
    def __init__(self, chunks: list[Chunk], llm: LLM):
        self.chunks = chunks
        self._llm = llm
        vectors, self.build_usage = llm.embed([embedding_text(c) for c in chunks])
        matrix = np.array(vectors, dtype=np.float32)
        self._vectors = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self._tokens = [tokenize(embedding_text(c)) for c in chunks]

    def search(
        self, query: str, persona: Persona, k: int = 6
    ) -> tuple[list[Chunk], RetrievalTrace, LLMUsage]:
        # 1. Security trimming and currency filter, before any scoring.
        allowed, no_permission, superseded = [], 0, 0
        for i, chunk in enumerate(self.chunks):
            if not can_read(persona, chunk):
                no_permission += 1
            elif chunk.status != "current":
                superseded += 1
            else:
                allowed.append(i)

        # 2. Meaning: cosine similarity between the query embedding and each chunk embedding.
        query_vectors, query_usage = self._llm.embed([query])
        q = np.array(query_vectors[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        similarity = {i: float(self._vectors[i] @ q) for i in allowed}
        vector_ranked = sorted(allowed, key=lambda i: -similarity[i])

        # 3. Keywords: BM25, with statistics computed only over what this persona may read.
        bm25 = self._bm25(tokenize(query), allowed)
        keyword_ranked = [i for i in sorted(allowed, key=lambda i: -bm25[i]) if bm25[i] > 0]

        # 4. Reciprocal Rank Fusion: a chunk ranked high by either method rises to the top.
        vector_rank = {i: rank for rank, i in enumerate(vector_ranked, start=1)}
        keyword_rank = {i: rank for rank, i in enumerate(keyword_ranked, start=1)}
        fused = {
            i: 1 / (60 + vector_rank[i]) + (1 / (60 + keyword_rank[i]) if i in keyword_rank else 0)
            for i in allowed
        }
        top = sorted(allowed, key=lambda i: -fused[i])[:k]

        trace = RetrievalTrace(
            query=query,
            persona_id=persona.id,
            searched_chunks=len(allowed),
            excluded_no_permission=no_permission,
            excluded_superseded=superseded,
            hits=[
                Hit(
                    chunk_id=self.chunks[i].chunk_id,
                    score=round(fused[i], 5),
                    similarity=round(similarity[i], 3),
                    vector_rank=vector_rank[i],
                    keyword_rank=keyword_rank.get(i),
                )
                for i in top
            ],
        )
        return [self.chunks[i] for i in top], trace, query_usage

    def _bm25(self, query_tokens: list[str], allowed: list[int], k1: float = 1.5,
              b: float = 0.75) -> dict[int, float]:
        doc_freq = Counter(t for i in allowed for t in set(self._tokens[i]))
        avg_len = sum(len(self._tokens[i]) for i in allowed) / max(len(allowed), 1)
        scores = {}
        for i in allowed:
            term_freq, length, score = Counter(self._tokens[i]), len(self._tokens[i]), 0.0
            for t in query_tokens:
                if t in term_freq:
                    idf = math.log(1 + (len(allowed) - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5))
                    tf = term_freq[t]
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
            scores[i] = score
        return scores
