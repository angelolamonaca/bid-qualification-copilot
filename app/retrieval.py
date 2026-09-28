"""Permission-aware hybrid search: meaning (embeddings) + keywords (BM25), fused with RRF.

Access control happens BEFORE ranking. Chunks a persona may not read are never scored, so they
can never reach the model's context. Azure AI Search calls the same idea "security trimming".

Terms used below:
- embedding: a list of 1,536 numbers representing a text's meaning. Similar meaning gives
  vectors pointing in similar directions.
- cosine similarity: how closely two vectors point the same way (1 = same meaning, 0 = unrelated).
- BM25: the classic keyword-ranking formula used by search engines.
- RRF (Reciprocal Rank Fusion): a simple way to merge two ranked lists into one.
"""

import math
import re
from collections import Counter

import numpy as np

from app.llm import LLM
from app.models import Chunk, Hit, LLMUsage, Persona, RetrievalTrace

# Very common words that carry no meaning for keyword search. A `set` gives fast "is x in it?".
STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "has", "have",
             "in", "is", "it", "of", "on", "or", "our", "the", "this", "to", "we", "what", "which",
             "who", "with"}


def tokenize(text: str) -> list[str]:
    """'Who approves RVA-2026-014?' -> ['approves', 'rva', '2026', '014'] (lowercase words)."""
    # re.findall returns every match of the pattern: runs of letters/digits. The list
    # comprehension keeps only the words that are not stopwords, like .filter() in JS.
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def can_read(persona: Persona, chunk: Chunk) -> bool:
    """True if the chunk is public to all employees, or shares a group with the persona."""
    # `set(a) & set(b)` = the groups both have in common; bool(...) is True if that is non-empty.
    return "all" in chunk.access or bool(set(chunk.access) & set(persona.groups))


def embedding_text(chunk: Chunk) -> str:
    # Prefixing title and section gives each chunk context it would lack on its own
    # (a section called "Team" means little without knowing which project it belongs to).
    return f"{chunk.title} — {chunk.section}\n{chunk.text}"


class SearchIndex:
    def __init__(self, chunks: list[Chunk], llm: LLM):
        """Build the index once at startup: one embedding call for all chunks."""
        self.chunks = chunks
        self._llm = llm
        vectors, self.build_usage = llm.embed([embedding_text(c) for c in chunks])
        # numpy stores all vectors as one matrix (one row per chunk) for fast math. Dividing
        # each row by its length ("normalizing") means a later dot product directly gives the
        # cosine similarity.
        matrix = np.array(vectors, dtype=np.float32)
        self._vectors = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self._tokens = [tokenize(embedding_text(c)) for c in chunks]  # words per chunk, for BM25

    def search(
        self, query: str, persona: Persona, k: int = 6
    ) -> tuple[list[Chunk], RetrievalTrace, LLMUsage]:
        """Return the k best chunks this persona may read, plus a trace explaining the ranking."""
        # 1. Security trimming and currency filter, BEFORE any scoring. We keep the positions
        #    (indexes) of allowed chunks and count why the others were dropped, for the trace.
        allowed, no_permission, superseded = [], 0, 0
        for i, chunk in enumerate(self.chunks):  # i = position, chunk = the item
            if not can_read(persona, chunk):
                no_permission += 1
            elif chunk.status != "current":
                superseded += 1
            else:
                allowed.append(i)

        # 2. Meaning: embed the question, then compare it with every allowed chunk.
        query_vectors, query_usage = self._llm.embed([query])
        q = np.array(query_vectors[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        # `@` is matrix multiplication; for two normalized vectors it gives the cosine similarity.
        similarity = {i: float(self._vectors[i] @ q) for i in allowed}  # {chunk index: similarity}
        # Python note: `sorted(..., key=f)` sorts by f(item); `-similarity[i]` puts the highest
        # first, like list.sort((a, b) => sim[b] - sim[a]) in JS. `lambda` = an arrow function.
        vector_ranked = sorted(allowed, key=lambda i: -similarity[i])

        # 3. Keywords: BM25, with statistics computed only over what this persona may read.
        bm25 = self._bm25(tokenize(query), allowed)
        # Only chunks sharing at least one keyword with the question get a keyword rank.
        keyword_ranked = [i for i in sorted(allowed, key=lambda i: -bm25[i]) if bm25[i] > 0]

        # 4. Reciprocal Rank Fusion. Each list contributes 1 / (60 + rank): rank 1 -> 1/61,
        #    rank 10 -> 1/70. Summing rewards chunks that rank high in either list, without
        #    having to compare similarity numbers with BM25 numbers (they use different scales).
        vector_rank = {i: rank for rank, i in enumerate(vector_ranked, start=1)}
        keyword_rank = {i: rank for rank, i in enumerate(keyword_ranked, start=1)}
        fused = {
            i: 1 / (60 + vector_rank[i]) + (1 / (60 + keyword_rank[i]) if i in keyword_rank else 0)
            for i in allowed
        }
        top = sorted(allowed, key=lambda i: -fused[i])[:k]  # [:k] = the first k items

        # The trace explains the result: how much was searched or excluded, and each hit's ranks.
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
                    keyword_rank=keyword_rank.get(i),  # .get returns None if there's no rank
                )
                for i in top
            ],
        )
        return [self.chunks[i] for i in top], trace, query_usage

    def _bm25(self, query_tokens: list[str], allowed: list[int], k1: float = 1.5,
              b: float = 0.75) -> dict[int, float]:
        """BM25 keyword score of every allowed chunk for the query words.

        In plain words: a chunk scores points for each query word it contains.
        - Rare words are worth more than common ones (idf: "inverse document frequency").
        - Repeating a word helps, but less each time (k1 limits this).
        - Long chunks are penalized a little, so they don't win just by containing more words (b).
        """
        # Counter counts things: in how many allowed chunks does each word appear?
        doc_freq = Counter(t for i in allowed for t in set(self._tokens[i]))
        avg_len = sum(len(self._tokens[i]) for i in allowed) / max(len(allowed), 1)
        scores = {}
        for i in allowed:
            term_freq, length, score = Counter(self._tokens[i]), len(self._tokens[i]), 0.0
            for t in query_tokens:
                if t in term_freq:
                    idf = math.log(1 + (len(allowed) - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5))
                    tf = term_freq[t]  # how often the word occurs in this chunk
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
            scores[i] = score
        return scores
