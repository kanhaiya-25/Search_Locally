"""Keyword (lexical) retrieval baseline using BM25.

Used on its own as the "keyword baseline" in the evaluation module, and
combined with semantic similarity for hybrid retrieval (see
search_service.py). BM25 is rebuilt from all chunks in memory on each
call to `build`; for the dataset sizes this project targets (hundreds
to low-thousands of chunks) this is fast enough to run per-request, and
avoids maintaining a second persistent index that could drift out of
sync with FAISS/SQLite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class KeywordIndex:
    """A BM25 index over a fixed snapshot of chunks, keyed by faiss_index."""

    chunk_ids: List[int]
    bm25: BM25Okapi
    corpus_token_sets: List[set]

    def search(self, query: str, top_k: int) -> Dict[int, float]:
        """Return {chunk_faiss_index: normalized_score} for the top_k
        matches. Scores are min-max normalized to [0, 1] within this
        result set so they can be linearly combined with cosine
        similarity scores in hybrid mode.

        A chunk is only considered a match if it shares at least one
        token with the query. Raw BM25 scores cannot be filtered by
        "> 0" for this: on small corpora (a handful of personal
        documents, as this project targets) BM25's IDF term can go
        negative for words that appear in most/all documents, which
        would otherwise incorrectly discard genuine matches.
        """
        if not self.chunk_ids:
            return {}
        tokens = _tokenize(query)
        if not tokens:
            return {}
        query_token_set = set(tokens)
        raw_scores = self.bm25.get_scores(tokens)

        candidates = [
            (cid, score)
            for cid, score, token_set in zip(self.chunk_ids, raw_scores, self.corpus_token_sets)
            if query_token_set & token_set
        ]
        ranked = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        if not ranked:
            return {}
        max_score = max(s for _, s in ranked)
        min_score = min(s for _, s in ranked)
        span = max_score - min_score or 1.0
        return {cid: (s - min_score) / span for cid, s in ranked}


def build_keyword_index(chunk_ids: List[int], texts: List[str]) -> KeywordIndex:
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else BM25Okapi([[""]])
    corpus_token_sets = [set(tokens) for tokens in tokenized_corpus]
    return KeywordIndex(chunk_ids=chunk_ids, bm25=bm25, corpus_token_sets=corpus_token_sets)
