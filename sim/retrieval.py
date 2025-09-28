from __future__ import annotations
import hashlib, json, os
from typing import Any, Dict, List, Tuple


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _lexical_tokens(s: str) -> set[str]:
    import re
    return set([t for t in re.findall(r"[A-Za-z0-9]+", (s or "").lower()) if len(t) >= 3])


class EmbeddingIndex:
    """Lightweight embedding index with pluggable backends and disk cache.

    Backends:
    - 'lexical': token overlap cosine (no deps)
    - 'st': sentence-transformers (optional)
    - 'openai': OpenAI embeddings (optional; needs OPENAI_API_KEY)
    """

    def __init__(self, backend: str = "lexical", model: str | None = None, cache_dir: str = ".cache/emb"):
        self.backend = backend
        self.model = model or (
            "sentence-transformers/all-MiniLM-L6-v2" if backend == "st" else ("text-embedding-3-small" if backend == "openai" else "lexical")
        )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._st_model = None
        self._openai_client = None

    # ---------- Embeddings ----------
    def _embed_lex(self, texts: List[str]) -> List[Dict[str, Any]]:
        # Represent text as token set; embed as dict {token:1}
        out = []
        for t in texts:
            toks = _lexical_tokens(t)
            out.append({"toks": toks})
        return out

    def _embed_st(self, texts: List[str]) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            # Fallback to lexical if not available
            return []
        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model)
        return self._st_model.encode(texts, normalize_embeddings=True).tolist()

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return []
        if self._openai_client is None:
            self._openai_client = OpenAI()
        embs = self._openai_client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in embs.data]

    def _cache_path(self, text: str) -> str:
        h = _hash(f"{self.backend}:{self.model}:{text}")
        return os.path.join(self.cache_dir, f"{h}.json")

    def embed(self, texts: List[str]) -> List[Any]:
        if self.backend == "lexical":
            return self._embed_lex(texts)
        # cache per-text for heavier backends
        outs: List[Any] = []
        misses: List[int] = []
        inputs: List[str] = []
        for i, t in enumerate(texts):
            p = self._cache_path(t)
            if os.path.exists(p):
                try:
                    outs.append(json.load(open(p, "r", encoding="utf-8")))
                    continue
                except Exception:
                    pass
            misses.append(i)
            inputs.append(t)
            outs.append(None)
        if misses:
            if self.backend == "st":
                em = self._embed_st(inputs)
            elif self.backend == "openai":
                em = self._embed_openai(inputs)
            else:
                em = []
            # If embedding failed (lib missing), downgrade to lexical
            if not em:
                return self._embed_lex(texts)
            # write back
            for idx, vec in zip(misses, em):
                outs[idx] = vec
                try:
                    json.dump(vec, open(self._cache_path(texts[idx]), "w", encoding="utf-8"))
                except Exception:
                    pass
        return outs

    # ---------- Similarity & Selection ----------
    def _cosine(self, a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)) or 1.0
        nb = math.sqrt(sum(y*y for y in b)) or 1.0
        return dot / (na * nb)

    def _lex_sim(self, qtoks: set[str], etoks: set[str]) -> float:
        inter = len(qtoks & etoks)
        denom = (len(qtoks) + len(etoks) - inter) or 1
        return inter / denom

    def knn(self, query: str, examples: List[Dict[str, Any]], k: int = 4, diverse: bool = False, mmr_lambda: float = 0.5) -> List[Dict[str, Any]]:
        texts = []
        for ex in examples:
            st = str(ex.get("stem") or ex.get("question") or "")
            opts = ex.get("options") or []
            if isinstance(opts, list) and opts:
                st += "\n" + " | ".join([str(o) for o in opts])
            texts.append(st)
        if not texts:
            return []
        if self.backend == "lexical":
            qtok = _lexical_tokens(query)
            sims: List[Tuple[float, int]] = []
            for i, t in enumerate(texts):
                etok = _lexical_tokens(t)
                sims.append((self._lex_sim(qtok, etok), i))
        else:
            ex_emb = self.embed(texts)
            q_emb = self.embed([query])[0]
            # If embed fallback returned lexical dicts
            if isinstance(q_emb, dict):
                qtok = q_emb.get("toks") or _lexical_tokens(query)
                sims = []
                for i, e in enumerate(ex_emb):
                    etok = e.get("toks") if isinstance(e, dict) else _lexical_tokens(texts[i])
                    sims.append((self._lex_sim(qtok, etok), i))
            else:
                sims = []
                for i, e in enumerate(ex_emb):
                    sims.append((self._cosine(q_emb, e), i))
        sims.sort(key=lambda x: x[0], reverse=True)
        order = [i for _, i in sims]
        # MMR diversification
        if diverse and len(order) > 1:
            import math
            selected: List[int] = []
            candidate = order.copy()
            while candidate and len(selected) < k:
                if not selected:
                    selected.append(candidate.pop(0))
                    continue
                best_i = None; best_score = -1e9
                for idx in candidate:
                    rel = next((s for s, j in sims if j == idx), 0.0)
                    rep = 0.0
                    for sidx in selected:
                        # approximate rep with lexical overlap on text
                        rep = max(rep, self._lex_sim(_lexical_tokens(texts[idx]), _lexical_tokens(texts[sidx])))
                    score = mmr_lambda * rel - (1 - mmr_lambda) * rep
                    if score > best_score:
                        best_score, best_i = score, idx
                selected.append(best_i)
                candidate = [x for x in candidate if x != best_i]
            chosen = selected
        else:
            chosen = order[:k]
        return [examples[i] for i in chosen]


class CrossEncoderReranker:
    """Optional cross-encoder reranker. Requires sentence-transformers CrossEncoder.
    If unavailable, acts as a no-op.
    """

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        self.model_name = model
        self._ce = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._ce = CrossEncoder(self.model_name)
        except Exception:
            self._ce = None

    def available(self) -> bool:
        return self._ce is not None

    def rerank(self, query: str, examples: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
        if self._ce is None or not examples:
            return examples
        pairs = []
        texts = []
        for ex in examples:
            st = str(ex.get("stem") or ex.get("question") or "")
            opts = ex.get("options") or []
            if isinstance(opts, list) and opts:
                st += "\n" + " | ".join([str(o) for o in opts])
            pairs.append((query, st))
            texts.append(st)
        try:
            scores = self._ce.predict(pairs)
        except Exception:
            return examples
        order = list(range(len(examples)))
        order.sort(key=lambda i: float(scores[i]), reverse=True)
        if top_k is None:
            top_k = len(order)
        order = order[: int(top_k)]
        return [examples[i] for i in order]
