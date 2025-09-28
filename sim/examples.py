from __future__ import annotations

"""
Few-shot example selection and preparation.

Encapsulates selection, optional reranking, ordering, and lightweight
compression used for including examples in prompts.
"""

from typing import Any, Dict, List
from .card_quality import tokens as _tokens


def select_and_prepare_examples(qtext: str, exemplars: List[Dict[str, Any]], cfg) -> List[Dict[str, Any]]:
    """Select up to cfg.shots_k examples for a question text.

    Implements the same behavior formerly in orchestrator: KNN/lexical fallback,
    optional CE reranker, ordering, and optional compression.
    """
    selected_examples: List[Dict[str, Any]] = []
    try:
        if not exemplars or int(getattr(cfg, "shots_k", 0) or 0) <= 0:
            return []
        if (cfg.shots_selector or "knn") == "random":
            import random as _r
            choices = exemplars.copy()
            _r.shuffle(choices)
            selected_examples = choices[: int(cfg.shots_k)]
        elif (cfg.shots_selector or "knn") == "as-is":
            selected_examples = exemplars[: int(cfg.shots_k)]
        else:
            # Embedding KNN with optional MMR diversity
            try:
                from sim.retrieval import EmbeddingIndex

                index = EmbeddingIndex(backend=(cfg.shots_embed_backend or "lexical"))
                selected_examples = index.knn(
                    qtext,
                    exemplars,
                    k=int(cfg.shots_k),
                    diverse=bool(getattr(cfg, "shots_diverse", False)),
                    mmr_lambda=float(getattr(cfg, "shots_mmr", 0.5) or 0.5),
                )
            except Exception:
                # lexical fallback
                def _text_of(ex: Dict[str, Any]) -> str:
                    st = str(ex.get("stem") or ex.get("question") or "")
                    opts = ex.get("options") or []
                    if isinstance(opts, list) and opts:
                        st += "\n" + " | ".join([str(o) for o in opts])
                    return st

                qt = set(_tokens(qtext))
                scored: List[tuple[int, Dict[str, Any]]] = []
                for ex in exemplars:
                    et = set(_tokens(_text_of(ex)))
                    score = len(qt & et)
                    scored.append((score, ex))
                scored.sort(key=lambda x: x[0], reverse=True)
                selected_examples = [ex for _, ex in scored[: int(cfg.shots_k)]]
        # Optional cross-encoder reranker
        try:
            if selected_examples and (getattr(cfg, "shots_reranker", "none") == "ce"):
                from sim.retrieval import CrossEncoderReranker

                rr = CrossEncoderReranker(model=getattr(cfg, "shots_reranker_model", "BAAI/bge-reranker-base"))
                if rr.available():
                    selected_examples = rr.rerank(qtext, selected_examples, top_k=int(cfg.shots_k))
        except Exception:
            pass

        # Optional ordering
        if (cfg.shots_order or "similar") == "easy-hard":
            def _lev(ex: Dict[str, Any]) -> int:
                d = (ex.get("difficulty") or ex.get("level") or "").strip().lower()
                if d in ("easy", "e"):
                    return 0
                if d in ("medium", "m", "med"):
                    return 1
                if d in ("hard", "h"):
                    return 2
                return 1

            selected_examples.sort(key=_lev)
        # 'similar' keeps KNN order; 'as-is' preserves file order

        # Optional compression (LLMLingua-lite heuristic)
        if getattr(cfg, "dials", None) and bool(getattr(cfg.dials, "compress_examples", False)):
            def _compress_text(s: str, factor: float) -> str:
                import re as _re
                toks = _re.findall(r"[A-Za-z0-9]+|\S", s or "")
                if not toks:
                    return s or ""
                keep = []
                stop = {"the", "a", "an", "of", "and", "or", "to", "is", "are", "was", "were", "in", "on", "for", "with"}
                for t in toks:
                    al = t.isalpha()
                    if (not al) or (t.lower() not in stop) or (len(t) >= 5):
                        keep.append(t)
                k = max(1, int(len(keep) / max(1.0, float(getattr(cfg.dials, "compress_ratio", 3.0) or 3.0))))
                keep2 = keep[:k] if k < len(keep) else keep
                out = " ".join(keep2)
                return out

            def _compress_ex(ex: Dict[str, Any]) -> Dict[str, Any]:
                ex2 = dict(ex)
                if ex2.get("stem"):
                    ex2["stem"] = _compress_text(ex2.get("stem"), getattr(cfg.dials, "compress_ratio", 3.0))
                if isinstance(ex2.get("options"), list):
                    ex2["options"] = [_compress_text(str(o), getattr(cfg.dials, "compress_ratio", 3.0)) for o in ex2["options"]]
                if ex2.get("rationale"):
                    ex2["rationale"] = _compress_text(ex2.get("rationale"), getattr(cfg.dials, "compress_ratio", 3.0))
                return ex2

            selected_examples = [_compress_ex(e) for e in selected_examples]
        return selected_examples
    except Exception:
        return []
