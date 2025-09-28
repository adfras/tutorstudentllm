from __future__ import annotations
import re
from typing import Dict, List


DEFAULT_TERMS = {
    # Domain-agnostic vocabulary to seed anonymization
    "concept","pattern","prototype","category","feature","example","sample","token","label",
    "input","output","signal","feedback","reward","penalty","habit","goal","rule","strategy",
    "evidence","consensus","contrast","similarity","difference","association","correlation","causation",
    "experiment","hypothesis","variable","independent","dependent","control","random","validity","reliability",
    "sequence","memory","attention","language","reasoning","learning","generalization","discrimination",
}


def build_vocab(skill_map: dict) -> List[str]:
    vocab: set[str] = set()
    try:
        for s in skill_map.get("skills", {}).values():
            name = (s.get("name") or "").lower()
            for tok in re.findall(r"[a-zA-Z]{4,}", name):
                vocab.add(tok)
    except Exception:
        pass
    vocab.update(DEFAULT_TERMS)
    stop = {"introductory","understand","analysis","apply","create","evaluate","research","methods"}
    return [t for t in vocab if t not in stop]


def compile_codebook(vocab: List[str], seed: int = 12345) -> Dict[str, str]:
    import random
    rng = random.Random(seed)
    def code_token() -> str:
        letters = ''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(4))
        num = rng.randrange(1, 99)
        return f"{letters}-{num:02d}"
    mapping: Dict[str, str] = {}
    for term in sorted(set(vocab)):
        mapping[term] = code_token()
    return mapping


def apply_codebook(text: str, codebook: Dict[str, str]) -> str:
    def repl(m):
        w = m.group(0)
        k = w.lower()
        return codebook.get(k, w)
    if not codebook:
        return text
    keys = sorted((k for k in codebook.keys() if k), key=len, reverse=True)
    if not keys:
        return text
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.I)
    try:
        return re.sub(pattern, repl, text)
    except Exception:
        return text


def scramble_numbers(text: str, a: int, b: int) -> str:
    def repl(m):
        try:
            x = int(m.group(0))
            y = a * x + b
            return str(y)
        except Exception:
            return m.group(0)
    return re.sub(r"\b\d{1,4}\b", repl, text)


def anonymize_mcq(stem: str, options: List[str], codebook: Dict[str, str], a: int = 3, b: int = 7):
    def tf(s: str) -> str:
        return scramble_numbers(apply_codebook(s or "", codebook), a, b)
    return tf(stem), [tf(o or "") for o in options]


def anonymize_text(text: str, codebook: Dict[str, str], a: int = 3, b: int = 7) -> str:
    return scramble_numbers(apply_codebook(text or "", codebook), a, b)
