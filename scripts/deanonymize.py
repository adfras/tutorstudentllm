from __future__ import annotations
import argparse
import json
import re
from typing import Dict

from tutor.skill_map import load_skill_map
from sim.domain import DomainStore
from sim.anonymize import compile_codebook


def build_codebook_from_header(header: dict) -> Dict[str, str]:
    # Reconstruct the same vocabulary used during the run
    smap = load_skill_map()
    # Extract vocab from skill names
    import re as _re
    vocab: set[str] = set()
    for s in smap["skills"].values():
        name = (s.get("name") or "").lower()
        for tok in _re.findall(r"[a-zA-Z]{4,}", name):
            vocab.add(tok)
    # Merge domain glossary
    domain_id = ((header or {}).get("config") or {}).get("domain") or "psych"
    ds = DomainStore()
    try:
        vocab.update(ds.glossary_terms(domain_id))
    except Exception:
        pass
    seed = ((header or {}).get("anonymization") or {}).get("seed")
    if seed is None:
        raise RuntimeError("Header missing anonymization seed")
    return compile_codebook(sorted(vocab), seed=int(seed))


def deanonymize_text(text: str, inverse: Dict[str, str]) -> str:
    if not inverse:
        return text
    # Replace code tokens with originals; match whole tokens
    keys = sorted(inverse.keys(), key=len, reverse=True)
    # tokens contain hyphens; use word-ish boundaries
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    def repl(m):
        t = m.group(0)
        return inverse.get(t, t)
    return pattern.sub(repl, text)


def process(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if not lines:
        open(out_path, "w").close()
        return
    header = json.loads(lines[0])
    codebook = build_codebook_from_header(header)
    inverse = {v: k for k, v in codebook.items()}
    out_lines = [lines[0]]
    for ln in lines[1:]:
        try:
            rec = json.loads(ln)
            if "presented_stem" in rec and isinstance(rec["presented_stem"], str):
                rec["presented_stem"] = deanonymize_text(rec["presented_stem"], inverse)
            out_lines.append(json.dumps(rec, ensure_ascii=False))
        except Exception:
            out_lines.append(ln)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="De-anonymize a simulator JSONL log using its header seed")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    args = p.parse_args(argv)
    process(args.in_path, args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

