#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from typing import Any, Dict, List


def perturb_text(s: str, pct: float = 7.0) -> str:
    # Adjust integers by +/- pct (default +pct)
    def num_repl(m):
        try:
            x = int(m.group(0))
            y = round(x * (1.0 + pct / 100.0))
            return str(y)
        except Exception:
            return m.group(0)
    s2 = re.sub(r"\b\d{2,6}\b", num_repl, s)
    # Shift year-like tokens (1900..2099) by +1
    def year_repl(m):
        y = int(m.group(0))
        if 1900 <= y <= 2099:
            return str(y + 1)
        return str(y)
    s3 = re.sub(r"\b(19\d{2}|20\d{2})\b", year_repl, s2)
    return s3


def perturb_cards_obj(obj: Dict[str, Any], pct: float) -> Dict[str, Any]:
    cards = list(obj.get("cards") or [])
    out: List[Dict[str, Any]] = []
    for c in cards:
        try:
            cc = dict(c)
            cc["claim"] = perturb_text(cc.get("claim") or "", pct)
            cc["quote"] = perturb_text(cc.get("quote") or "", pct)
            out.append(cc)
        except Exception:
            out.append(c)
    return {**obj, "cards": out}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Counterfactually perturb Fact-Cards JSON/JSONL (numbers/dates)")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--percent", type=float, default=7.0, help="percent change to apply to numeric values")
    args = p.parse_args(argv)

    lines = []
    with open(args.in_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    out_lines: List[str] = []
    for ln in lines:
        try:
            js = json.loads(ln)
            if isinstance(js, dict) and isinstance(js.get("cards"), list):
                out_lines.append(json.dumps(perturb_cards_obj(js, args.percent), ensure_ascii=False))
            else:
                out_lines.append(ln)
        except Exception:
            out_lines.append(ln)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

