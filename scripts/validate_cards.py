from __future__ import annotations
import argparse, json, re
from typing import Any, Dict, List


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def tok_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", s or ""))


def validate_cards_for_step(step: Dict[str, Any]) -> Dict[str, Any]:
    task = step.get("task") or {}
    opts: List[str] = task.get("options") or []
    cards: List[Dict[str, Any]] = ((step.get("fact_cards_after") or {}).get("cards")) or []
    retrieved = (step.get("tool_outputs") or [])
    snips: List[str] = []
    for to in retrieved:
        if to.get("name") == "retriever":
            snips.extend(to.get("snippets") or [])
    issues: List[str] = []
    counts = {"option": 0, "context": 0, "other": 0}
    seen_ids = set()
    per_opt = {i: 0 for i in range(len(opts))}
    for c in cards:
        cid = c.get("id")
        if cid in seen_ids:
            issues.append(f"duplicate_id:{cid}")
        if cid:
            seen_ids.add(cid)
        w = c.get("where") or {}
        scope = w.get("scope")
        if scope == "option":
            counts["option"] += 1
            oi = w.get("option_index")
            if not isinstance(oi, int) or not (0 <= oi < len(opts)):
                issues.append(f"bad_option_index:{oi}")
                continue
            per_opt[oi] += 1
            q = c.get("quote") or ""
            if tok_count(q) > 15:
                issues.append("long_quote_option")
            opt_text = opts[oi]
            if q and (q not in opt_text):
                issues.append("quote_not_in_option")
        elif scope == "context":
            counts["context"] += 1
            q = c.get("quote") or ""
            if tok_count(q) > 15:
                issues.append("long_quote_context")
            # require substring in any snippet if available, else in presented_stem
            src = "\n".join(snips) or (step.get("presented_stem") or "")
            if q and (q not in src):
                issues.append("quote_not_in_context")
        else:
            counts["other"] += 1
            issues.append("unknown_scope")
        tags = c.get("tags") or []
        if (not isinstance(tags, list)) or (len(tags) == 0):
            issues.append("missing_tags")
    # ensure ≥1 per option
    for i, cnt in per_opt.items():
        if cnt <= 0:
            issues.append(f"missing_option_card:{i}")
    return {
        "counts": counts,
        "issues": issues,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate Fact-Cards in a simulator log")
    ap.add_argument("--log", required=True, help="path to JSONL log")
    args = ap.parse_args(argv)
    steps = [s for s in load_jsonl(args.log) if not s.get("run_header")]
    total = len(steps)
    agg_issues = {}
    sum_counts = {"option": 0, "context": 0, "other": 0}
    bad_steps = 0
    for s in steps:
        res = validate_cards_for_step(s)
        for k, v in res["counts"].items():
            sum_counts[k] += v
        if res["issues"]:
            bad_steps += 1
            for it in res["issues"]:
                agg_issues[it] = agg_issues.get(it, 0) + 1
    out = {
        "steps": total,
        "avg_cards_per_step": {k: (sum_counts[k] / total if total else 0.0) for k in sum_counts},
        "bad_steps": bad_steps,
        "issues": dict(sorted(agg_issues.items(), key=lambda kv: kv[1], reverse=True)),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
