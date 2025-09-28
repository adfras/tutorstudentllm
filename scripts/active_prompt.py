import argparse, json, math


def vote_entropy(votes):
    from collections import Counter
    vs = [v for v in votes if v is not None]
    if not vs:
        return 0.0
    total = len(vs)
    cnt = Counter(vs)
    import math as _m
    probs = [c/total for c in cnt.values()]
    H = -sum(p*_m.log(p+1e-12) for p in probs)
    Hmax = _m.log(len(probs)) if probs else 1.0
    return (H / Hmax) if Hmax else 0.0


def main():
    p = argparse.ArgumentParser(description="Select most uncertainty-reducing items for Active Prompt labeling")
    p.add_argument("--log", required=True, help="simulator JSONL log")
    p.add_argument("--topk", type=int, default=8)
    args = p.parse_args()
    cand = []
    with open(args.log, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("{") and '"run_header": true' in ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            ans = rec.get("answer") or {}
            votes = ans.get("votes") or []
            conf = (ans.get("raw") or {}).get("confidence")
            Hn = vote_entropy(votes)
            score = max(Hn, 1.0 - float(conf or 0.0))
            cand.append((score, {"stem": rec.get("presented_stem") or rec.get("stem"), "votes": votes, "confidence": conf}))
    cand.sort(key=lambda x: x[0], reverse=True)
    out = [c for _, c in cand[: args.topk]]
    print(json.dumps({"selected": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

