#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from typing import Any, Dict

from scripts.analyze import parse_log, metrics_for_run


def summarize(path: str) -> Dict[str, Any]:
    run = parse_log(path)
    met = metrics_for_run(run)
    mcq = met.get("mcq") or {}
    out = {
        "path": path,
        "n": int(met.get("n") or 0),
        "raw_acc": float(mcq.get("acc_final") or 0.0),
        "credited_acc": float(mcq.get("credited_final") or 0.0),
    }
    out["attribution_gap"] = out["raw_acc"] - out["credited_acc"]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compute RAG delta and attribution gaps from two logs")
    p.add_argument("--on", dest="on_path", required=True, help="log with retrieval/tools ON")
    p.add_argument("--off", dest="off_path", required=True, help="log with retrieval/tools OFF")
    args = p.parse_args(argv)

    on = summarize(args.on_path)
    off = summarize(args.off_path)
    rag_delta = on["credited_acc"] - off["credited_acc"]
    out = {
        "with_tools": on,
        "no_tools": off,
        "rag_delta": rag_delta,
        "interpretation": {
            "healthy_rag_delta_min": 0.15,
            "attribution_gap_max": 0.05,
        },
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

