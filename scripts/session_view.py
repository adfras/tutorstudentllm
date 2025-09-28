#!/usr/bin/env python3
"""
Build a per-session (per-run_id) table from the flattened steps CSV.

Inputs:
- runs/_aggregated/all_steps_flat.csv.gz (default)

Outputs:
- runs/_aggregated/session_view.csv.gz

No pandas dependency; uses csv + gzip from stdlib.
"""

from __future__ import annotations

import argparse
import csv
import sys
import gzip
import math
from collections import defaultdict
from typing import Any, Dict


def _to_float(x: Any) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate step-level CSV into a session/run view")
    ap.add_argument("--steps-csv", default="runs/_aggregated/all_steps_flat.csv.gz")
    ap.add_argument("--out-csv", default="runs/_aggregated/session_view.csv.gz")
    args = ap.parse_args()

    # Accumulators per run_id
    acc: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "run_id": None,
        "source_path_first": None,
        "source_dir": None,
        "run_ts_min": None,
        "task": None,
        "difficulty": None,
        "domain": None,
        "closed_book": None,
        "use_fact_cards": None,
        "require_citations": None,
        "self_consistency_n": None,
        "idk_enabled": None,
        "fact_cards_budget": None,
        "context_position": None,
        "tools": None,
        "controller": None,
        # aggregates
        "steps_n": 0,
        "correct_sum": 0,
        "credited_sum": 0,
        "witness_sum": 0,
        "abstain_sum": 0,
        "coverage_sum": 0.0,
        "coverage_n": 0,
        "duration_ms_sum": 0.0,
        "student_tokens_sum": 0,
        "tutor_tokens_sum": 0,
        "turns_sum": 0,
        "first_attempt_correct": None,
    })

    # Increase CSV max field size to handle long presented_stem
    try:
        csv.field_size_limit(10_000_000)
    except Exception:
        pass
    with gzip.open(args.steps_csv, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("run_id") or ""
            if not run_id:
                # may be a JSON summary row
                continue
            a = acc[run_id]
            a["run_id"] = run_id
            if not a["source_path_first"]:
                a["source_path_first"] = row.get("source_path")
            a["source_dir"] = row.get("source_dir") or a["source_dir"]
            ts = _to_int(row.get("ts"))
            if ts is not None:
                a["run_ts_min"] = ts if a["run_ts_min"] is None else min(int(a["run_ts_min"]), ts)
            # capture stable config fields (first non-empty wins)
            for k_csv, k_out in [
                ("config.task", "task"),
                ("config.difficulty", "difficulty"),
                ("config.domain", "domain"),
                ("config.dials.closed_book", "closed_book"),
                ("config.dials.use_fact_cards", "use_fact_cards"),
                ("config.dials.require_citations", "require_citations"),
                ("config.dials.self_consistency_n", "self_consistency_n"),
                ("config.dials.idk_enabled", "idk_enabled"),
                ("config.dials.fact_cards_budget", "fact_cards_budget"),
                ("config.dials.context_position", "context_position"),
                ("config.dials.tools", "tools"),
                ("config.dials.controller", "controller"),
            ]:
                v = row.get(k_csv)
                if v not in (None, "") and a[k_out] in (None, ""):
                    a[k_out] = v
            # aggregates
            a["steps_n"] += 1
            is_correct = (row.get("evaluation.correct") == "True")
            if is_correct:
                a["correct_sum"] += 1
            if row.get("evaluation.citations_evidence.credited") == "True":
                a["credited_sum"] += 1
            if row.get("evaluation.citations_evidence.witness_pass") == "True":
                a["witness_sum"] += 1
            if row.get("evaluation.abstained") == "True":
                a["abstain_sum"] += 1
            cov = _to_float(row.get("evaluation.citations_evidence.coverage"))
            if cov is not None:
                a["coverage_sum"] += cov
                a["coverage_n"] += 1
            # Count turns/messages if present
            try:
                mjs = row.get("messages")
                if mjs:
                    import json as _json
                    msgs = _json.loads(mjs)
                    if isinstance(msgs, list):
                        a["turns_sum"] += len(msgs)
            except Exception:
                pass

            dur = _to_float(row.get("duration_ms"))
            if dur is not None:
                a["duration_ms_sum"] += dur
            st = _to_int(row.get("student_usage.total_tokens"))
            if st is not None:
                a["student_tokens_sum"] += st
            tt = _to_int(row.get("tutor_usage.total_tokens"))
            if tt is not None:
                a["tutor_tokens_sum"] += tt

    # Write output
    fieldnames = [
        "run_id",
        "run_ts_min",
        "source_dir",
        "source_path_first",
        "task",
        "difficulty",
        "domain",
        "closed_book",
        "use_fact_cards",
        "require_citations",
        "self_consistency_n",
        "idk_enabled",
        "fact_cards_budget",
        "context_position",
        "tools",
        "controller",
        "steps_n",
        "correct_sum",
        "credited_sum",
        "witness_sum",
        "abstain_sum",
        "acc_final",
        "credited_final",
        "witness_final",
        "abstain_rate",
        "coverage_mean",
        "mean_step_seconds",
        "duration_seconds",
        "student_tokens_sum",
        "tutor_tokens_sum",
        "tokens_per_step",
        "tutor_tokens_per_step",
        "talk_ratio_tokens",
        "n_turns",
        "first_attempt_correct",
    ]

    with gzip.open(args.out_csv, "wt", encoding="utf-8", newline="") as g:
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for rid, a in acc.items():
            steps = max(1, int(a["steps_n"]))
            student = int(a["student_tokens_sum"])
            tutor = int(a["tutor_tokens_sum"])
            denom = student + tutor
            talk = (tutor / denom) if denom > 0 else ""
            row = {
                "run_id": rid,
                "run_ts_min": a["run_ts_min"],
                "source_dir": a["source_dir"],
                "source_path_first": a["source_path_first"],
                "task": a["task"],
                "difficulty": a["difficulty"],
                "domain": a["domain"],
                "closed_book": a["closed_book"],
                "use_fact_cards": a["use_fact_cards"],
                "require_citations": a["require_citations"],
                "self_consistency_n": a["self_consistency_n"],
                "idk_enabled": a["idk_enabled"],
                "fact_cards_budget": a["fact_cards_budget"],
                "context_position": a["context_position"],
                "tools": a["tools"],
                "controller": a["controller"],
                "steps_n": steps,
                "correct_sum": a["correct_sum"],
                "credited_sum": a["credited_sum"],
                "witness_sum": a["witness_sum"],
                "abstain_sum": a["abstain_sum"],
                "acc_final": a["correct_sum"] / steps,
                "credited_final": a["credited_sum"] / steps,
                "witness_final": a["witness_sum"] / steps,
                "abstain_rate": a["abstain_sum"] / steps,
                "coverage_mean": (a["coverage_sum"] / a["coverage_n"]) if a["coverage_n"] > 0 else "",
                "mean_step_seconds": (a["duration_ms_sum"] / 1000.0) / steps,
                "duration_seconds": (a["duration_ms_sum"] / 1000.0),
                "student_tokens_sum": student,
                "tutor_tokens_sum": tutor,
                "tokens_per_step": (student/steps) if steps>0 else "",
                "tutor_tokens_per_step": (tutor/steps) if steps>0 else "",
                "talk_ratio_tokens": talk,
                "n_turns": int(a["turns_sum"]),
                "first_attempt_correct": (0 if a["first_attempt_correct"] in (None, "") else int(a["first_attempt_correct"]))
            }
            w.writerow(row)
            # first attempt correct (step==0)
            stp = _to_int(row.get("step"))
            if stp == 0 and a["first_attempt_correct"] is None:
                a["first_attempt_correct"] = 1 if is_correct else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
