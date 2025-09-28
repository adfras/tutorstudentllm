from __future__ import annotations
import argparse, glob, json, os, statistics, math
from collections import Counter
from typing import Any, Dict, List, Tuple


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_log(path: str) -> Dict[str, Any]:
    header = None
    steps: List[Dict[str, Any]] = []
    for rec in load_jsonl(path):
        if rec.get("run_header"):
            header = rec
            continue
        steps.append(rec)
    return {"path": path, "header": header, "steps": steps}


def metrics_for_run(run: Dict[str, Any]) -> Dict[str, Any]:
    steps = run.get("steps", [])
    out: Dict[str, Any] = {"n": len(steps)}
    # Timing and usage per step
    step_secs: List[float] = []
    step_tokens: List[int] = []
    for s in steps:
        ms = s.get("duration_ms")
        if isinstance(ms, (int, float)):
            step_secs.append(float(ms) / 1000.0)
        su = s.get("student_usage") or {}
        tt = su.get("total_tokens")
        if isinstance(tt, int):
            step_tokens.append(int(tt))
    if step_secs:
        import statistics as _st
        out["timing"] = {
            "mean_step_seconds": float(_st.mean(step_secs)),
            "median_step_seconds": float(_st.median(step_secs)),
            "total_seconds": float(sum(step_secs)),
            "steps": len(step_secs),
        }
    if step_tokens:
        import statistics as _st
        out["usage"] = {
            "tokens_per_step_mean": float(_st.mean(step_tokens)),
            "tokens_total": int(sum(step_tokens)),
            "steps": len(step_tokens),
        }
    mcq = [s for s in steps if s.get("task", {}).get("type") == "mcq"]
    saq = [s for s in steps if s.get("task", {}).get("type") == "saq"]
    if mcq:
        inst = [1 if s.get("evaluation", {}).get("correct") else 0 for s in mcq]
        # credited via citations_evidence (generic) or alias_evidence
        cred_inst: List[int] = []
        witness_inst: List[int] = []
        reason_counts: Counter[str] = Counter()
        for s in mcq:
            ev = s.get("evaluation", {})
            ce = ev.get("citations_evidence")
            if ce is not None:
                cred_inst.append(1 if ce.get("credited") else 0)
                witness_inst.append(1 if ce.get("witness_pass") else 0)
                reasons = ce.get("reasons")
                if isinstance(reasons, list):
                    reason_counts.update(r for r in reasons if isinstance(r, str))
            else:
                ae = (s.get("alias_evidence") or {})
                cred_inst.append(1 if ae.get("credited") else 0)
                # alias evidence has witness only in alias runs; default 0/None
                w = ae.get("witness_pass")
                if w is None:
                    witness_inst.append(0)
                else:
                    witness_inst.append(1 if w else 0)
        # cumulative curves
        cum = []
        c = 0
        for i, v in enumerate(inst, 1):
            c += v
            cum.append(c / i)
        cred_cum = []
        c2 = 0
        for i, v in enumerate(cred_inst, 1):
            c2 += v
            cred_cum.append(c2 / i)
        wit_cum = []
        c3 = 0
        for i, v in enumerate(witness_inst, 1):
            c3 += v
            wit_cum.append(c3 / i)

        # Early vs late windows (two-proportion z test)
        k = min(30, max(0, len(cred_inst) // 2))
        early = cred_inst[:k]
        late = cred_inst[-k:] if k > 0 else []
        s_e, n_e = sum(early), len(early)
        s_l, n_l = sum(late), len(late)
        p_e = (s_e / n_e) if n_e else None
        p_l = (s_l / n_l) if n_l else None
        delta = ((p_l - p_e) if (p_e is not None and p_l is not None) else None)
        p_val = (two_prop_pvalue(s_e, n_e, s_l, n_l) if (n_e and n_l) else None)

        # Wilson CI for credited_final
        s_total = sum(cred_inst)
        n_total = len(cred_inst)
        w_low, w_high = (None, None)
        if n_total:
            w_low, w_high = wilson_ci(s_total, n_total, 0.05)

        # final gaps and summaries
        attr_gap_final = None
        if cred_cum:
            try:
                attr_gap_final = (cum[-1] - cred_cum[-1])
            except Exception:
                attr_gap_final = None
        mcq_metrics = {
            "n": len(mcq),
            "acc_final": cum[-1],
            "acc_auc": statistics.mean(cum),
            "instant": inst,
            "cumulative": cum,
            "credited_final": (cred_cum[-1] if cred_cum else 0.0),
            "credited_auc": (statistics.mean(cred_cum) if cred_cum else 0.0),
            "attribution_gap_final": attr_gap_final,
            "credited_instant": cred_inst,
            "credited_cumulative": cred_cum,
            "witness_final": (wit_cum[-1] if wit_cum else None),
            "witness_auc": (statistics.mean(wit_cum) if wit_cum else None),
            "witness_instant": witness_inst,
            "early_late": {
                "k": k,
                "p_early": p_e,
                "p_late": p_l,
                "delta": delta,
                "p_value": p_val,
                "credited_wilson95": {"low": w_low, "high": w_high, "n": n_total, "successes": s_total},
            },
            "retention": retention_stats(cred_inst),
        }
        snippet_count = 0
        if reason_counts:
            mcq_metrics["evidence_reasons"] = {
                "counts": dict(reason_counts),
                "rates": {k: (reason_counts[k] / len(mcq)) for k in reason_counts},
            }
            snippet_count = reason_counts.get("no_snippet_quote", 0)
            mcq_metrics["no_snippet_quote_rate"] = snippet_count / len(mcq)
        else:
            mcq_metrics["no_snippet_quote_rate"] = 0.0
        mcq_metrics["no_snippet_quote_count"] = snippet_count
        out["mcq"] = mcq_metrics
    if saq:
        scores = [float(s.get("grading", {}).get("score") or 0.0) for s in saq]
        out.update({
            "saq": {
                "n": len(saq),
                "score_mean": statistics.mean(scores) if scores else 0.0,
                "scores": scores,
            }
        })
    # Alias-swap metrics
    fam_stats: Dict[str, Dict[str, Any]] = {}
    for s in steps:
        alias = s.get("alias") or {}
        fam = alias.get("family_id")
        if not fam:
            continue
        st = fam_stats.setdefault(fam, {"A": [], "B": [], "B_credited": []})
        if alias.get("phase") == "A":
            st["A"].append(1 if (s.get("evaluation", {}).get("correct")) else 0)
        elif alias.get("phase") == "B":
            st["B"].append(1 if (s.get("evaluation", {}).get("correct")) else 0)
            ev = (s.get("alias_evidence") or {})
            st["B_credited"].append(1 if ev.get("credited") else 0)
    if fam_stats:
        alias_out = {}
        for fam, st in fam_stats.items():
            acc_a = sum(st["A"]) / max(1, len(st["A"]))
            acc_b = sum(st["B"]) / max(1, len(st["B"]))
            b_count = sum(st["B_credited"]) if st["B_credited"] else 0
            b_total = len(st["B_credited"]) if st["B_credited"] else 0
            credited_b = (b_count / b_total) if b_total else None
            w_low = w_high = None
            if b_total:
                w_low, w_high = wilson_ci(b_count, b_total, 0.05)
            alias_out[fam] = {
                "acc_A": acc_a,
                "acc_B": acc_b,
                "delta": (acc_a - acc_b),
                "credited_B": credited_b,
                "B_counts": {"credited": b_count, "total": b_total, "wilson95": {"low": w_low, "high": w_high}},
            }
        out["alias"] = alias_out
    return out


def group_key(header: Dict[str, Any]) -> str:
    cfg = (header or {}).get("config", {})
    d = cfg.get("dials", {})
    keys = [
        f"closed={d.get('closed_book')}",
        f"anon={d.get('anonymize')}",
        f"accum={d.get('accumulate_notes')}",
        f"ctxpos={d.get('context_position')}",
        f"verify={d.get('verify')}",
        f"sc={d.get('self_consistency_n')}",
        f"rich={d.get('rich')}",
        f"task={cfg.get('task')}",
        f"diff={cfg.get('difficulty')}",
    ]
    return ";".join(keys)


def aggregate_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate MCQ and SAQ metrics across runs (aligned per-step averages)
    stats: Dict[str, Any] = {}
    mcq_runs = [r for r in runs if r.get("metrics", {}).get("mcq")]
    saq_runs = [r for r in runs if r.get("metrics", {}).get("saq")]
    if mcq_runs:
        finals = [r["metrics"]["mcq"]["acc_final"] for r in mcq_runs]
        aucs = [r["metrics"]["mcq"]["acc_auc"] for r in mcq_runs]
        cred_finals = [r["metrics"]["mcq"].get("credited_final", 0.0) for r in mcq_runs]
        cred_aucs = [r["metrics"]["mcq"].get("credited_auc", 0.0) for r in mcq_runs]
        wit_finals = [r["metrics"]["mcq"].get("witness_final") for r in mcq_runs if r["metrics"]["mcq"].get("witness_final") is not None]
        wit_aucs = [r["metrics"]["mcq"].get("witness_auc") for r in mcq_runs if r["metrics"]["mcq"].get("witness_auc") is not None]
        # step-wise average of cumulative acc
        max_len = max(len(r["metrics"]["mcq"]["cumulative"]) for r in mcq_runs)
        step_avgs = []
        cred_step_avgs = []
        for i in range(max_len):
            vals = []
            cvals = []
            for r in mcq_runs:
                cum = r["metrics"]["mcq"]["cumulative"]
                ccum = r["metrics"]["mcq"].get("credited_cumulative", [])
                if i < len(cum):
                    vals.append(cum[i])
                if i < len(ccum):
                    cvals.append(ccum[i])
            if vals:
                step_avgs.append(sum(vals)/len(vals))
            if cvals:
                cred_step_avgs.append(sum(cvals)/len(cvals))
        # pooled early/late across runs
        s_e = n_e = s_l = n_l = 0
        s_total = n_total = 0
        for r in mcq_runs:
            ci = r["metrics"]["mcq"].get("credited_instant", [])
            k = min(30, max(0, len(ci)//2))
            if k > 0:
                s_e += sum(ci[:k]); n_e += k
                s_l += sum(ci[-k:]); n_l += k
            s_total += sum(ci); n_total += len(ci)
        p_e = (s_e / n_e) if n_e else None
        p_l = (s_l / n_l) if n_l else None
        delta = ((p_l - p_e) if (p_e is not None and p_l is not None) else None)
        p_val = (two_prop_pvalue(s_e, n_e, s_l, n_l) if (n_e and n_l) else None)
        w_low, w_high = (None, None)
        if n_total:
            w_low, w_high = wilson_ci(s_total, n_total, 0.05)
        stats["mcq"] = {
            "runs": len(mcq_runs),
            "acc_final_mean": statistics.mean(finals),
            "acc_auc_mean": statistics.mean(aucs),
            "cumulative_mean": step_avgs,
            "credited_final_mean": (statistics.mean(cred_finals) if cred_finals else 0.0),
            "credited_auc_mean": (statistics.mean(cred_aucs) if cred_aucs else 0.0),
            "credited_cumulative_mean": cred_step_avgs,
            "witness_final_mean": (statistics.mean(wit_finals) if wit_finals else None),
            "witness_auc_mean": (statistics.mean(wit_aucs) if wit_aucs else None),
            "early_late": {
                "k_total": n_e,  # sum over runs
                "p_early": p_e,
                "p_late": p_l,
                "delta": delta,
                "p_value": p_val,
                "credited_wilson95": {"low": w_low, "high": w_high, "n": n_total, "successes": s_total},
            },
        }
        snippet_total = sum(r["metrics"]["mcq"].get("no_snippet_quote_count", 0) for r in mcq_runs)
        step_total = sum(len(r["metrics"]["mcq"].get("instant", [])) for r in mcq_runs)
        snippet_rate = (snippet_total / step_total) if step_total else 0.0
        stats["mcq"]["no_snippet_quote_rate"] = snippet_rate
        stats["mcq"]["no_snippet_quote_count"] = snippet_total
        stats["mcq"]["steps"] = step_total
        if snippet_rate > 0.2:
            stats.setdefault("alerts", []).append({
                "type": "no_snippet_quote_rate",
                "rate": snippet_rate,
                "threshold": 0.2,
                "count": snippet_total,
                "steps": step_total,
            })
    # Aggregate timing/usage across runs
    timings = [r.get("metrics", {}).get("timing") for r in runs if r.get("metrics", {}).get("timing")]
    if timings:
        tot_secs = sum(t.get("total_seconds", 0.0) for t in timings)
        steps = sum(int(t.get("steps", 0)) for t in timings)
        mean_step = (tot_secs / steps) if steps else None
        medians = [t.get("median_step_seconds") for t in timings if t.get("median_step_seconds") is not None]
        stats["timing"] = {
            "total_seconds": tot_secs,
            "mean_step_seconds": mean_step,
            "median_step_seconds": (statistics.median(medians) if medians else None),
            "steps": steps,
        }
    usages = [r.get("metrics", {}).get("usage") for r in runs if r.get("metrics", {}).get("usage")]
    if usages:
        tokens_total = sum(int(u.get("tokens_total", 0)) for u in usages)
        steps = sum(int(u.get("steps", 0)) for u in usages)
        stats["usage"] = {
            "tokens_total": tokens_total,
            "tokens_per_step_mean": ((tokens_total / steps) if steps else None),
            "steps": steps,
        }
    if saq_runs:
        means = [r["metrics"]["saq"]["score_mean"] for r in saq_runs]
        stats["saq"] = {
            "runs": len(saq_runs),
            "score_mean": statistics.mean(means),
        }
    # Alias aggregation: average per-family delta and credited_B; also pooled counts for Wilson
    alias_coll: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        alias = r.get("metrics", {}).get("alias") or {}
        for fam, m in alias.items():
            alias_coll.setdefault(fam, []).append(m)
    if alias_coll:
        alias_out = {}
        for fam, lst in alias_coll.items():
            deltas = [x.get("delta") for x in lst if x.get("delta") is not None]
            creds = [x.get("credited_B") for x in lst if x.get("credited_B") is not None]
            # pooled counts
            c_sum = 0
            n_sum = 0
            for x in lst:
                bc = ((x.get("B_counts") or {}).get("credited") or 0)
                bt = ((x.get("B_counts") or {}).get("total") or 0)
                c_sum += int(bc)
                n_sum += int(bt)
            w_low = w_high = None
            pooled = None
            if n_sum:
                pooled = c_sum / n_sum
                w_low, w_high = wilson_ci(c_sum, n_sum, 0.05)
            alias_out[fam] = {
                "runs": len(lst),
                "delta_mean": (statistics.mean(deltas) if deltas else None),
                "credited_B_mean": (statistics.mean(creds) if creds else None),
                "B_counts_pooled": {"credited": c_sum, "total": n_sum, "pooled": pooled, "wilson95": {"low": w_low, "high": w_high}},
            }
        stats["alias"] = alias_out
    return stats


def analyze_many(paths: List[str]) -> Dict[str, Any]:
    runs = []
    for p in paths:
        run = parse_log(p)
        run_metrics = metrics_for_run(run)
        run["metrics"] = run_metrics
        runs.append(run)
    # Group by dials
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        k = group_key(r.get("header") or {})
        grouped.setdefault(k, []).append(r)
    groups_out = {k: aggregate_group(v) for k, v in grouped.items()}
    overall = aggregate_group(runs)
    return {"runs": runs, "groups": groups_out, "overall": overall}


# ---------------------- Math helpers ----------------------

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = z_for(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def two_prop_pvalue(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value."""
    if min(n1, n2) <= 0:
        return float("nan")
    p1 = s1 / n1
    p2 = s2 / n2
    p = (s1 + s2) / (n1 + n2)
    denom = math.sqrt(p*(1-p)*(1/n1 + 1/n2)) if p*(1-p) > 0 else 0.0
    if denom == 0.0:
        return 1.0
    z = abs((p1 - p2) / denom)
    # two-sided
    return 2 * (1 - phi(z))


def phi(x: float) -> float:
    """Standard normal CDF via error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def z_for(quantile: float) -> float:
    """Approximate inverse CDF for standard normal using Abramowitz-Stegun approximation."""
    # For our limited needs we can use a simple binary search on phi
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if phi(mid) < quantile:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def retention_stats(cred_inst: List[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(cred_inst)
    for lag in (3, 4, 5):
        if n <= lag:
            continue
        # Pair step i with i+lag
        n_pairs = n - lag
        s_first = sum(cred_inst[:n_pairs])
        s_retest = sum(cred_inst[lag:])
        p_first = s_first / n_pairs
        p_retest = s_retest / n_pairs
        p_val = two_prop_pvalue(s_first, n_pairs, s_retest, n_pairs)
        out[f"lag{lag}"] = {"pairs": n_pairs, "p_first": p_first, "p_retest": p_retest, "delta": (p_retest - p_first), "p_value": p_val}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Analyze simulator JSONL logs")
    p.add_argument("--log", action="append", default=[], help="path to JSONL log (can repeat)")
    p.add_argument("--glob", default=None, help="glob pattern for JSONL logs")
    p.add_argument("--dir", default=None, help="directory with *.jsonl logs")
    args = p.parse_args(argv)
    paths = []
    paths += args.log
    if args.glob:
        paths += glob.glob(args.glob)
    if args.dir:
        paths += [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith(".jsonl")]
    paths = [p for p in paths if p]
    if not paths:
        try:
            print(json.dumps({"error": "no logs"}))
        except BrokenPipeError:
            pass
        return 1
    res = analyze_many(paths)
    out = json.dumps(res, ensure_ascii=False)
    try:
        print(out)
    except BrokenPipeError:
        try:
            import sys
            sys.stdout.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
