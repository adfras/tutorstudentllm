import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
from statistics import mean

import yaml

DEFAULT_RUNS_GLOB = "runs/guardrailed/*.jsonl"
DEFAULT_OUTPUT = "runs/_aggregated/model_profiles.yaml"
DEFAULT_EXTRA = "runs/_aggregated/model_profiles_phase2.yaml"

FAILURE_KEYS = (
    "coverage_below_tau",
    "witness_mismatch",
    "no_option_quote_for_choice",
    "no_snippet_quote",
    "link_below_q_min",
    "no_citations",
)


def slug_to_model(slug: str) -> str:
    """Best-effort inverse of tr '/.' '_' used in sweep scripts."""
    if "_" not in slug:
        return slug
    provider, rest = slug.split("_", 1)
    candidate = rest.replace("_", ".")
    return f"{provider}/{candidate}"


def load_runs(pattern: str):
    runs_by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted(Path().glob(pattern)):
        if path.is_dir():
            continue
        slug = path.stem.split("_state", 1)[0]
        model_id = slug_to_model(slug)
        try:
            with path.open("r", encoding="utf-8") as handle:
                per_run: dict[str, dict] = {}
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    run_id = obj.get("run_id")
                    if not run_id:
                        continue
                    data = per_run.setdefault(run_id, {"steps": []})
                    if obj.get("run_header"):
                        data["header"] = obj
                        continue
                    data["steps"].append(obj)
        except FileNotFoundError:
            continue
        for run_id, payload in per_run.items():
            steps = payload.get("steps", [])
            if not steps:
                continue
            metrics = compute_run_metrics(steps)
            metrics.update({
                "file": str(path),
                "run_id": run_id,
                "slug": slug,
            })
            runs_by_model[model_id].append(metrics)
    return runs_by_model


def compute_run_metrics(steps: list[dict]) -> dict:
    total_steps = len(steps)
    raw = 0
    credited = 0
    coverage_vals = []
    witness_hits = 0
    token_totals = []
    durations = []
    failures = collections.Counter()
    for step in steps:
        evaluation = step.get("evaluation", {})
        if evaluation.get("correct"):
            raw += 1
        citations = evaluation.get("citations_evidence", {})
        if citations.get("credited"):
            credited += 1
        else:
            reasons = citations.get("reasons") or []
            failures.update(reasons)
        evidence_report = evaluation.get("evidence_report", {})
        cov = evidence_report.get("coverage")
        if cov is not None:
            coverage_vals.append(cov)
        if citations.get("witness_pass"):
            witness_hits += 1
        usage = step.get("student_usage") or {}
        tutor = step.get("tutor_usage") or {}
        token_totals.append((usage.get("total_tokens", 0) or 0) + (tutor.get("total_tokens", 0) or 0))
        duration_ms = step.get("duration_ms")
        if duration_ms is not None:
            durations.append(duration_ms / 1000.0)
    avg_coverage = mean(coverage_vals) if coverage_vals else 0.0
    avg_tokens = mean(token_totals) if token_totals else 0.0
    avg_step_seconds = mean(durations) if durations else 0.0
    return {
        "steps": total_steps,
        "raw": raw,
        "credited": credited,
        "coverage": avg_coverage,
        "witness": witness_hits / total_steps if total_steps else 0.0,
        "tokens": avg_tokens,
        "step_seconds": avg_step_seconds,
        "failures": failures,
    }


def aggregate_model_metrics(runs_by_model: dict[str, list[dict]]):
    by_model: dict[str, dict] = {}
    for model, runs in runs_by_model.items():
        total_steps = sum(r["steps"] for r in runs)
        if not total_steps:
            continue
        raw_rate = sum(r["raw"] for r in runs) / total_steps
        credited_rate = sum(r["credited"] for r in runs) / total_steps
        coverage = sum(r["coverage"] * r["steps"] for r in runs) / total_steps
        witness = sum(r["witness"] * r["steps"] for r in runs) / total_steps
        tokens = sum(r["tokens"] * r["steps"] for r in runs) / total_steps
        step_seconds = sum(r["step_seconds"] * r["steps"] for r in runs) / total_steps
        failures = collections.Counter()
        for r in runs:
            failures.update(r.get("failures", {}))
        recommendations = recommend_overrides(
            raw_rate,
            credited_rate,
            coverage,
            witness,
            tokens,
            failures,
            total_steps,
        )
        snippet_rate = failures.get("no_snippet_quote", 0) / total_steps if total_steps else 0.0
        by_model[model] = {
            "runs": len(runs),
            "steps": total_steps,
            "metrics": {
                "raw_rate": round(raw_rate, 4),
                "credited_rate": round(credited_rate, 4),
                "coverage": round(coverage, 4),
                "witness": round(witness, 4),
                "tokens_per_step": round(tokens, 1),
                "mean_step_seconds": round(step_seconds, 2),
                "no_snippet_quote_rate": round(snippet_rate, 4),
            },
            "failures": {k: failures[k] for k in FAILURE_KEYS if failures.get(k)},
            "profile": recommendations["profile"],
            "overrides": recommendations["overrides"],
            "notes": recommendations["notes"],
        }
    return by_model


def merge_extra_profiles(model_data: dict[str, dict], path_str: str | None = None) -> None:
    if not path_str:
        return
    extra_path = Path(path_str)
    if not extra_path.exists():
        return
    try:
        with extra_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return
    extras = payload.get("models") or {}

    for key, extra in extras.items():
        if not isinstance(extra, dict):
            continue
        # Compress seed-suffixed keys like model.seedNNN.N10 → model
        base = key.split(".seed", 1)[0]
        if "/" not in base:
            # Skip synthetic witness configs, keep only model ids
            continue
        dest = model_data.get(base)
        if dest is None:
            model_data[base] = extra.copy()
            continue
        merge_profile_entry(dest, extra)


def merge_profile_entry(dest: dict, extra: dict) -> None:
    existing_steps = dest.get("steps", 0) or 0
    extra_steps = extra.get("steps", 0) or 0
    existing_runs = dest.get("runs", 0) or 0
    extra_runs = extra.get("runs", 0) or 0

    dest["runs"] = existing_runs + extra_runs
    dest["steps"] = existing_steps + extra_steps

    dest_failures = dest.setdefault("failures", {})
    for k, v in (extra.get("failures") or {}).items():
        dest_failures[k] = dest_failures.get(k, 0) + v

    dest_notes = dest.setdefault("notes", [])
    for note in (extra.get("notes") or []):
        if note not in dest_notes:
            dest_notes.append(note)

    override_map: dict[str, dict] = {}
    for item in dest.get("overrides", []) or []:
        flag = item.get("flag")
        if not flag or flag == "--min-card-len":
            continue
        override_map[flag] = item
    for item in extra.get("overrides", []) or []:
        flag = item.get("flag")
        if not flag:
            continue
        if flag == "--min-card-len":
            continue
        override_map[flag] = item
    dest["overrides"] = list(override_map.values())

    extra_profile = extra.get("profile")
    if extra_profile and dest.get("profile") != extra_profile:
        dest["profile"] = extra_profile

    dest_metrics = dest.setdefault("metrics", {})
    extra_metrics = extra.get("metrics") or {}
    if extra_steps:
        total_steps = existing_steps + extra_steps
        if not existing_steps:
            dest_metrics.update(extra_metrics)
        else:
            for field, value in extra_metrics.items():
                if value is None:
                    continue
                existing_value = dest_metrics.get(field)
                if existing_value is None:
                    dest_metrics[field] = value
                    continue
                weighted = ((existing_value * existing_steps) + (value * extra_steps)) / total_steps
                if field == "tokens_per_step":
                    dest_metrics[field] = round(weighted, 1)
                elif field == "mean_step_seconds":
                    dest_metrics[field] = round(weighted, 2)
                else:
                    dest_metrics[field] = round(weighted, 4)



def failure_fraction(failures: collections.Counter, key: str) -> float:
    total = sum(failures.values())
    if total == 0:
        return 0.0
    return failures.get(key, 0) / total


def recommend_overrides(
    raw_rate: float,
    credited_rate: float,
    coverage: float,
    witness: float,
    tokens: float,
    failures: collections.Counter,
    steps: int,
):
    overrides: list[dict] = []

    def append_override(flag: str, value: str | None = None):
        for existing in overrides:
            if existing["flag"] == flag:
                if value is not None:
                    existing["value"] = value
                return
        entry = {"flag": flag, "value": value}
        overrides.append(entry)
    notes: list[str] = []
    credit_gap = raw_rate - credited_rate
    profile = "strict_default"

    no_option_frac = failure_fraction(failures, "no_option_quote_for_choice")
    no_snippet_frac = failure_fraction(failures, "no_snippet_quote")
    coverage_frac = failure_fraction(failures, "coverage_below_tau")
    witness_frac = failure_fraction(failures, "witness_mismatch")

    evidence_limited = coverage < 0.5 or (no_option_frac + no_snippet_frac + coverage_frac) > 0.4
    witness_limited = (not evidence_limited) and witness < 0.6 and credit_gap > 0.1
    high_cost = tokens > 45000

    snippet_rate = (failures.get("no_snippet_quote", 0) / steps) if steps else 0.0
    snippet_problem = snippet_rate > 0.2

    if evidence_limited:
        profile = "evidence_limited"
        append_override("--sc-extract", "5")
        append_override("--cards-budget", "14")
        append_override("--max-learn-boosts", "1")
        if coverage < 0.35:
            append_override("--coverage-tau", "0.30")
        notes.append("Boost LEARN extraction and relax tau to recover coverage")
    elif witness_limited:
        profile = "witness_limited"
        append_override("--best-of", "10")
        append_override("--rerank", "evidence")
        append_override("--evidence-weighted-selection", None)
        notes.append("High coverage but low credited — emphasize witness-aligned selection")
    else:
        notes.append("Within default guardrails; keep strict recipe")

    if snippet_problem and profile == "strict_default":
        profile = "snippet_limited"
    if snippet_problem:
        append_override("--max-learn-boosts", "1")
        append_override("--evidence-weighted-selection", None)
        message = "Frequent no_snippet_quote failures — preserve retrieved snippets and weight evidence"
        if message not in notes:
            notes.append(message)

    if high_cost:
        append_override("--best-of", "6")
        notes.append("Token usage above 45k/step — reduce best-of to keep guardrails")

    # Deduplicate while keeping order (last occurrence wins for same flag)
    dedup: dict[str, dict] = {}
    for item in overrides:
        dedup[item["flag"]] = item
    ordered_overrides = list(dedup.values())
    return {"profile": profile, "overrides": ordered_overrides, "notes": notes}


def write_yaml(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"models": data}, handle, sort_keys=True)


def format_shell(overrides: list[dict]) -> str:
    parts = []
    for item in overrides:
        flag = item["flag"]
        value = item.get("value")
        if value is None or value == "":
            parts.append(flag)
        else:
            parts.append(f"{flag} {value}")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate per-model dial profiles")
    parser.add_argument("--runs-glob", default=DEFAULT_RUNS_GLOB, help="Glob for run JSONL files (default runs/guardrailed/*.jsonl)")
    parser.add_argument("--profiles-out", default=DEFAULT_OUTPUT, help="Where to write the aggregate profile YAML (default runs/_aggregated/model_profiles.yaml)")
    parser.add_argument("--extra-profiles", default=DEFAULT_EXTRA, help="Optional YAML with additional per-model overrides to merge (default runs/_aggregated/model_profiles_phase2.yaml)")
    parser.add_argument("--model", help="Return information for a single DeepInfra model id")
    parser.add_argument("--show", action="store_true", help="Print the computed profile(s) to stdout")
    parser.add_argument("--format", choices=["yaml", "shell"], default="yaml", help="Output format when --model is given")
    args = parser.parse_args(argv)

    runs_by_model = load_runs(args.runs_glob)
    if not runs_by_model:
        print("No guardrailed runs found. Did you run the simulator?", file=sys.stderr)
        return 1

    model_data = aggregate_model_metrics(runs_by_model)
    merge_extra_profiles(model_data, args.extra_profiles)
    write_yaml(model_data, Path(args.profiles_out))

    if args.model:
        profile = model_data.get(args.model)
        if profile is None:
            print(f"No profile data for model '{args.model}'", file=sys.stderr)
            return 2
        if args.format == "shell":
            print(format_shell(profile["overrides"]))
        else:
            yaml.safe_dump(profile, sys.stdout, sort_keys=False)
        return 0

    if args.show:
        yaml.safe_dump({"models": model_data}, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
