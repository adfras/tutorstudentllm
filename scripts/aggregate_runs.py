#!/usr/bin/env python3
"""
Aggregate simulator run logs into a single dataset.

Outputs (by default under runs/_aggregated/):
- all_steps.jsonl.gz  : concatenated JSON records with file metadata merged
- all_steps_flat.csv.gz: flattened, modeling-friendly CSV

Standard library only (no pandas dependency). Handles .jsonl and .jsonl.recovered.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import os
import sys
from glob import glob
from typing import Any, Dict, Iterable, Tuple


def find_run_files(runs_dir: str, include_aggregate: bool = False, include_json: bool = False) -> list[str]:
    patterns = [
        os.path.join(runs_dir, "**", "*.jsonl"),
        os.path.join(runs_dir, "**", "*.jsonl.recovered"),
    ]
    if include_json:
        patterns.extend([
            os.path.join(runs_dir, "**", "*.json"),
        ])
    files: list[str] = []
    for pat in patterns:
        files.extend(glob(pat, recursive=True))
    # Exclude aggregator outputs and archives
    def keep(p: str) -> bool:
        q = p.replace("\\", "/")
        if "/_aggregated/" in q:
            return False
        if "/_archive/" in q and not p.endswith(".jsonl"):
            return False
        if q.endswith(".aggregate.json") and not include_aggregate:
            return False
        # Exclude obvious reports/markdown/CSVs
        if any(q.endswith(suf) for suf in (".md", ".csv", ".gz", ".parquet")):
            return False
        return True

    files = [p for p in files if keep(p)]
    files.sort()
    return files


def flatten(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested JSON-like object into dot-keyed scalars.

    - Dicts: recurse with key prefix
    - Lists: JSON-encode to preserve all information
    - Scalars: keep as-is
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(flatten(v, key, sep=sep))
    elif isinstance(obj, (list, tuple)):
        # Preserve content exactly as JSON
        try:
            out[prefix] = json.dumps(obj, ensure_ascii=False)
        except Exception:
            out[prefix] = str(obj)
    else:
        out[prefix] = obj
    return out


def parse_jsonl(path: str) -> Iterable[Tuple[Dict[str, Any], str]]:
    """Yield (record, raw_line) from JSONL file, skipping malformed lines."""
    # Some logs might be gzipped; handle transparently
    opener = open
    if path.endswith(".gz"):
        opener = gzip.open  # type: ignore

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            yield rec, s


def parse_json_file(path: str) -> Iterable[Tuple[Dict[str, Any], None]]:
    """Yield records from a .json file.

    - If the root is a list, yield each element (wrap non-dicts under {"value": ...}).
    - If the root is a dict, yield it as a single record.
    - Otherwise, wrap as {"value": root}.
    """
    opener = open
    if path.endswith(".gz"):
        opener = gzip.open  # type: ignore
    try:
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, list):
        for i, elem in enumerate(data):
            if isinstance(elem, dict):
                rec = dict(elem)
            else:
                rec = {"value": elem}
            rec["json_array_index"] = i
            yield rec, None
    elif isinstance(data, dict):
        yield data, None
    else:
        yield {"value": data}, None


def merge_header(rec: Dict[str, Any], header: Dict[str, Any]) -> Dict[str, Any]:
    if not header:
        return rec
    merged = dict(rec)
    # Attach top-level header fields under namespaced keys to avoid collisions
    for k, v in header.items():
        if k in ("run_id", "ts"):
            # Prefer step's own run_id/ts if present
            merged.setdefault(k, v)
        else:
            merged[f"header.{k}"] = v
    return merged


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> int:
    ensure_dir(out_path)
    count = 0
    # Always gzip to save space
    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        for rec in records:
            gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(records: Iterable[Dict[str, Any]], out_path: str) -> int:
    ensure_dir(out_path)
    # First pass: collect fieldnames in stable order
    fieldnames: list[str] = []
    rows: list[Dict[str, Any]] = []
    for rec in records:
        # CSV accepts only scalars/strings; JSON-encode non-scalars
        flat = {}
        for k, v in rec.items():
            if isinstance(v, (dict, list, tuple)):
                try:
                    flat[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    flat[k] = str(v)
            else:
                flat[k] = v
        rows.append(flat)
        for k in flat.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    # Write gzipped CSV
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as gz:
        writer = csv.DictWriter(gz, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate JSONL run logs into one dataset")
    ap.add_argument("--runs-dir", default="runs", help="Directory containing run logs")
    ap.add_argument("--out-jsonl", default=os.path.join("runs", "_aggregated", "all_steps.jsonl.gz"))
    ap.add_argument("--out-csv", default=os.path.join("runs", "_aggregated", "all_steps_flat.csv.gz"))
    ap.add_argument("--include-aggregate", action="store_true", help="Include *.aggregate.json files as records")
    ap.add_argument("--include-json", action="store_true", help="Include standalone .json files (objects/arrays)")
    args = ap.parse_args()

    files = find_run_files(
        args.runs_dir,
        include_aggregate=args.include_aggregate,
        include_json=args.include_json,
    )
    if not files:
        print(f"No JSONL files found under {args.runs_dir}", file=sys.stderr)
        return 2

    all_records_for_jsonl: list[Dict[str, Any]] = []
    all_records_for_csv: list[Dict[str, Any]] = []

    for path in files:
        header: Dict[str, Any] = {}
        src_dir = os.path.dirname(path)
        src_file = os.path.basename(path)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        is_json = path.endswith(".json") and not path.endswith(".jsonl") and not path.endswith(".jsonl.recovered")

        iter_records: Iterable[Tuple[Dict[str, Any], Any]]
        if is_json:
            iter_records = parse_json_file(path)
        else:
            iter_records = parse_jsonl(path)

        for rec, raw in iter_records:
            # Identify run header
            if not is_json and rec.get("run_header") is True:
                header = {
                    "run_id": rec.get("run_id"),
                    "ts": rec.get("ts"),
                    "config": rec.get("config"),
                    "anonymization": rec.get("anonymization"),
                }
                continue

            # Attach metadata
            merged = merge_header(rec, header if not is_json else {})
            merged["source_path"] = path
            merged["source_file"] = src_file
            merged["source_dir"] = src_dir
            merged["source_kind"] = "json" if is_json else "jsonl"
            if mtime is not None:
                merged["source_mtime"] = int(mtime)

            # For JSONL output we preserve the merged record
            all_records_for_jsonl.append(merged)

            # For CSV we produce a flattened view that still preserves nested fields via JSON strings
            flat = {}
            # Always include a few useful metadata fields
            for key in ("run_id", "ts", "step"):
                if key in merged:
                    flat[key] = merged[key]
            flat.update({
                "source_path": path,
                "source_file": src_file,
                "source_dir": src_dir,
                "source_mtime": int(mtime) if mtime is not None else "",
            })

            # Flatten known nested structures selectively for readability
            for key in ("task", "answer", "evaluation", "student_usage", "tutor_usage", "evidence_telemetry"):
                if key in merged and isinstance(merged[key], dict):
                    flat.update({f"{key}.{k}": v for k, v in flatten(merged[key]).items()})
            # Header config
            if "header.config" in merged and isinstance(merged["header.config"], dict):
                flat.update({f"config.{k}": v for k, v in flatten(merged["header.config"]).items()})
            # Anonymization settings
            if "header.anonymization" in merged and isinstance(merged["header.anonymization"], dict):
                flat.update({f"anon.{k}": v for k, v in flatten(merged["header.anonymization"]).items()})

            # For standalone JSONs, flatten entire payload under 'json.*'
            if is_json:
                flat.update({f"json.{k}": v for k, v in flatten(rec).items()})

            # Include presented_stem and tools/cards fields verbatim to keep information
            for key in ("presented_stem", "tools_used", "tool_outputs", "fact_cards_before", "fact_cards_after", "card_validation", "messages"):
                if key in merged:
                    v = merged[key]
                    if isinstance(v, (dict, list, tuple)):
                        try:
                            flat[key] = json.dumps(v, ensure_ascii=False)
                        except Exception:
                            flat[key] = str(v)
                    else:
                        flat[key] = v

            all_records_for_csv.append(flat)

    # Write outputs
    n_jsonl = write_jsonl(all_records_for_jsonl, args.out_jsonl)
    n_csv = write_csv(all_records_for_csv, args.out_csv)

    # Summary to stderr
    earliest = min((r.get("ts", 0) for r in all_records_for_jsonl if isinstance(r.get("ts", 0), int)), default=None)
    latest = max((r.get("ts", 0) for r in all_records_for_jsonl if isinstance(r.get("ts", 0), int)), default=None)
    def fmt_ts(x: Any) -> str:
        try:
            return dt.datetime.utcfromtimestamp(int(x)).isoformat() + "Z"
        except Exception:
            return ""

    print(
        json.dumps(
            {
                "files_scanned": len(files),
                "records_jsonl": n_jsonl,
                "records_csv": n_csv,
                "time_range": {
                    "min_ts": earliest,
                    "min_ts_iso": fmt_ts(earliest) if earliest else "",
                    "max_ts": latest,
                    "max_ts_iso": fmt_ts(latest) if latest else "",
                },
                "out_jsonl": args.out_jsonl,
                "out_csv": args.out_csv,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
