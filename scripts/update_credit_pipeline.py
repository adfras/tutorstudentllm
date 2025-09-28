#!/usr/bin/env python3
"""End-to-end pipeline for evidence credit modeling and dial recommendations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
AGG_DATA = REPO_ROOT / "runs/_aggregated/run_training_data_clean.csv"
MODEL_PATH = REPO_ROOT / "runs/_aggregated/model_credit_ridge_poly.json"
COEFF_PATH = REPO_ROOT / "runs/_aggregated/model_credit_ridge_poly_coeffs.json"
PRED_PATH = REPO_ROOT / "runs/_aggregated/model_predictions.csv"
RECS_PATH = REPO_ROOT / "runs/_aggregated/dial_recommendations.csv"


def run(cmd: list[str]) -> None:
    print(f"[pipeline] running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def aggregate_runs() -> None:
    run([str(REPO_ROOT / ".venv/bin/python"), "-m", "scripts.aggregate_runs"])
    run([str(REPO_ROOT / ".venv/bin/python"), "-m", "scripts.session_view"])


def build_dataset_and_model() -> None:
    data_source = REPO_ROOT / "runs/_aggregated/run_training_data.csv"
    if not data_source.exists():
        raise SystemExit("run_training_data.csv is missing; ensure aggregation succeeded")
    df = pd.read_csv(data_source)

    numeric_base = [
        "correct_rate",
        "witness_rate",
        "no_snippet_rate",
        "avg_coverage",
        "tokens_per_step",
        "self_consistency_n",
        "best_of_n",
        "sc_extract_n",
        "q_min",
        "cards_budget",
        "max_learn_boosts",
        "coverage_tau",
    ]
    bool_cols = ["idk_enabled", "require_citations", "use_fact_cards", "evidence_weighted_selection"]

    df["provider"] = df["provider"].fillna("unknown")
    df["stage"] = df["stage"].fillna("unknown")

    for col in numeric_base:
        if col not in df:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    for col in bool_cols:
        if col not in df:
            df[col] = 0
        df[col] = df[col].fillna(False).astype(int)

    mask = df["steps"].fillna(0) >= 5
    mask &= df["credited_rate"].notna()
    mask &= ~df[numeric_base].isna().any(axis=1)
    df = df[mask].reset_index(drop=True)

    df.to_csv(AGG_DATA, index=False)

    numeric_data = df[numeric_base].to_numpy(dtype=float)
    bool_data = df[bool_cols].to_numpy(dtype=float)
    providers = sorted(df["provider"].unique())
    stages = sorted(df["stage"].unique())
    prov_map = {p: i for i, p in enumerate(providers)}
    stage_map = {s: i for i, s in enumerate(stages)}

    mean = numeric_data.mean(axis=0)
    std = numeric_data.std(axis=0)
    std[std == 0] = 1.0
    numeric_std = (numeric_data - mean) / std

    poly_idx = list(range(min(5, len(numeric_base))))
    cols = []
    feature_names = []

    for i, name in enumerate(numeric_base):
        feature_names.append(name)
        cols.append(numeric_std[:, i])

    for idx in poly_idx:
        name = numeric_base[idx]
        feature_names.append(f"{name}^2")
        cols.append(numeric_std[:, idx] ** 2)

    for i in range(len(poly_idx)):
        for j in range(i + 1, len(poly_idx)):
            fi = numeric_base[poly_idx[i]]
            fj = numeric_base[poly_idx[j]]
            feature_names.append(f"{fi}*{fj}")
            cols.append(numeric_std[:, poly_idx[i]] * numeric_std[:, poly_idx[j]])

    def add_interaction(a: str, b: str, label: str) -> None:
        if a in numeric_base and b in numeric_base:
            ia, ib = numeric_base.index(a), numeric_base.index(b)
            feature_names.append(label)
            cols.append(numeric_std[:, ia] * numeric_std[:, ib])

    add_interaction("witness_rate", "cards_budget", "witness_rate*cards_budget")
    add_interaction("no_snippet_rate", "cards_budget", "no_snippet_rate*cards_budget")
    add_interaction("tokens_per_step", "best_of_n", "tokens_per_step*best_of_n")

    X_numeric = np.column_stack(cols) if cols else np.zeros((len(df), 0))

    feature_names.extend(bool_cols)
    X_bool = bool_data

    prov_arr = np.zeros((len(df), len(providers)))
    stage_arr = np.zeros((len(df), len(stages)))
    for idx, (prov, stage) in enumerate(zip(df["provider"], df["stage"])):
        prov_arr[idx, prov_map[prov]] = 1.0
        stage_arr[idx, stage_map[stage]] = 1.0
    feature_names.extend([f"provider={p}" for p in providers])
    feature_names.extend([f"stage={s}" for s in stages])

    X_all = np.hstack([
        np.ones((len(df), 1)),
        X_numeric,
        X_bool,
        prov_arr,
        stage_arr,
    ])

    y = df["credited_rate"].to_numpy(dtype=float)

    rng = np.random.default_rng(4242)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    train_size = int(len(df) * 0.8)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train = X_all[train_idx]
    y_train = y[train_idx]
    X_test = X_all[test_idx]
    y_test = y[test_idx]

    lam = 1.0
    I = np.eye(X_train.shape[1])
    I[0, 0] = 0
    beta = np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ y_train)

    test_pred = X_test @ beta
    mae = float(np.mean(np.abs(test_pred - y_test)))
    ss_res = float(np.sum((y_test - test_pred) ** 2))
    y_mean = float(np.mean(y_test))
    ss_tot = float(np.sum((y_test - y_mean) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    artifact = {
        "numeric_base": numeric_base,
        "numeric_mean": mean.tolist(),
        "numeric_std": std.tolist(),
        "bool_cols": bool_cols,
        "providers": providers,
        "stages": stages,
        "feature_names": ["intercept"] + feature_names,
        "beta": beta.tolist(),
        "lambda": lam,
        "metrics": {
            "samples_total": int(len(df)),
            "train_samples": int(train_size),
            "test_samples": int(len(df) - train_size),
            "holdout_mae": mae,
            "holdout_r2": r2,
        },
    }
    MODEL_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    coeffs = sorted(
        zip(artifact["feature_names"], beta.tolist()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:20]
    COEFF_PATH.write_text(json.dumps(coeffs, indent=2), encoding="utf-8")


def run_diagnostics() -> None:
    run([str(REPO_ROOT / ".venv/bin/python"), str(REPO_ROOT / "scripts/credit_model_diagnostics.py")])


def build_recommendations() -> None:
    if not PRED_PATH.exists():
        raise SystemExit("model_predictions.csv missing; diagnostics must run first")
    preds = pd.read_csv(PRED_PATH)

    dial_cols = [
        "provider",
        "stage",
        "self_consistency_n",
        "best_of_n",
        "sc_extract_n",
        "q_min",
        "cards_budget",
        "max_learn_boosts",
        "idk_enabled",
        "evidence_weighted_selection",
    ]
    for col in dial_cols:
        if col not in preds:
            preds[col] = 0

    grouped = (
        preds.groupby(dial_cols)
        .agg(
            runs=("credited_rate", "size"),
            steps=("steps", "sum"),
            credited_mean=("credited_rate", "mean"),
            predicted_mean=("predicted_credited_rate", "mean"),
            mae=("predicted_credited_rate", lambda s: np.mean(np.abs(s - preds.loc[s.index, "credited_rate"].to_numpy()))),
        )
        .reset_index()
    )

    grouped = grouped[grouped["runs"] >= 2]
    grouped = grouped.sort_values(["provider", "predicted_mean"], ascending=[True, False])

    top_rows = grouped.groupby("provider").head(3).reset_index(drop=True)
    top_rows.to_csv(RECS_PATH, index=False)


def main() -> int:
    aggregate_runs()
    build_dataset_and_model()
    run_diagnostics()
    build_recommendations()
    print("[pipeline] finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
