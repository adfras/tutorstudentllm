#!/usr/bin/env python3
"""Helper to score run-level logs with the ridge predictor and emit diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA = Path("runs/_aggregated/run_training_data_clean.csv")
DEFAULT_MODEL = Path("runs/_aggregated/model_credit_ridge_poly.json")


def load_artifact(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"model artifact not found: {path}") from exc


def load_dataset(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"dataset not found: {path}") from exc
    return df


def build_features(df: pd.DataFrame, art: dict) -> Tuple[np.ndarray, pd.DataFrame]:
    numeric_base = art["numeric_base"]
    bool_cols = art["bool_cols"]
    providers = art["providers"]
    stages = art["stages"]
    mean = np.array(art["numeric_mean"], dtype=float)
    std = np.array(art["numeric_std"], dtype=float)

    df = df.copy()
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
    filtered = df[mask].reset_index(drop=True)

    numeric_data = filtered[numeric_base].to_numpy(dtype=float)
    numeric_std = (numeric_data - mean) / std

    bool_data = filtered[bool_cols].to_numpy(dtype=float)

    # Polynomial features for first five numeric columns
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

    X_numeric = np.column_stack(cols) if cols else np.zeros((len(filtered), 0))

    feature_names.extend(bool_cols)
    X_bool = bool_data

    prov_map = {p: i for i, p in enumerate(providers)}
    stage_map = {s: i for i, s in enumerate(stages)}
    prov_arr = np.zeros((len(filtered), len(providers)))
    stage_arr = np.zeros((len(filtered), len(stages)))
    for idx, (prov, stage) in enumerate(zip(filtered["provider"], filtered["stage"])):
        prov_arr[idx, prov_map[prov]] = 1.0
        stage_arr[idx, stage_map[stage]] = 1.0

    feature_names.extend([f"provider={p}" for p in providers])
    feature_names.extend([f"stage={s}" for s in stages])

    X = np.hstack([
        np.ones((len(filtered), 1)),
        X_numeric,
        X_bool,
        prov_arr,
        stage_arr,
    ])

    feature_names = ["intercept"] + feature_names
    return X, filtered, feature_names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score run-level logs with the ridge predictor")
    ap.add_argument("--data", default=DEFAULT_DATA, type=Path, help="CSV with run-level features (default: runs/_aggregated/run_training_data_clean.csv)")
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path, help="Model artifact JSON (default: runs/_aggregated/model_credit_ridge_poly.json)")
    ap.add_argument("--out", default=Path("runs/_aggregated/model_predictions.csv"), type=Path, help="Where to write per-run predictions")
    args = ap.parse_args(argv)

    artifact = load_artifact(args.model)
    df = load_dataset(args.data)
    X, filtered_df, feature_names = build_features(df, artifact)

    beta = np.array(artifact["beta"], dtype=float)
    if X.shape[1] != len(beta):
        raise SystemExit("feature mismatch between dataset and model artifact")

    preds = X @ beta
    filtered_df = filtered_df.assign(predicted_credited_rate=preds)

    filtered_df.to_csv(args.out, index=False)

    resid = preds - filtered_df["credited_rate"].to_numpy()
    abs_err = np.abs(resid)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    y = filtered_df["credited_rate"].to_numpy()
    r2 = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) else float("nan")

    provider_grp = filtered_df.groupby("provider").agg(
        n_runs=("credited_rate", "size"),
        credited_mean=("credited_rate", "mean"),
        predicted_mean=("predicted_credited_rate", "mean"),
        mae=("predicted_credited_rate", lambda s: np.mean(np.abs(s - filtered_df.loc[s.index, "credited_rate"].to_numpy()))),
    ).reset_index().sort_values("mae")

    stage_grp = filtered_df.groupby("stage").agg(
        n_runs=("credited_rate", "size"),
        credited_mean=("credited_rate", "mean"),
        predicted_mean=("predicted_credited_rate", "mean"),
        mae=("predicted_credited_rate", lambda s: np.mean(np.abs(s - filtered_df.loc[s.index, "credited_rate"].to_numpy()))),
    ).reset_index().sort_values("mae")

    provider_path = args.out.with_name(args.out.stem + "_by_provider.csv")
    stage_path = args.out.with_name(args.out.stem + "_by_stage.csv")
    provider_grp.to_csv(provider_path, index=False)
    stage_grp.to_csv(stage_path, index=False)

    summary = {
        "runs_scored": int(len(filtered_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "abs_error_q25": float(np.quantile(abs_err, 0.25)),
        "abs_error_q50": float(np.quantile(abs_err, 0.5)),
        "abs_error_q75": float(np.quantile(abs_err, 0.75)),
        "predictions_csv": str(args.out),
        "provider_summary_csv": str(provider_path),
        "stage_summary_csv": str(stage_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
