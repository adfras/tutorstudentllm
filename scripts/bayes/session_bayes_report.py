#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dirs(root: Path) -> dict[str, Path]:
    d = {
        "root": root,
        "models": root / "models",
        "figs": root / "figs",
        "reports": root / "reports",
        "tables": root / "tables",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def summarize_idata(idata, out_csv: Path, params: list[str]) -> pd.DataFrame:
    import arviz as az
    summ = az.summary(idata, var_names=params)
    summ.to_csv(out_csv)
    return summ


def plot_posteriors(idata, out_dir: Path, params: list[str]) -> list[Path]:
    import arviz as az
    import matplotlib.pyplot as plt
    paths: list[Path] = []
    az.style.use("arviz-darkgrid")
    figs = az.plot_posterior(idata, var_names=params, hdi_prob=0.95, kind="hist")
    # az.plot_posterior may return a numpy array of axes
    try:
        import numpy as np
        axes = np.array(figs).ravel().tolist()
        fig = axes[0].get_figure()
    except Exception:
        fig = getattr(figs, 'figure', None)
    if fig is not None:
        out = out_dir / "posterior_params.png"
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def token_curve_plot(trace_df: pd.DataFrame, out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        xs = np.linspace(-2.5, 2.5, 200)
        # median curve on z_tokens: eta = beta_tok1*z + beta_tok2*z^2 (others at 0)
        b1 = float(trace_df.loc["beta_tok1", "mean"]) if "beta_tok1" in trace_df.index else 0.0
        b2 = float(trace_df.loc["beta_tok2", "mean"]) if "beta_tok2" in trace_df.index else 0.0
        ys = 1.0 / (1.0 + np.exp(-(b1*xs + b2*(xs**2))))
        plt.figure(figsize=(5,3))
        plt.plot(xs, ys, label="tokens effect (median)")
        plt.xlabel("z_tokens (student_tokens_sum, standardized)")
        plt.ylabel("Pr(success) partial")
        plt.title("Tokens → success (partial, other terms 0)")
        plt.grid(True, alpha=0.3)
        out = out_dir / "token_effect_curve.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        return out
    except Exception:
        return None


def write_report(report_path: Path, model_path: Path, summary_csv: Path, posterior_png: Path | None, token_png: Path | None, notes: dict) -> None:
    ts = dt.datetime.utcnow().isoformat() + "Z"
    lines = []
    lines.append(f"# Bayesian Session Model — {ts}")
    lines.append("")
    lines.append("Directories")
    lines.append(f"- models: {model_path.parent}")
    lines.append(f"- tables: {summary_csv.parent}")
    if posterior_png:
        lines.append(f"- figs: {posterior_png.parent}")
    lines.append("")
    lines.append("Artifacts")
    lines.append(f"- Model (netCDF): {model_path}")
    lines.append(f"- Summary table (CSV): {summary_csv}")
    if posterior_png:
        lines.append(f"- Posterior plots: {posterior_png.name}")
    if token_png:
        lines.append(f"- Token effect curve: {token_png.name}")
    lines.append("")
    lines.append("Key Findings (provisional)")
    bt = notes.get("beta_talk", {})
    lines.append(f"- Tutor talk ratio (z_talk): mean {bt.get('mean'):.3f}, 95% HDI [{bt.get('hdi3'):.3f}, {bt.get('hdi97'):.3f}] (positive implies higher tutor share → higher success).")
    b1 = notes.get("beta_tok1", {})
    b2 = notes.get("beta_tok2", {})
    lines.append(f"- Student tokens z (linear): mean {b1.get('mean'):.3f}, 95% HDI [{b1.get('hdi3'):.3f}, {b1.get('hdi97'):.3f}].")
    lines.append(f"- Student tokens z^2 (curvature): mean {b2.get('mean'):.3f}, 95% HDI [{b2.get('hdi3'):.3f}, {b2.get('hdi97'):.3f}] (diminishing/curved effect).")
    lines.append("")
    lines.append("Sampler Notes")
    lines.append(f"- Target accept: {notes.get('target_accept')} | Tune: {notes.get('tune')} | Draws: {notes.get('draws')}")
    if notes.get("divergences", 0) > 0:
        lines.append(f"- Divergences: {notes.get('divergences')} — consider higher target_accept or reparameterization.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/_aggregated/session_view.csv.gz")
    ap.add_argument("--out-root", default="runs/_aggregated/bayes")
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    d = ensure_dirs(root)

    df = pd.read_csv(args.csv)
    if "task" in df.columns:
        df = df[(df.task.isna()) | (df.task == "mcq")]
    successes = df["correct_sum"].fillna((df["acc_final"] * df["steps_n"]).round()).astype(int)
    trials = df["steps_n"].astype(int)
    talk = df["talk_ratio_tokens"].fillna(0.5).astype(float)
    z_talk = (talk - talk.mean()) / (talk.std() + 1e-9)
    tokens = df["student_tokens_sum"].fillna(0.0).astype(float)
    z_tok = (tokens - tokens.mean()) / (tokens.std() + 1e-9)
    # controls
    import numpy as np
    n_turns = df.get("n_turns", pd.Series([np.nan]*len(df))).fillna(0).astype(float)
    z_n_turns = (n_turns - n_turns.mean()) / (n_turns.std() + 1e-9)
    dur = df.get("duration_seconds", pd.Series([np.nan]*len(df))).fillna(df.get("mean_step_seconds", 0)*df["steps_n"]).astype(float)
    z_dur = (dur - dur.mean()) / (dur.std() + 1e-9)
    first_ok = df.get("first_attempt_correct", pd.Series([0]*len(df))).fillna(0).astype(int)
    dom_idx, dom_levels = pd.factorize(df["domain"].fillna(""))
    src_idx, src_levels = pd.factorize(df["source_dir"].fillna(""))

    # Fit PyMC model
    import pymc as pm
    import pytensor.tensor as at
    tune = 800 if args.fast else args.tune
    draws = 800 if args.fast else args.draws
    target = args.target_accept
    with pm.Model() as m:
        # Priors (slightly tighter on z-scales)
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta_talk = pm.Normal("beta_talk", 0.0, 0.8)
        beta_tok1 = pm.Normal("beta_tok1", 0.0, 0.8)
        beta_tok2 = pm.Normal("beta_tok2", 0.0, 0.8)
        beta_inter = pm.Normal("beta_inter", 0.0, 0.8)  # z_talk : z_tok
        beta_turns = pm.Normal("beta_turns", 0.0, 0.8)
        beta_dur = pm.Normal("beta_dur", 0.0, 0.8)
        beta_first = pm.Normal("beta_first", 0.0, 0.8)
        # Random effects: intercepts and talk slopes by domain; intercepts by source_dir
        sd_dom = pm.Exponential("sd_dom", 1.0)
        sd_src = pm.Exponential("sd_src", 1.0)
        sd_talk_dom = pm.Exponential("sd_talk_dom", 1.0)
        z_dom = pm.Normal("z_dom", 0.0, 1.0, shape=len(dom_levels))
        z_src = pm.Normal("z_src", 0.0, 1.0, shape=len(src_levels))
        z_talk_dom = pm.Normal("z_talk_dom", 0.0, 1.0, shape=len(dom_levels))
        u_dom = pm.Deterministic("u_dom", sd_dom * z_dom)
        u_src = pm.Deterministic("u_src", sd_src * z_src)
        u_talk_dom = pm.Deterministic("u_talk_dom", sd_talk_dom * z_talk_dom)
        # Orthogonalized quadratic term for tokens
        ztok2_raw = z_tok.values ** 2
        ztok2 = (ztok2_raw - ztok2_raw.mean()) / (ztok2_raw.std() + 1e-9)
        # Linear predictor
        eta = (
            alpha
            + (beta_talk + u_talk_dom[dom_idx]) * z_talk.values
            + beta_tok1 * z_tok.values
            + beta_tok2 * ztok2
            + beta_inter * (z_talk.values * z_tok.values)
            + beta_turns * z_n_turns.values
            + beta_dur * z_dur.values
            + beta_first * first_ok.values
            + u_dom[dom_idx]
            + u_src[src_idx]
        )
        p = pm.Deterministic("p", pm.math.invlogit(eta))
        y = pm.Binomial("y", n=trials.values, p=p, observed=successes.values)
        step = pm.NUTS(target_accept=target, max_treedepth=15)
        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=4, step=step, progressbar=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = d["models"] / f"pymc_session_model_{ts}.nc"
    import arviz as az
    az.to_netcdf(idata, model_path)

    params = [
        "alpha", "beta_talk", "beta_tok1", "beta_tok2", "beta_inter",
        "beta_turns", "beta_dur", "beta_first",
        "sd_dom", "sd_src", "sd_talk_dom"
    ]
    summary_csv = d["tables"] / f"session_posterior_summary_{ts}.csv"
    summ = summarize_idata(idata, summary_csv, params)

    # figures
    posterior_pngs = plot_posteriors(idata, d["figs"], params)
    token_png = token_curve_plot(summ, d["figs"])

    # notes for report (incl. optimum z*)
    hdi3 = summ["hdi_3%"].to_dict(); hdi97 = summ["hdi_97%"].to_dict(); mean = summ["mean"].to_dict()
    import numpy as np
    try:
        b1 = idata.posterior["beta_tok1"].values.reshape(-1)
        b2 = idata.posterior["beta_tok2"].values.reshape(-1)
        zstar = -b1/(2*(b2+1e-9))
        zstar = zstar[np.isfinite(zstar)]
        zstar_mean = float(np.mean(zstar)) if zstar.size else 0.0
        zstar_hdi = (float(np.quantile(zstar, 0.03)), float(np.quantile(zstar, 0.97))) if zstar.size else (0.0,0.0)
    except Exception:
        zstar_mean, zstar_hdi = 0.0, (0.0, 0.0)
    notes = {
        "beta_talk": {"mean": float(mean.get("beta_talk", 0)), "hdi3": float(hdi3.get("beta_talk", 0)), "hdi97": float(hdi97.get("beta_talk", 0))},
        "beta_tok1": {"mean": float(mean.get("beta_tok1", 0)), "hdi3": float(hdi3.get("beta_tok1", 0)), "hdi97": float(hdi97.get("beta_tok1", 0))},
        "beta_tok2": {"mean": float(mean.get("beta_tok2", 0)), "hdi3": float(hdi3.get("beta_tok2", 0)), "hdi97": float(hdi97.get("beta_tok2", 0))},
        "target_accept": target, "tune": tune, "draws": draws,
        "divergences": int(np.sum(np.array(idata.sample_stats["diverging"].values))) if "diverging" in idata.sample_stats else 0,
        "z_star": {"mean": zstar_mean, "hdi3": zstar_hdi[0], "hdi97": zstar_hdi[1]},
    }

    # Guardrails JSON (dataset-level)
    mean_tokens = float(df["student_tokens_sum"].fillna(0.0).mean())
    sd_tokens = float(df["student_tokens_sum"].fillna(0.0).std(ddof=1))
    mean_steps = float(df["steps_n"].mean()) if len(df) else 1.0
    z_low = zstar_mean - 0.5
    z_high = zstar_mean + 0.5
    opt_tokens = mean_tokens + zstar_mean * sd_tokens
    low_tokens = mean_tokens + z_low * sd_tokens
    high_tokens = mean_tokens + z_high * sd_tokens
    guardrails = {
        "z_star": {"mean": zstar_mean, "hdi3": zstar_hdi[0], "hdi97": zstar_hdi[1]},
        "tokens_run": {
            "mean": mean_tokens, "sd": sd_tokens,
            "opt": opt_tokens, "band_low": low_tokens, "band_high": high_tokens,
        },
        "tokens_per_step": {
            "opt": (opt_tokens/mean_steps if mean_steps else None),
            "band_low": (low_tokens/mean_steps if mean_steps else None),
            "band_high": (high_tokens/mean_steps if mean_steps else None),
            "mean_steps": mean_steps,
        }
    }
    (d["tables"] / "guardrails.json").write_text(json.dumps(guardrails, indent=2), encoding="utf-8")

    # Per-domain talk slopes: beta_talk + u_talk_dom
    talk_slopes_rows = []
    try:
        bt = idata.posterior["beta_talk"].values.reshape(-1, 1)
        u = idata.posterior["u_talk_dom"].values  # shape (chains, draws, Dom)
        arr = u.reshape(-1, u.shape[-1])  # (samples, Dom)
        slopes = bt + arr  # broadcast
        for j, dom in enumerate(dom_levels):
            sj = slopes[:, j]
            sj = sj[np.isfinite(sj)]
            if sj.size:
                m = float(np.mean(sj)); s = float(np.std(sj, ddof=1))
                h3 = float(np.quantile(sj, 0.03)); h97 = float(np.quantile(sj, 0.97))
                ppos = float(np.mean(sj > 0))
                talk_slopes_rows.append({"domain": str(dom), "mean": m, "sd": s, "hdi_3%": h3, "hdi_97%": h97, "prob_positive": ppos})
    except Exception:
        pass
    if talk_slopes_rows:
        import csv
        out = d["tables"] / f"talk_slopes_by_domain_{ts}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(talk_slopes_rows[0].keys()))
            w.writeheader(); w.writerows(talk_slopes_rows)

    report_md = d["reports"] / f"session_bayes_report_{ts}.md"
    write_report(report_md, model_path, summary_csv, posterior_pngs[0] if posterior_pngs else None, token_png, notes)
    print(json.dumps({
        "model": str(model_path),
        "summary_csv": str(summary_csv),
        "report": str(report_md),
        "figs": [str(p) for p in (posterior_pngs + ([token_png] if token_png else []))],
        "guardrails": str(d["tables"] / "guardrails.json"),
        "talk_slopes_by_domain": (str(out) if talk_slopes_rows else None),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
