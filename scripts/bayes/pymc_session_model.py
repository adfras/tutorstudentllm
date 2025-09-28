#!/usr/bin/env python
"""
Bayesian session-level model with PyMC.

Binomial logit: successes ~ 1 + z_talk + s(z_tokens) + (1|domain) + (1|source_dir)

Usage (after installing deps):
  .venv/bin/pip install pymc numpy pandas arviz patsy
  .venv/bin/python scripts/bayes/pymc_session_model.py --csv runs/_aggregated/session_view.csv.gz --out runs/_aggregated/pymc_session_model.nc
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/_aggregated/session_view.csv.gz")
    ap.add_argument("--out", default="runs/_aggregated/pymc_session_model.nc")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "task" in df.columns:
        df = df[(df.task.isna()) | (df.task == "mcq")]
    successes = df["correct_sum"].fillna((df["acc_final"] * df["steps_n"]).round()).astype(int)
    trials = df["steps_n"].astype(int)
    talk = df["talk_ratio_tokens"].fillna(0.5).astype(float)
    z_talk = (talk - talk.mean()) / (talk.std() + 1e-9)
    tokens = df["student_tokens_sum"].fillna(0.0).astype(float)
    z_tok = (tokens - tokens.mean()) / (tokens.std() + 1e-9)
    # Group indices
    dom_idx, dom_levels = pd.factorize(df["domain"].fillna(""))
    src_idx, src_levels = pd.factorize(df["source_dir"].fillna(""))

    import pymc as pm
    import pytensor.tensor as at  # PyMC 5 uses PyTensor backend

    with pm.Model() as m:
        # Priors
        alpha = pm.Normal("alpha", 0.0, 1.5)
        beta_talk = pm.Normal("beta_talk", 0.0, 1.0)
        # spline on tokens via quadratic term (simple, avoids spline libs)
        beta_tok1 = pm.Normal("beta_tok1", 0.0, 1.0)
        beta_tok2 = pm.Normal("beta_tok2", 0.0, 1.0)
        # random intercepts
        sd_dom = pm.Exponential("sd_dom", 1.0)
        sd_src = pm.Exponential("sd_src", 1.0)
        z_dom = pm.Normal("z_dom", 0.0, 1.0, shape=len(dom_levels))
        z_src = pm.Normal("z_src", 0.0, 1.0, shape=len(src_levels))
        u_dom = pm.Deterministic("u_dom", sd_dom * z_dom)
        u_src = pm.Deterministic("u_src", sd_src * z_src)

        eta = (
            alpha
            + beta_talk * z_talk.values
            + beta_tok1 * z_tok.values
            + beta_tok2 * (z_tok.values ** 2)
            + u_dom[dom_idx]
            + u_src[src_idx]
        )
        p = pm.Deterministic("p", pm.math.invlogit(eta))
        y = pm.Binomial("y", n=trials.values, p=p, observed=successes.values)
        idata = pm.sample(1000, tune=1000, target_accept=0.9, chains=4, cores=4)
    import arviz as az
    az.to_netcdf(idata, args.out)
    print("Saved:", args.out)
    print(az.summary(idata, var_names=["alpha","beta_talk","beta_tok1","beta_tok2","sd_dom","sd_src"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
