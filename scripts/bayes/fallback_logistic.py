#!/usr/bin/env python3
"""
Fallback logistic regression (no external deps).

Fits a binomial logistic model:
  successes ~ 1 + z_talk + z_tokens

Inputs: runs/_aggregated/session_view.csv.gz
Outputs: prints coefficients and quick diagnostics.
"""
from __future__ import annotations
import csv, gzip, math, sys
from typing import List, Tuple


def read_session_view(path: str) -> Tuple[List[float], List[float], List[int], List[int]]:
    X_talk: List[float] = []; X_tok: List[float] = []
    y: List[int] = []; n: List[int] = []
    # increase csv field size
    try:
        import csv as _csv
        _csv.field_size_limit(10_000_000)
    except Exception:
        pass
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                steps = int(float(row.get('steps_n') or 0))
                if steps <= 0:
                    continue
                # success counts
                csum = row.get('correct_sum')
                if csum is None or csum == '':
                    acc = float(row.get('acc_final') or 0.0)
                    succ = int(round(acc * steps))
                else:
                    succ = int(csum)
                talk = row.get('talk_ratio_tokens')
                if talk in (None, ''):
                    t = 0.5
                else:
                    t = float(talk)
                tok = float(row.get('student_tokens_sum') or 0.0)
            except Exception:
                continue
            y.append(succ); n.append(steps)
            X_talk.append(t); X_tok.append(tok)
    # standardize predictors
    def zscore(a: List[float]) -> List[float]:
        if not a:
            return []
        m = sum(a)/len(a)
        v = sum((x-m)*(x-m) for x in a)/max(1, len(a)-1)
        s = math.sqrt(v) if v>0 else 1.0
        return [(x-m)/s for x in a]
    z_talk = zscore(X_talk)
    z_tok = zscore(X_tok)
    return z_talk, z_tok, y, n


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0/(1.0+z)
    else:
        z = math.exp(x); return z/(1.0+z)


def fit_logit(z_talk: List[float], z_tok: List[float], y: List[int], n: List[int], lr: float=0.05, iters: int=5000) -> List[float]:
    # w = [intercept, b_talk, b_tok]
    w = [0.0, 0.0, 0.0]
    m = len(y)
    for it in range(iters):
        g0=g1=g2=0.0
        for i in range(m):
            eta = w[0] + w[1]*z_talk[i] + w[2]*z_tok[i]
            p = sigmoid(eta)
            g = (y[i] - n[i]*p)
            g0 += g
            g1 += g * z_talk[i]
            g2 += g * z_tok[i]
        # update (ascend log-lik)
        w[0] += lr * g0/m
        w[1] += lr * g1/m
        w[2] += lr * g2/m
        # small step decay
        if (it+1) % 1000 == 0:
            lr *= 0.7
    return w


def main() -> int:
    path = sys.argv[1] if len(sys.argv)>1 else 'runs/_aggregated/session_view.csv.gz'
    z_talk, z_tok, y, n = read_session_view(path)
    if not y:
        print('No rows found in', path)
        return 2
    w = fit_logit(z_talk, z_tok, y, n)
    # quick diagnostics
    import statistics as st
    probs = []
    for i in range(len(y)):
        eta = w[0] + w[1]*z_talk[i] + w[2]*z_tok[i]
        probs.append(sigmoid(eta))
    # pseudo-R2: variance of mean-corrected success rate vs. predicted
    ybar = sum(y_i/n_i for y_i,n_i in zip(y,n))/len(y)
    r2 = 1.0 - (sum((y_i/n_i - p)**2 for y_i,n_i,p in zip(y,n,probs)) / sum((y_i/n_i - ybar)**2 for y_i,n_i in zip(y,n)))
    print('Logistic (binomial) coefficients:')
    print('  Intercept:', round(w[0],4))
    print('  z_talk   :', round(w[1],4), '(positive → higher tutor share predicts higher success)')
    print('  z_tokens :', round(w[2],4), '(positive → more student tokens predicts higher success)')
    print('Diagnostics:')
    print('  Runs:', len(y))
    print('  Mean success rate:', round(ybar,3))
    print('  Pseudo-R2 (naive):', round(r2,3))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

