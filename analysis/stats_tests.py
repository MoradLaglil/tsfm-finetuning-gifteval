"""Statistical tests behind the significance markers of the paper tables.

For each model and metric, the input is one row per configuration (dataset, frequency, horizon) and one column per
strategy, with scores normalized by Seasonal Naive and averaged over seeds (same data as analysis/aggregate.py).
Lower is better; alpha = 0.05.

  1. Friedman test across the four strategies (average ranks reported, 1 = best).
  2. Only if the Friedman test is significant: two-sided Wilcoxon signed-rank tests on the 6 pairs of strategies,
     with Holm correction over the 6 pairs.

Families:
  all          all configurations (the "All" row), tested on its own;
  size         small / medium / large datasets                  \\
  n_series     small / medium / large number of series           } Friedman p-values Holm-corrected across
  mean_length  short / medium / long series                     /  the 3 groups of each grouping

    python analysis/stats_tests.py                 # results used in the paper (results/paper/)
    python analysis/stats_tests.py --source runs   # your own runs (results/chronos2/, results/ttm/)

Markers compare each fine-tuning strategy with zero-shot: "▲" significantly lower error, "▼" significantly higher
error, "–" not significant. CSV files are written to analysis/output/<source>/stats/.
Needs numpy, pandas and scipy.
"""

import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from aggregate import GROUPINGS, HERE, LABELS, METRICS, MODELS, ORDER, ROOT, STRATEGIES, load

ALPHA = 0.05
UP, DOWN, NS = "▲", "▼", "–"


def holm(pvals):
    """Holm step-down adjusted p-values (NaN entries are ignored and kept as NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    idx = np.argsort(p[ok])
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def per_configuration(df, model, metric):
    """Wide table: one row per configuration, one column per strategy (mean over seeds)."""
    d = df[df.model == model]
    keys = ["dataset", "term", *GROUPINGS]
    w = d.groupby([*keys, "strategy"])[metric].mean().unstack("strategy")
    missing = [s for s in STRATEGIES if s not in w.columns]
    if missing:
        return None, f"missing strategies {[LABELS[s] for s in missing]}"
    w = w[STRATEGIES]
    w = w[np.isfinite(w).all(axis=1)].reset_index()  # configurations available for all four strategies
    return w, None


def friedman(df):
    X = df[STRATEGIES].to_numpy()
    if len(X) < 2:
        chi2, p = np.nan, np.nan
    else:
        chi2, p = friedmanchisquare(*X.T)
    ranks = rankdata(X, axis=1).mean(axis=0) if len(X) else [np.nan] * len(STRATEGIES)
    return dict(n=len(X), chi2=chi2, p=p, **{f"rank_{LABELS[s]}": r for s, r in zip(STRATEGIES, ranks)})


def pairwise(df):
    rows = []
    for a, b in combinations(STRATEGIES, 2):
        d = df[a].to_numpy() - df[b].to_numpy()
        if np.all(d == 0):
            stat, p = np.nan, 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = wilcoxon(df[a], df[b], alternative="two-sided")
        med = np.median(d)
        rows.append(dict(A=LABELS[a], B=LABELS[b], n=len(d), wins_A=int((d < 0).sum()), wins_B=int((d > 0).sum()),
                         ties=int((d == 0).sum()), median_diff_A_minus_B=med, W=stat, p_raw=p,
                         better=LABELS[a] if med < 0 else (LABELS[b] if med > 0 else "tie")))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"])  # Holm over the 6 pairs
    out["significant"] = out["p_holm"] < ALPHA
    return out


def run_family(tables, family):
    """Friedman per (model, metric, group), Holm across the groups of the family, Wilcoxon pairs where significant."""
    F, P = [], []
    for (model, metric), df in tables.items():
        groups = [("all", df)] if family == "all" else \
            [(g, df[df[family] == g]) for g in ORDER[family] if (df[family] == g).any()]
        fr = pd.DataFrame([dict(family=family, model=MODELS[model], metric=metric, group=g, **friedman(d))
                           for g, d in groups])
        fr["p_holm_groups"] = holm(fr["p"]) if len(fr) > 1 else fr["p"]
        fr["significant"] = fr["p_holm_groups"] < ALPHA
        for (g, d), sig in zip(groups, fr["significant"]):
            if sig:
                pw = pairwise(d)
                for col, val in [("group", g), ("metric", metric), ("model", MODELS[model]), ("family", family)]:
                    pw.insert(0, col, val)
                P.append(pw)
        F.append(fr)
    return pd.concat(F, ignore_index=True), (pd.concat(P, ignore_index=True) if P else pd.DataFrame())


def markers(F, P):
    """Strategy vs zero-shot marker for every (family, model, metric, group, fine-tuning strategy)."""
    rows = []
    for _, f in F.iterrows():
        for s in STRATEGIES[1:]:
            mark = NS
            if f["significant"]:
                p = P[(P.family == f.family) & (P.model == f.model) & (P.metric == f.metric) & (P.group == f.group)
                      & (P.A == LABELS["zero-shot"]) & (P.B == LABELS[s])].iloc[0]
                if p["significant"]:
                    mark = UP if p["better"] == LABELS[s] else (DOWN if p["better"] == LABELS["zero-shot"] else NS)
            rows.append(dict(family=f.family, model=f.model, metric=f.metric, group=f.group, strategy=LABELS[s],
                             marker=mark))
    return pd.DataFrame(rows)


def fmt_p(p):
    return "n/a" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["paper", "runs"], default="paper")
    args = p.parse_args()

    df = load(args.source)
    tables = {}
    for model in MODELS:
        for metric in METRICS:
            if model not in set(df.model):
                continue
            w, err = per_configuration(df, model, metric)
            if err:
                print(f"{MODELS[model]} {metric}: skipped ({err})")
                continue
            tables[(model, metric)] = w
    if not tables:
        raise SystemExit("Nothing to test: each model needs results for all four strategies.")
    for (model, metric), w in tables.items():
        if metric == METRICS[0]:
            print(f"{MODELS[model]}: {len(w)} configurations")

    results = [run_family(tables, fam) for fam in ["all", *GROUPINGS]]
    F = pd.concat([r[0] for r in results], ignore_index=True)
    P = pd.concat([r[1] for r in results], ignore_index=True)
    M = markers(F, P)

    for fam in ["all", *GROUPINGS]:
        title = "all configurations" if fam == "all" else f"{fam} groups (Friedman p Holm-corrected across groups)"
        print(f"\n### {title}\n")
        print("| model | metric | group | n | Friedman p | " + " | ".join(LABELS[s] for s in STRATEGIES[1:])
              + " (vs Zero-Shot) |")
        print("|---|---|---|---|---|---|---|---|")
        for _, f in F[F.family == fam].iterrows():
            m = M[(M.family == fam) & (M.model == f.model) & (M.metric == f.metric) & (M.group == f.group)]
            print(f"| {f.model} | {f.metric} | {f.group} | {f.n} | {fmt_p(f.p_holm_groups)} | "
                  + " | ".join(m.marker) + " |")

    out_dir = HERE / "output" / args.source / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    F.to_csv(out_dir / "friedman.csv", index=False)
    P.to_csv(out_dir / "wilcoxon_pairs.csv", index=False)
    M.to_csv(out_dir / "markers_vs_zero_shot.csv", index=False)
    print(f"\n{UP}/{DOWN}: significantly lower/higher error than Zero-Shot (Holm over 6 pairs, alpha = {ALPHA}); "
          f"{NS}: not significant or Friedman not significant.")
    print(f"CSV files written to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
