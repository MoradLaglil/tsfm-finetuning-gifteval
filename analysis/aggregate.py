"""Aggregate GIFT-Eval results into the tables of the paper.

Every configuration (dataset, frequency, horizon) is normalized by the Seasonal Naive baseline of GIFT-Eval:
    MWQL_n = WQL_model / WQL_naive        MAPE_n = MAPE_model / MAPE_naive      (lower is better, 1 = naive)
Scores are then averaged over configurations within each group (dataset size, number of series, mean series
length, domain x horizon) and reported as mean ± std over seeds.

    python analysis/aggregate.py                 # results used in the paper (results/paper/)
    python analysis/aggregate.py --check         # ... and check them against the values printed in the paper
    python analysis/aggregate.py --source runs   # your own runs (results/chronos2/, results/ttm/)

Tables are printed as Markdown and written as CSV to analysis/output/<source>/.
Only the Python standard library, numpy and pandas are needed.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "analysis"
STRATEGIES = ["zero-shot", "fft", "lora-fullshot", "lora-fewshot"]
LABELS = {"zero-shot": "Zero-Shot", "fft": "FFT", "lora-fullshot": "LoRA-FullShot", "lora-fewshot": "LoRA-FewShot"}
MODELS = {"chronos2": "Chronos-2", "ttm": "TTM-R3-PT"}
METRICS = ["MWQL_n", "MAPE_n"]
GROUPINGS = {"size": "total_obs", "n_series": "n_series", "mean_length": "mean_len"}
ORDER = {"size": ["small", "medium", "large"], "n_series": ["small", "medium", "large"],
         "mean_length": ["short", "medium", "long"]}

# TTM result files name datasets as "<key>/<freq>/<term>"; this maps them back to GIFT-Eval names.
TTM_PRETTY = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
              "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
TTM_VERSIONS = {r"TTM-R3-PT-Zero-Shot": "zero-shot", r"TTM-R3-FullFinetuning-FullShot-Seed(\d+)": "fft",
                r"TTM-R3-LORA-FullShot-Seed(\d+)": "lora-fullshot", r"TTM-R3-LORA-FewShot-20-Seed(\d+)": "lora-fewshot"}

# Values printed in the paper (model, metric, grouping, group, strategy, value), used by --check.
PAPER_VALUES = [
    ("chronos2", "MWQL_n", "size", "small", "zero-shot", 0.423),
    ("chronos2", "MWQL_n", "size", "medium", "lora-fullshot", 0.357),
    ("chronos2", "MWQL_n", "size", "large", "lora-fullshot", 0.301),
    ("chronos2", "MAPE_n", "size", "small", "zero-shot", 0.544),
    ("chronos2", "MAPE_n", "size", "large", "lora-fewshot", 0.931),
    ("ttm", "MWQL_n", "size", "small", "zero-shot", 0.494),
    ("ttm", "MWQL_n", "size", "large", "fft", 0.350),
    ("ttm", "MAPE_n", "size", "large", "zero-shot", 1.240),
    ("chronos2", "MWQL_n", "n_series", "small", "zero-shot", 0.372),
    ("chronos2", "MWQL_n", "n_series", "large", "lora-fullshot", 0.398),
    ("ttm", "MWQL_n", "n_series", "large", "fft", 0.458),
    ("ttm", "MAPE_n", "n_series", "medium", "zero-shot", 1.212),
    ("chronos2", "MWQL_n", "mean_length", "short", "zero-shot", 0.503),
    ("chronos2", "MWQL_n", "mean_length", "short", "fft", 0.623),
    ("ttm", "MWQL_n", "mean_length", "short", "zero-shot", 0.574),
    ("ttm", "MWQL_n", "mean_length", "long", "lora-fullshot", 0.322),
    ("chronos2", "MWQL_n", "domain_horizon", "Transport|medium", "fft", 0.138),
    ("chronos2", "MWQL_n", "domain_horizon", "Nature|short", "zero-shot", 0.367),
    ("chronos2", "MAPE_n", "domain_horizon", "Energy|long", "lora-fewshot", 1.364),
    ("ttm", "MWQL_n", "domain_horizon", "Healthcare|short", "lora-fullshot", 0.593),
    ("ttm", "MAPE_n", "domain_horizon", "Web/CloudOps|medium", "fft", 0.928),
]


# ------------------------------------------------------------------------------------------------ loading
def ttm_name_map(names):
    props = json.loads((ROOT / "ttm" / "resources" / "dataset_properties.json").read_text())
    out = {}
    for name in names:
        if "/" in name:
            key, freq = name.split("/")
            key = TTM_PRETTY.get(key.lower(), key.lower())
        else:
            key = TTM_PRETTY.get(name.lower(), name.lower())
            freq = props[key]["frequency"]
        out[f"{key}/{freq}"] = name
    return out


def read_result(path, naive_names):
    """Return a DataFrame [dataset, term, WQL, MAPE] for any result file produced by this repository."""
    df = pd.read_csv(path)
    if "eval_metrics/mean_weighted_sum_quantile_loss" in df.columns:  # raw TTM output
        names = ttm_name_map(naive_names)
        parts = df["dataset"].str.rsplit("/", n=1, expand=True)
        df = pd.DataFrame({"dataset": parts[0].map(names), "term": parts[1],
                           "WQL": df["eval_metrics/mean_weighted_sum_quantile_loss"],
                           "MAPE": df["eval_metrics/MAPE[0.5]"]})
        if df["dataset"].isna().any():
            raise ValueError(f"{path}: unknown dataset names")
    else:  # Chronos-2 / AutoGluon leaderboards and the paper files (AutoGluon stores errors with a minus sign)
        df = df[["dataset", "term", "WQL", "MAPE"]].copy()
    df[["WQL", "MAPE"]] = df[["WQL", "MAPE"]].abs()
    if df.duplicated(["dataset", "term"]).any():
        raise ValueError(f"{path}: duplicated configurations")
    return df


def find_files(source):
    """Yield (model, strategy, seed, path)."""
    if source == "paper":
        for path in sorted((ROOT / "results" / "paper").glob("*/*_seed*.csv")):
            strategy, seed = re.match(r"(.+)_seed(\d+)\.csv", path.name).groups()
            yield path.parent.name, strategy, int(seed), path
        return
    for path in sorted((ROOT / "results" / "chronos2").glob("*/seed*/leaderboard.csv")):
        yield "chronos2", path.parent.parent.name, int(path.parent.name[4:]), path
    for path in sorted((ROOT / "results" / "ttm").glob("*/all_results.csv")):
        for pattern, strategy in TTM_VERSIONS.items():
            m = re.fullmatch(pattern, path.parent.name)
            if m:
                yield "ttm", strategy, int(m.group(1)) if m.groups() else 42, path


def load(source):
    naive = pd.read_csv(HERE / "seasonal_naive.csv")[["dataset", "term", "domain", "WQL", "MAPE"]]
    props = pd.read_csv(HERE / "dataset_properties.csv")
    for g, col in GROUPINGS.items():  # paper thresholds: 0.33 / 0.66 quantiles over the 55 GIFT-Eval datasets
        q1, q2 = props[col].quantile(0.33), props[col].quantile(0.66)
        props[g] = np.where(props[col] <= q1, ORDER[g][0], np.where(props[col] <= q2, ORDER[g][1], ORDER[g][2]))
    rows = []
    for model, strategy, seed, path in find_files(source):
        r = read_result(path, naive["dataset"].unique()).merge(naive, on=["dataset", "term"], how="left",
                                                               suffixes=("", "_naive"), validate="one_to_one")
        missing = r["WQL_naive"].isna()
        if missing.any():
            print(f"warning: {path}: {missing.sum()} configurations without Seasonal Naive baseline are ignored")
            r = r[~missing]
        r["MWQL_n"] = r["WQL"] / r["WQL_naive"]
        r["MAPE_n"] = r["MAPE"] / r["MAPE_naive"]
        rows.append(r.assign(model=model, strategy=strategy, seed=seed))
    if not rows:
        raise SystemExit(f"No result files found for source '{source}'.")
    df = pd.concat(rows, ignore_index=True)
    df = df.merge(props[["dataset", *GROUPINGS]], on="dataset", how="left", validate="many_to_one")
    df["domain_horizon"] = df["domain"] + "|" + df["term"]
    df["all"] = "all"
    return df


# ------------------------------------------------------------------------------------------------ tables
def table(df, model, metric, grouping):
    """Mean over configurations (per seed), then mean ± std over seeds."""
    d = df[df.model == model]
    per_seed = d.groupby([grouping, "strategy", "seed"])[metric].mean().reset_index()
    agg = per_seed.groupby([grouping, "strategy"])[metric].agg(["mean", "std", "count"]).reset_index()
    n_conf = d[d.strategy == "zero-shot"].groupby(grouping).size() if "zero-shot" in set(d.strategy) else None
    out = agg.pivot(index=grouping, columns="strategy", values="mean")
    std = agg.pivot(index=grouping, columns="strategy", values="std")
    order = ORDER.get(grouping) or sorted(out.index)
    out, std = out.reindex([g for g in order if g in out.index]), std.reindex([g for g in order if g in out.index])
    cols = [s for s in STRATEGIES if s in out.columns]
    return out[cols], std[cols], n_conf


def markdown(mean, std, n_conf, grouping):
    head = f"| {grouping} | n | " + " | ".join(LABELS[c] for c in mean.columns) + " |"
    lines = [head, "|" + "---|" * (len(mean.columns) + 2)]
    for g, row in mean.iterrows():
        best = row.min()
        cells = []
        for c, v in row.items():
            s = "–" if pd.isna(v) else f"{v:.3f}" + ("" if pd.isna(std.loc[g, c]) else f" ± {std.loc[g, c]:.3f}")
            cells.append(f"**{s}**" if v == best else s)
        n = "" if n_conf is None else int(n_conf.get(g, 0))
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["paper", "runs"], default="paper")
    p.add_argument("--check", action="store_true", help="compare with the values printed in the paper")
    args = p.parse_args()

    df = load(args.source)
    out_dir = HERE / "output" / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_configuration.csv", index=False)

    found = df.groupby(["model", "strategy"])["seed"].agg(lambda s: sorted(set(s))).reset_index()
    print("Result files:")
    for _, r in found.iterrows():
        n = df[(df.model == r.model) & (df.strategy == r.strategy)].groupby("seed").size().max()
        print(f"  {MODELS[r.model]:10s} {LABELS[r.strategy]:14s} seeds={r.seed}  configurations={n}")

    for model in [m for m in MODELS if m in set(df.model)]:
        for grouping in ["all", *GROUPINGS, "domain_horizon"]:
            for metric in METRICS:
                mean, std, n_conf = table(df, model, metric, grouping)
                print(f"\n### {MODELS[model]} — {metric} by {grouping}\n")
                print(markdown(mean, std, n_conf, grouping))
                res = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)
                res.to_csv(out_dir / f"{model}_{grouping}_{metric}.csv")

    if args.check:
        print("\nCheck against the paper:")
        bad = 0
        for model, metric, grouping, group, strategy, ref in PAPER_VALUES:
            got = df[(df.model == model) & (df.strategy == strategy) & (df[grouping] == group)] \
                .groupby("seed")[metric].mean().mean()
            ok = abs(got - ref) <= 0.0006  # paper values are rounded to 3 decimals
            bad += not ok
            print(f"  {'OK ' if ok else 'BAD'} {MODELS[model]:10s} {metric} {grouping}={group:20s} "
                  f"{LABELS[strategy]:14s} paper={ref:.3f} recomputed={got:.4f}")
        print(f"{len(PAPER_VALUES) - bad}/{len(PAPER_VALUES)} values reproduced.")
        if bad:
            raise SystemExit(1)
    print(f"\nCSV tables written to {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
