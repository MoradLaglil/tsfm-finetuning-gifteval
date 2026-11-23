"""Zero-shot evaluation and fine-tuning of Chronos-2 on GIFT-Eval (AutoGluon backend).

One run = one strategy x one seed over all GIFT-Eval configurations (dataset, frequency, horizon).

    python chronos2/run_chronos2.py --strategy lora-fullshot --seed 42

Strategies (Section 4 of the paper):
    zero-shot      pre-trained model, no parameter update
    fft            full fine-tuning on the whole training split
    lora-fullshot  LoRA adapters, whole training split
    lora-fewshot   LoRA adapters, last 20% of the time steps of each series

Results are appended to <output-dir>/<strategy>/seed<seed>/leaderboard.csv, one row per configuration.
Finished configurations are skipped, so an interrupted run can simply be restarted.
"""

import argparse
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from dotenv import load_dotenv
from gift_eval.data import Dataset

load_dotenv()  # reads GIFT_EVAL=<data dir> from the .env file at the repository root (or set it in the shell)

# ---------------------------------------------------------------------------------------------
# Configurations: 52 dataset/frequency entries; the 21 high-frequency ones are also evaluated at
# medium and long horizons -> 94 configurations.
# ---------------------------------------------------------------------------------------------
SHORT_DATASETS = (
    "m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D "
    "electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M "
    "us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D "
    "ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T "
    "bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
    "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T "
    "jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()
TERMS = ["short", "medium", "long"]

# ---------------------------------------------------------------------------------------------
# Fine-tuning hyperparameters (identical for every dataset; no per-dataset tuning).
# AutoGluon defaults apply to everything not listed here (LoRA r=8, alpha=16; batch size 32;
# no validation during fine-tuning, so the final weights are used).
# ---------------------------------------------------------------------------------------------
FEW_SHOT_FRACTION = 0.2
STRATEGIES = {
    "zero-shot": dict(fine_tune=False),
    "fft": dict(fine_tune=True, fine_tune_mode="full", fine_tune_lr=1e-5),
    "lora-fullshot": dict(fine_tune=True, fine_tune_mode="lora", fine_tune_lr=1e-5),
    "lora-fewshot": dict(fine_tune=True, fine_tune_mode="lora", fine_tune_lr=1e-5, fine_tune_batch_size=64),
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def to_full_windows(test_data):
    """GluonTS test instances -> one series per test window (context followed by its horizon)."""
    out = []
    for context, forecast in test_data:
        out.append({
            "target": np.concatenate([context["target"], forecast["target"]]),
            "start": pd.Period(context["start"], freq=forecast["freq"]),
        })
    return out


def load_dataset(ds_name: str, term: str):
    """Load a configuration with the official GIFT-Eval splits.

    Multivariate datasets are split into univariate series (one per variate).
    `history` holds everything before the test windows (training + validation part of GIFT-Eval);
    it is what the model is fine-tuned on. AutoGluon keeps the last `prediction_length` steps of
    each series aside to report a validation score.
    """
    multivariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim > 1
    ds = Dataset(name=ds_name, term=term, to_univariate=multivariate)
    history = TimeSeriesDataFrame.from_iterable_dataset(ds.validation_dataset)
    test = TimeSeriesDataFrame.from_iterable_dataset(to_full_windows(ds.test_data))
    return history, test, ds.prediction_length, multivariate, ds.freq


def truncate_to_few_shot(data: TimeSeriesDataFrame, fraction: float, min_length: int) -> TimeSeriesDataFrame:
    """Keep only the last `fraction` of time steps of each series, with at least `min_length` steps."""
    frames = []
    for _, group in data.groupby(level="item_id"):
        n_keep = max(min_length, int(len(group) * fraction))
        frames.append(group.iloc[-n_keep:])
    return TimeSeriesDataFrame(pd.concat(frames))


def configurations(datasets=None, terms=None):
    names = sorted(set(SHORT_DATASETS + MED_LONG_DATASETS))
    if datasets:
        unknown = set(datasets) - set(names)
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        names = [n for n in names if n in datasets]
    for name in names:
        for term in terms or TERMS:
            if term == "short" or name in MED_LONG_DATASETS:
                yield name, term


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=2000, help="gradient steps per configuration (fine-tuning only)")
    p.add_argument("--datasets", nargs="+", default=None, help="subset of dataset names, e.g. ett1/D hospital")
    p.add_argument("--terms", nargs="+", default=None, choices=TERMS, help="subset of horizons")
    p.add_argument("--output-dir", default="results/chronos2")
    args = p.parse_args()

    set_all_seeds(args.seed)
    run_dir = Path(args.output_dir) / args.strategy / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "leaderboard.csv"
    ag_path = run_dir / "autogluon"  # overwritten for every configuration

    done = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        done = set(zip(prev["dataset"], prev["term"]))

    hp = dict(STRATEGIES[args.strategy])
    if hp["fine_tune"]:
        hp["fine_tune_steps"] = args.steps
    suffix = "FineTuned" if hp["fine_tune"] else "ZeroShot"
    hp["ag_args"] = {"name_suffix": suffix}

    todo = [c for c in configurations(args.datasets, args.terms) if c not in done]
    print(f"Chronos-2 | {args.strategy} | seed {args.seed} | {len(todo)} configurations to run, {len(done)} already done")

    for i, (ds_name, term) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {ds_name} / {term}", flush=True)
        t0 = time.time()
        history, test, h, multivariate, freq = load_dataset(ds_name, term)
        train = history
        if args.strategy == "lora-fewshot":
            train = truncate_to_few_shot(history, FEW_SHOT_FRACTION, min_length=2 * h + 1)

        if ag_path.exists():
            shutil.rmtree(ag_path)
        predictor = TimeSeriesPredictor(path=str(ag_path), prediction_length=h, verbosity=2).fit(
            train_data=train,
            enable_ensemble=False,
            random_seed=args.seed,
            hyperparameters={"Chronos2": [hp]},
        )
        if not predictor.model_names():
            raise RuntimeError(f"Chronos-2 failed to train on {ds_name}/{term}; see the AutoGluon log above.")
        lb = predictor.leaderboard(test, extra_metrics=["WQL", "MAPE"])
        lb = lb.assign(dataset=ds_name, term=term, seed=args.seed, strategy=args.strategy,
                       steps=args.steps if hp["fine_tune"] else 0, **{"Multi-variate": multivariate}, freq=freq,
                       runtime_s=round(time.time() - t0, 1))
        cols = ["model", "dataset", "term", "seed", "strategy", "steps", "WQL", "MAPE", "Multi-variate", "freq", "runtime_s"]
        lb[cols].to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())
        print(f"    WQL={abs(lb['WQL'].iloc[0]):.4f}  MAPE={abs(lb['MAPE'].iloc[0]):.4f}  "
              f"({time.time() - t0:.0f}s) -> {csv_path}", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
