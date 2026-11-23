"""Draw the README figure: change in MWQL_n of each fine-tuning strategy relative to zero-shot, by dataset size.

    python analysis/make_figure.py            # from results/paper/, writes assets/finetuning_vs_zeroshot_{light,dark}.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import LABELS, MODELS, ROOT, load  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df", zero="#8a8984",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#33332f", zero="#8a8984",
                 series=["#3987e5", "#d95926", "#199e70"]),
}
FT = ["fft", "lora-fullshot", "lora-fewshot"]
GROUPS = [("small", "Small datasets"), ("medium", "Medium datasets"), ("large", "Large datasets"), ("all", "All")]


def relative_change(df, model):
    """% change of the mean MWQL_n (over configurations, then seeds) relative to zero-shot."""
    d = df[df.model == model].copy()
    out = {}
    for key, _ in GROUPS:
        sub = d if key == "all" else d[d["size"] == key]
        mean = sub.groupby(["strategy", "seed"])["MWQL_n"].mean().groupby("strategy").mean()
        out[key] = {s: 100 * (mean[s] - mean["zero-shot"]) / mean["zero-shot"] for s in FT}
    return out


def draw(df, theme):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": t["text"],
                         "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["text"]})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharex=True, facecolor=t["surface"])
    bar_h, gap = 0.22, 0.03
    for ax, model in zip(axes, MODELS):
        rc = relative_change(df, model)
        ax.set_facecolor(t["surface"])
        for gi, (key, _) in enumerate(GROUPS):
            for si, s in enumerate(FT):
                y = gi + (si - 1) * (bar_h + gap)
                ax.barh(y, rc[key][s], height=bar_h, color=t["series"][si], linewidth=0, zorder=3,
                        label=LABELS[s] if gi == 0 else None)
        ax.axvline(0, color=t["zero"], linewidth=1.2, zorder=4)
        ax.set_yticks(range(len(GROUPS)), [label for _, label in GROUPS])
        ax.invert_yaxis()
        ax.axhline(2.5, color=t["grid"], linewidth=1, zorder=1)  # separates "All" from the size groups
        ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(MODELS[model], loc="left", fontsize=12.5, fontweight="bold", color=t["text"], pad=10)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0")
    axes[0].set_xlim(-12, 40)
    for ax in axes:
        ax.set_xlabel("MWQL$_n$ change vs. zero-shot   (← better · worse →)")
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 1.0),
               handlelength=1.2, handleheight=0.9, labelcolor=t["text"])
    fig.text(0.01, 0.975, "Fine-tuning vs. zero-shot on GIFT-Eval", fontsize=13.5, fontweight="bold",
             color=t["text"], va="top")
    fig.text(0.01, 0.905, "Mean normalized WQL over configurations, by dataset size (lower is better)",
             fontsize=10, color=t["muted"], va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=2.5)
    out = ROOT / "assets" / f"finetuning_vs_zeroshot_{theme}.png"
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    data = load("paper")
    for theme in THEMES:
        draw(data, theme)
