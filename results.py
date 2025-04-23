"""
results.py

Utility to aggregate the *final‑run* Highway‑PPO artifacts, compute
presentation‑ready metrics, and spit out a handful of publication‑quality
figures (PNG).

Run examples
------------
python results.py --art-dir ~/Downloads/4-20-highway-ppo-artifacts/artifacts/highway-ppo \
                 --out-dir  figures \
                 --thr 120
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------- #
# 1. Helpers – parsing & loading                                              #
# --------------------------------------------------------------------------- #

EXP_RX = re.compile(
    r"^(?P<prefix>sorted|shuffled)"
    r"(?:_(?P<pe_type>rankpe|distpe|rope))?"  # optional PE subtype
    r"_lr(?P<lr>[0-9.eE+-]+)"
    r"_hidden_dim(?P<hidden_dim>\d+)"
    r"_clip_eps(?P<clip_eps>[0-9.eE+-]+)"
    r"_entropy_coef(?P<entropy_coef>[0-9.eE+-]+)"
    r"_epochs(?P<epochs>\d+)"
    r"_batch_size(?P<batch_size>\d+)"
    r"_d_embed(?P<d_embed>\d+)"
    r"_seed(?P<seed>\d+)$"
)


def _parse_exp_name(name: str) -> dict:
    m = EXP_RX.match(name)
    if not m:
        raise ValueError(f"Cannot parse experiment name: {name}")
    d = m.groupdict()
    # normalise types
    d["hidden_dim"] = int(d["hidden_dim"])
    d["batch_size"] = int(d["batch_size"])
    d["d_embed"] = int(d["d_embed"])
    d["epochs"] = int(d["epochs"])
    d["seed"] = int(d["seed"])
    d["lr"] = float(d["lr"])
    d["clip_eps"] = float(d["clip_eps"])
    d["entropy_coef"] = float(d["entropy_coef"])
    return d


def load_json_metrics(art_dir: Path) -> pd.DataFrame:
    """Read every training_metrics_*.json in art_dir → long‑form DF."""
    rows = []
    for path in art_dir.glob("training_metrics_*.json"):
        with open(path) as f:
            data = json.load(f)
        meta = _parse_exp_name(data["experiment_name"])
        ep_nums = data["eval_episode_numbers"]
        evals = data["avg_eval_rewards"]
        for ep, r in zip(ep_nums, evals):
            rows.append(
                {
                    "experiment": data["experiment_name"],
                    "episode": ep,
                    "avg_eval": r,
                    **meta,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Metric aggregations                                                      #
# --------------------------------------------------------------------------- #


def episodes_to_threshold(df_long: pd.DataFrame, thr: float) -> pd.DataFrame:
    """First eval‑episode where avg_eval ≥ thr."""
    hits = (
        df_long[df_long["avg_eval"] >= thr]
        .groupby("experiment")["episode"]
        .min()
        .reset_index(name="ep_to_thr")
    )
    return hits


def auc(df_long: pd.DataFrame) -> pd.DataFrame:
    """Trapezoidal area under the eval curve, normalised by last episode."""
    aucs = []
    for exp, sub in df_long.groupby("experiment"):
        x = sub["episode"].to_numpy()
        y = sub["avg_eval"].to_numpy()
        # Ensure sorted
        order = np.argsort(x)
        x, y = x[order], y[order]
        area = np.trapz(y, x)
        norm_area = area / x[-1]  # units: reward‑per‑episode
        aucs.append({"experiment": exp, "auc": norm_area})
    return pd.DataFrame(aucs)


# --------------------------------------------------------------------------- #
# 3. Plotting                                                                 #
# --------------------------------------------------------------------------- #


def _boxplot(ax, data, xlab, ylab, title):
    bp = ax.boxplot(
        [group[1] for group in data.groupby(xlab)[ylab]],
        labels=data.groupby(xlab).groups.keys(),
        showmeans=True,
    )
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return bp


def make_plots(df_meta: pd.DataFrame, df_long: pd.DataFrame, out_dir: Path, thr: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Final reward box‑plot by condition ------------------------------ #
    cond_map = (
        df_meta[["experiment", "prefix", "pe_type"]]
        .drop_duplicates()
        .assign(
            condition=lambda d: d.apply(
                lambda r: r["prefix"]
                if pd.isna(r["pe_type"])
                else f"{r['prefix']}_{r['pe_type']}",
                axis=1,
            )
        )
    )
    df_fin = (
        df_long.sort_values("episode")
        .groupby("experiment")
        .tail(1)
        .merge(cond_map[["experiment", "condition"]], on="experiment")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    _boxplot(
        ax,
        df_fin[["condition", "avg_eval"]],
        "condition",
        "avg_eval",
        "Final eval reward by condition",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "box_final_reward.png", dpi=200)
    plt.close(fig)

    # --- 2. Episodes‑to‑threshold box‑plot --------------------------------- #
    df_thr = episodes_to_threshold(df_long, thr).merge(
        cond_map[["experiment", "condition"]], on="experiment"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    _boxplot(
        ax,
        df_thr,
        "condition",
        "ep_to_thr",
        f"Episodes to reach {thr} reward",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "box_ep_to_thr.png", dpi=200)
    plt.close(fig)

    # --- 3. Heat‑map: mean final reward vs hidden_dim × pe_type ------------ #
    pivot = (
        df_fin.assign(pe=lambda d: d["condition"].str.replace("shuffled_", ""))
        .pivot_table(
            index="hidden_dim", columns="pe", values="avg_eval", aggfunc="mean"
        )
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("Positional‑Encoding Type")
    ax.set_ylabel("Hidden Dim")
    ax.set_title("Mean final reward")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("reward")
    fig.tight_layout()
    fig.savefig(out_dir / "heat_hidden_dim_vs_pe.png", dpi=200)
    plt.close(fig)

    # --- 4. Δ‑recovery bar chart ------------------------------------------- #
    keycols = ["hidden_dim", "batch_size", "d_embed", "seed"]
    baseline = (
        df_fin[df_fin["condition"] == "shuffled"]
        .set_index(keycols)["avg_eval"]
        .rename("shuffled_base")
    )
    # Ordering penalty reference
    sorted_base = (
        df_fin[df_fin["condition"] == "sorted"]
        .set_index(keycols)["avg_eval"]
        .rename("sorted_ref")
    )

    bars = {}
    for pe in ["shuffled_rankpe", "shuffled_rope", "shuffled_distpe"]:
        diff = (
            df_fin[df_fin["condition"] == pe]
            .set_index(keycols)["avg_eval"]
            .sub(baseline, fill_value=np.nan)
        )
        bars[pe.replace("shuffled_", "")] = diff.dropna()

    # Reference delta (sorted – shuffled_noPE)
    ordering_penalty = (sorted_base - baseline).dropna()
    ord_pen = ordering_penalty.mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    labels, means, stds = [], [], []
    for lab, series in bars.items():
        labels.append(lab)
        means.append(series.mean())
        stds.append(series.std())
    ax.bar(labels, means, yerr=stds, capsize=4, alpha=0.8)
    ax.axhline(ord_pen, ls="--", color="k", label="sorted − shuffled")
    ax.text(
        0.02,
        ord_pen + 0.3,
        f"{ord_pen:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.set_ylabel("Δ reward vs. shuffled baseline")
    ax.set_title("How much ordering penalty each PE recovers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delta_recovery.png", dpi=200)
    plt.close(fig)

    # --- 5. AULC box‑plot --------------------------------------------------- #
    df_auc = auc(df_long).merge(cond_map[["experiment", "condition"]], on="experiment")
    fig, ax = plt.subplots(figsize=(8, 6))
    _boxplot(
        ax,
        df_auc,
        "condition",
        "auc",
        "Area‑Under‑Learning‑Curve (normalized)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "box_auc.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------- #
    # 6. Console summary                                                  #
    # ------------------------------------------------------------------- #
    print("\n=== QUICK SUMMARY ===")
    # Final reward medians
    med_final = df_fin.groupby("condition")["avg_eval"].median().round(1).sort_values()
    print("\nMedian final reward:")
    for cond, val in med_final.items():
        print(f"  {cond:15}  {val:5.1f}")

    # AULC medians
    med_auc = df_auc.groupby("condition")["auc"].median().round(1).sort_values()
    print("\nMedian AULC (normalised):")
    for cond, val in med_auc.items():
        print(f"  {cond:15}  {val:5.1f}")

    # Episodes‑to‑threshold medians
    med_thr = (
        df_thr.groupby("condition")["ep_to_thr"].median().astype(int).sort_values()
    )
    print(f"\nMedian episodes to reach {thr} reward:")
    for cond, val in med_thr.items():
        print(f"  {cond:15}  {val:5d}")

    # Ordering penalty
    print(f"\nOrdering penalty (sorted – shuffled_noPE) ≈ {ord_pen:.2f} reward points.")

    # Δ‑recovery means
    print("\nΔ‑recovery means (positive would be good):")
    for lab, series in bars.items():
        print(f"  {lab:8} {series.mean():6.2f} ± {series.std():4.2f}")


# --------------------------------------------------------------------------- #
# 4. CLI -------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate Highway‑PPO final‑run artifacts and produce figures."
    )
    p.add_argument("--art-dir", type=Path, required=True, help="Directory with *.json")
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    p.add_argument(
        "--thr",
        type=float,
        default=120.0,
        help="Reward threshold for sample‑efficiency metric",
    )
    args = p.parse_args()

    print(f"Loading metrics from {args.art_dir} …")
    df_long = load_json_metrics(args.art_dir)
    print(
        f"Loaded {df_long['experiment'].nunique()} experiments, "
        f"{len(df_long)} eval checkpoints."
    )

    # Derive meta once
    meta_cols = [
        "experiment",
        "prefix",
        "pe_type",
        "hidden_dim",
        "batch_size",
        "d_embed",
        "seed",
    ]
    df_meta = df_long[meta_cols].drop_duplicates()

    print("Computing & saving plots …")
    make_plots(df_meta, df_long, args.out_dir, thr=args.thr)

    print("Done. Figures written to", args.out_dir.resolve())


if __name__ == "__main__":
    main()
