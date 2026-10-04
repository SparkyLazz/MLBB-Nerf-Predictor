"""Build the model training set: every snapshot x hero, with trend features and a label.

How this differs from scripts/promote.py -- and why both exist:

``promote.py`` builds ``data/train.csv``, which is deliberately *one row per hero per patch*:
the single latest snapshot inside each patch window. That is the documented contract of that
file and this script does not touch it.

``dataset.py`` builds ``data/features.csv``, which keeps *every* snapshot. Three weeks of daily
Mythic collection inside one patch window is 21 rows per hero here versus 1 there -- and with
all five rank brackets collected it is 105. That is the whole point: the archive is the asset,
and throwing 95% of it away before training is the single biggest limit on how good the model
can get.

Labelling, identical in spirit to promote.py so the two never disagree:
  a snapshot taken inside patch N's window is labelled 1 if that hero appears in patch N+1's
  ``heroes_nerfed`` list, else 0. Snapshots inside the newest patch window have no next patch
  recorded yet, so they are *unlabelled* -- those are the rows to predict, not to train on.

Rows are weighted so each patch window counts the same regardless of how many days were
collected during it (see ``window_weights``); otherwise a long patch would dominate training
purely for lasting longer.

``data/features.csv`` is pure derived output and gitignored. Delete it and rerun any time.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as featmod  # noqa: E402
import store  # noqa: E402
from promote import PATCHES, load_patches  # noqa: E402

OUT = "data/features.csv"
LABEL = "nerfed_next"


def label_snapshots(feat: pd.DataFrame, patches: pd.DataFrame) -> pd.DataFrame:
    """Attach ``nerfed_next`` (and which patch supplied it) to every feature row."""
    cal = patches.sort_values("release_date").reset_index(drop=True)

    next_id: dict[str, str] = {}
    next_nerfs: dict[str, set[str]] = {}
    for i in range(len(cal) - 1):
        next_id[cal.loc[i, "patch_id"]] = cal.loc[i + 1, "patch_id"]
        next_nerfs[cal.loc[i, "patch_id"]] = cal.loc[i + 1, "nerf_set"]

    feat = feat.copy()
    feat["next_patch_id"] = feat["patch_id"].map(next_id).fillna("")
    feat[LABEL] = [
        1 if (pid in next_nerfs and hero in next_nerfs[pid]) else (0 if pid in next_nerfs else np.nan)
        for pid, hero in zip(feat["patch_id"], feat["hero"])
    ]
    return feat


def window_weights(labeled: pd.DataFrame) -> pd.Series:
    """One unit of influence per patch window, split across however many rows it contributed."""
    counts = labeled.groupby("patch_id")["hero"].transform("size")
    return (1.0 / counts).astype(float)


def build(
    ranks: tuple[str, ...] | list[str] | None = None,
    patches_csv: str = PATCHES,
    use_cache: bool = True,
) -> pd.DataFrame:
    long_df = store.load_snapshots(ranks=ranks, use_cache=use_cache)
    if long_df.empty:
        raise RuntimeError(
            "No snapshots found in data/daily/ or data/archive/ -- nothing to build a dataset from."
        )
    patches = load_patches(patches_csv)
    feat = featmod.build_features(long_df, patches)
    return label_snapshots(feat, patches)


def labeled_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows the model can learn from, rows waiting on the next patch to be recorded)."""
    labeled = dataset[dataset[LABEL].notna()].copy()
    unlabeled = dataset[dataset[LABEL].isna()].copy()
    return labeled, unlabeled


def describe(dataset: pd.DataFrame) -> dict:
    labeled, unlabeled = labeled_split(dataset)
    return {
        "rows": len(dataset),
        "labeled_rows": len(labeled),
        "unlabeled_rows": len(unlabeled),
        "labeled_patches": sorted(labeled["patch_id"].unique().tolist()),
        "positives": int(labeled[LABEL].sum()) if len(labeled) else 0,
        "features": len(featmod.feature_columns(dataset)),
        "ranks": sorted(dataset["rank_bracket"].unique().tolist()),
        "snapshots": int(dataset["ts"].nunique()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default="all", help='Comma-separated rank list, or "all" (default).')
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-cache", action="store_true", help="Ignore data/.cache and re-read every CSV.")
    args = ap.parse_args()

    ranks = None if args.ranks == "all" else tuple(r.strip() for r in args.ranks.split(",") if r.strip())

    try:
        dataset = build(ranks=ranks, use_cache=not args.no_cache)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dataset.to_csv(args.out, index=False)

    info = describe(dataset)
    print(f"Wrote {info['rows']} rows x {info['features']} features -> {args.out}")
    print(f"  snapshots : {info['snapshots']} across ranks {info['ranks']}")
    print(f"  labeled   : {info['labeled_rows']} rows, {info['positives']} nerfed, "
          f"patch windows {info['labeled_patches'] or '(none)'}")
    print(f"  unlabeled : {info['unlabeled_rows']} rows (newest patch window -- these are what gets predicted)")
    if not info["labeled_patches"]:
        print(
            "\n  NOTE: nothing is labeled yet. A snapshot only becomes training data once the\n"
            "  patch AFTER it is recorded in data/patches.csv with its nerf list. Record the\n"
            "  newest patch (menu option 4, or scripts/scraper.py labeled) and rerun.",
        )


if __name__ == "__main__":
    main()
