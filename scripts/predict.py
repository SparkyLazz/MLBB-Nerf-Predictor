"""Score the latest daily snapshot for nerf-candidate likelihood.

The formula below is a first-pass heuristic with UNVALIDATED weights (0.45 / 0.55) -- that's
deliberate, not an oversight. Do not change, "fix", or extend it (no new features, no reweighting,
no re-deriving from train.csv) without discussing first.

wr_z    = z-score of win_rate
ban_z   = z-score of log1p(ban_rate)
conf    = log1p(pick_rate) / max(log1p(pick_rate))
skor    = (0.45*wr_z + 0.55*ban_z) * conf

Reads directly from data/daily/ (the latest snapshot by filename date), not data/train.csv --
this scores "right now", independent of which patch window that day ends up assigned to later.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from promote import PATCHES, load_patches, patch_window_index  # noqa: E402

DAILY_DIR = "data/daily"
PREDICTIONS_DIR = "predictions"
TOP_N = 10


def _latest_daily_snapshot() -> str:
    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.csv")))
    if not files:
        raise RuntimeError(f"No files found in {DAILY_DIR}.")
    return files[-1]


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["win_rate"] = df["win_rate"].astype(float)
    df["pick_rate"] = df["pick_rate"].astype(float)
    df["ban_rate"] = df["ban_rate"].astype(float)

    wr_z = _zscore(df["win_rate"])
    ban_z = _zscore(np.log1p(df["ban_rate"]))
    log_pick = np.log1p(df["pick_rate"])
    conf = log_pick / log_pick.max()

    df["skor"] = (0.45 * wr_z + 0.55 * ban_z) * conf
    return df


def _patch_id_for(date_str: str) -> str:
    """Best-effort label for the output filename, derived from the human-curated patch calendar
    (data/patches.csv) -- not guessed from the web. Falls back to "unlabeled" if patches.csv is
    missing or the date predates every recorded patch.
    """
    if not os.path.exists(PATCHES):
        return "unlabeled"
    patches = load_patches(PATCHES)
    idx = patch_window_index(pd.Timestamp(date_str), patches)
    return patches.loc[idx, "patch_id"] if idx is not None else "unlabeled"


def main() -> None:
    path = _latest_daily_snapshot()
    date_str = os.path.splitext(os.path.basename(path))[0]

    df = pd.read_csv(path)
    scored = score(df).sort_values("skor", ascending=False)
    top = scored.head(TOP_N)[["hero", "role", "win_rate", "pick_rate", "ban_rate", "skor"]]

    print(top.to_string(index=False))

    patch_id = _patch_id_for(date_str)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_path = os.path.join(PREDICTIONS_DIR, f"{date_str}_{patch_id}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Nerf candidates -- snapshot {date_str} (patch {patch_id})\n\n")
        f.write(f"Source: `{path}`\n\n")
        f.write("Score = (0.45*wr_z + 0.55*ban_z) * confidence; unvalidated heuristic weights.\n\n")
        f.write("| Rank | Hero | Role | Win% | Pick% | Ban% | Score |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            f.write(
                f"| {rank} | {row['hero']} | {row['role']} | {row['win_rate']:.2f} | "
                f"{row['pick_rate']:.2f} | {row['ban_rate']:.2f} | {row['skor']:.3f} |\n"
            )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
