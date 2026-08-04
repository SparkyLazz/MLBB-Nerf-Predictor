"""Build data/train.csv from data/daily/*.csv + data/patches.csv.

train.csv is a pure derived artifact: delete it, rerun this script from repo root, get it back --
same inputs produce the same output every time. It is never a source of truth; data/daily/ and
data/patches.csv are.

Join logic:
- patches.csv is the hand-curated patch calendar. Each patch's "active window" runs from its own
  release_date up to (but not including) the next patch's release_date, sorted by release_date.
  The most recent patch's window is open-ended -- it covers every daily snapshot from its
  release_date up to whatever's newest in data/daily/ right now.
- For each patch window, this script keeps only the SINGLE latest daily snapshot date that falls
  inside it -- not every daily snapshot taken during that patch's life. That matches the original
  spec: one row per hero per patch, taken near the end of the patch (not the first day, and not
  once per day). Extra daily snapshots inside the same window just let a later day win.
- nerfed_next for a patch's rows = 1 if that hero appears in the *next* patch's heroes_nerfed
  list, 0 otherwise. The most recent patch (no next patch recorded yet) gets nerfed_next left
  blank -- that's the row this whole pipeline exists to predict.
- A daily snapshot dated before the earliest patch in patches.csv can't be assigned to any window
  and is skipped with a warning, not guessed into the nearest patch.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pandas as pd

DAILY_DIR = "data/daily"
PATCHES = "data/patches.csv"
OUT = "data/train.csv"

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")

TRAIN_COLUMNS = ["patch_id", "patch_date", "hero", "role", "win_rate", "pick_rate", "ban_rate", "rank_bracket", "nerfed_next"]


def _split_heroes(cell) -> set[str]:
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {h.strip() for h in str(cell).split("|") if h.strip()}


def load_patches(patches_csv: str = PATCHES) -> pd.DataFrame:
    """Load and sort the patch calendar, with release_date parsed and nerf lists pre-split."""
    patches = pd.read_csv(patches_csv, dtype=str).fillna("")
    if patches.empty:
        raise RuntimeError(f"{patches_csv} is empty -- nothing to join against.")
    patches["release_date"] = pd.to_datetime(patches["release_date"])
    patches = patches.sort_values("release_date").reset_index(drop=True)
    patches["nerf_set"] = patches["heroes_nerfed"].apply(_split_heroes)
    return patches


def patch_window_index(date: pd.Timestamp, patches: pd.DataFrame) -> int | None:
    """Which row of `patches` was active on `date`? None if date predates the earliest patch."""
    n = len(patches)
    for i in range(n):
        window_start = patches.loc[i, "release_date"]
        window_end = patches.loc[i + 1, "release_date"] if i + 1 < n else None
        if date >= window_start and (window_end is None or date < window_end):
            return i
    return None


def _daily_files(daily_dir: str = DAILY_DIR) -> list[tuple[str, str]]:
    """[(date_str, path), ...] for every file in data/daily/, sorted by date."""
    files = []
    for path in sorted(glob.glob(os.path.join(daily_dir, "*.csv"))):
        m = _DATE_RE.search(os.path.basename(path))
        if not m:
            print(f"WARNING: skipping {path}, filename isn't YYYY-MM-DD.csv", file=sys.stderr)
            continue
        files.append((m.group(1), path))
    files.sort()
    return files


def build_train(daily_dir: str = DAILY_DIR, patches_csv: str = PATCHES) -> pd.DataFrame:
    patches = load_patches(patches_csv)
    daily_files = _daily_files(daily_dir)
    if not daily_files:
        raise RuntimeError(f"No files found in {daily_dir} -- nothing to build train.csv from.")

    n_patches = len(patches)
    latest_date_for_patch: dict[int, str] = {}
    latest_path_for_patch: dict[int, str] = {}
    for date_str, path in daily_files:
        idx = patch_window_index(pd.Timestamp(date_str), patches)
        if idx is None:
            print(
                f"WARNING: daily snapshot {date_str} predates the earliest patch in {patches_csv}, skipping.",
                file=sys.stderr,
            )
            continue
        if idx not in latest_date_for_patch or date_str > latest_date_for_patch[idx]:
            latest_date_for_patch[idx] = date_str
            latest_path_for_patch[idx] = path

    if not latest_date_for_patch:
        raise RuntimeError(
            f"No daily snapshot in {daily_dir} falls inside any patch window in {patches_csv} -- "
            "check release_date values."
        )

    rows = []
    for idx, date_str in sorted(latest_date_for_patch.items()):
        snap = pd.read_csv(latest_path_for_patch[idx], dtype=str)
        snap["win_rate"] = snap["win_rate"].astype(float)
        snap["pick_rate"] = snap["pick_rate"].astype(float)
        snap["ban_rate"] = snap["ban_rate"].astype(float)

        patch_id = patches.loc[idx, "patch_id"]
        patch_date = patches.loc[idx, "release_date"].date().isoformat()
        next_nerf_set = patches.loc[idx + 1, "nerf_set"] if idx + 1 < n_patches else None

        snap["patch_id"] = patch_id
        snap["patch_date"] = patch_date
        snap["nerfed_next"] = "" if next_nerf_set is None else snap["hero"].apply(lambda h: 1 if h in next_nerf_set else 0)

        rows.append(snap[TRAIN_COLUMNS])

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    try:
        train = build_train()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    train.to_csv(OUT, index=False)
    print(f"Wrote {len(train)} rows ({train['patch_id'].nunique()} patch(es)) -> {OUT}")


if __name__ == "__main__":
    main()
