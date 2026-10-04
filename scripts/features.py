"""Turn the raw snapshot archive into trend-aware feature rows.

This is the step the original pipeline never had. ``predict.py`` scores a single day in
isolation: it can see that a hero is strong *today*, but not that it has been climbing for a
week, that its ban rate just doubled, or that it is strong in Mythic but not in Mythical Glory.
Those are exactly the signals that precede a nerf, and they only exist once you read the whole
archive at once.

``build_features(long_df)`` takes the long frame from scripts/store.py and returns one row per
(snapshot, rank, hero) with:

  level      win/pick/ban as scraped, plus contest_rate (pick+ban) and log transforms
  cohort     z-scores inside the same (snapshot, rank), and inside the same role, plus the
             confidence weight the original heuristic uses
  trend      1/3/7/14-day deltas for win, pick, ban and contest -- looked up per hero with
             merge_asof, so irregular collection gaps don't silently become zeros
  stability  7/14-day rolling mean and standard deviation, 14-day peak, how long the hero has
             been above the "nerf-worthy" thresholds, how much history exists at all
  spread     same-snapshot gap between this rank and the hero's average across every tracked
             rank (NaN while only one rank is being collected)
  patch      days since the active patch released, from data/patches.csv

Missing history is NaN, never 0 -- a hero with four days of data must not look like a hero
whose win rate held perfectly flat for two weeks. The model handles NaN natively and the trend
heuristic fills only at the point of scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = ["ts", "date", "rank_bracket", "hero", "role"]

LAG_DAYS = (1, 3, 7, 14)
RATE_COLUMNS = ("win_rate", "pick_rate", "ban_rate", "contest_rate")

# Thresholds for the "how long has this been true" streak counters. Deliberately crude: they
# exist as model inputs, not as a verdict.
HOT_WIN_RATE = 52.0
HOT_BAN_RATE = 20.0


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def _add_cohort_features(df: pd.DataFrame) -> pd.DataFrame:
    """z-scores within each (snapshot, rank), and within each (snapshot, rank, role)."""
    by_snap = df.groupby(["ts", "rank_bracket"], sort=False)
    for col, name in (("win_rate", "wr_z"), ("pick_rate", "pick_z"), ("contest_rate", "contest_z")):
        df[name] = by_snap[col].transform(_zscore)
    df["ban_z"] = by_snap["log_ban"].transform(_zscore)

    by_role = df.groupby(["ts", "rank_bracket", "role"], sort=False)
    df["wr_z_role"] = by_role["win_rate"].transform(_zscore)
    df["ban_z_role"] = by_role["log_ban"].transform(_zscore)
    df["pick_z_role"] = by_role["pick_rate"].transform(_zscore)

    # Same confidence weight predict.py uses: a 0.1%-pick hero's win rate is noise.
    df["conf"] = by_snap["log_pick"].transform(lambda s: s / s.max() if s.max() else 0.0)

    df["wr_rank_in_snapshot"] = by_snap["win_rate"].rank(ascending=False, method="min")
    df["ban_rank_in_snapshot"] = by_snap["ban_rate"].rank(ascending=False, method="min")
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deltas against the most recent earlier snapshot for the same hero+rank.

    merge_asof with a tolerance rather than ``.shift()``: collection can skip days (the repo's
    own archive has gaps), and shifting would quietly compare Tuesday with the previous Friday
    while calling it a 1-day delta.
    """
    right = df[["ts", "rank_bracket", "hero", *RATE_COLUMNS]].sort_values("ts")

    for lag in LAG_DAYS:
        left = df[["ts", "rank_bracket", "hero"]].copy()
        left["target_ts"] = left["ts"] - pd.Timedelta(days=lag)
        left = left.sort_values("target_ts")

        merged = pd.merge_asof(
            left,
            right.rename(columns={c: f"_past_{c}" for c in RATE_COLUMNS}),
            left_on="target_ts",
            right_on="ts",
            by=["rank_bracket", "hero"],
            direction="nearest",
            tolerance=pd.Timedelta(days=max(1, lag // 2)),
            suffixes=("", "_r"),
        ).set_index(left.index)

        for col in RATE_COLUMNS:
            short = col.replace("_rate", "")
            df[f"{short}_d{lag}"] = df[col] - merged[f"_past_{col}"]

    # Per-day slope over the last week -- the single most useful "is this still rising" number.
    df["wr_slope7"] = df["win_d7"] / 7.0
    df["ban_slope7"] = df["ban_d7"] / 7.0
    df["contest_slope7"] = df["contest_d7"] / 7.0
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-windowed rolling stats per hero+rank.

    Windowed on the timestamp (``'7D'``), not on row count, so a rank collected twice a day and
    a rank collected weekly both get a genuine seven days of context.
    """
    pieces = []
    for _, g in df.groupby(["rank_bracket", "hero"], sort=False):
        g = g.sort_values("ts")
        idx = g.index
        rolled = pd.DataFrame(index=idx)
        indexed = g.set_index("ts")

        for window, tag in (("7D", "7"), ("14D", "14")):
            r = indexed[["win_rate", "ban_rate", "pick_rate", "contest_rate"]].rolling(window, min_periods=1)
            mean, std, mx = r.mean(), r.std(ddof=0), r.max()
            rolled[f"wr_mean{tag}"] = mean["win_rate"].to_numpy()
            rolled[f"wr_std{tag}"] = std["win_rate"].to_numpy()
            rolled[f"ban_mean{tag}"] = mean["ban_rate"].to_numpy()
            rolled[f"ban_std{tag}"] = std["ban_rate"].to_numpy()
            rolled[f"pick_mean{tag}"] = mean["pick_rate"].to_numpy()
            rolled[f"contest_mean{tag}"] = mean["contest_rate"].to_numpy()
            rolled[f"wr_max{tag}"] = mx["win_rate"].to_numpy()
            rolled[f"ban_max{tag}"] = mx["ban_rate"].to_numpy()

        hot_wr = indexed["win_rate"].gt(HOT_WIN_RATE).rolling("14D", min_periods=1)
        hot_ban = indexed["ban_rate"].gt(HOT_BAN_RATE).rolling("14D", min_periods=1)
        rolled["snaps_hot_wr14"] = hot_wr.sum().to_numpy()
        rolled["snaps_hot_ban14"] = hot_ban.sum().to_numpy()
        rolled["snaps_seen14"] = indexed["win_rate"].rolling("14D", min_periods=1).count().to_numpy()
        rolled["days_tracked"] = (indexed.index - indexed.index[0]).days.to_numpy()

        pieces.append(rolled)

    rolling = pd.concat(pieces).reindex(df.index)
    df = pd.concat([df, rolling], axis=1)

    # Share of the last fortnight spent hot, rather than a raw count that depends on cadence.
    df["share_hot_wr14"] = df["snaps_hot_wr14"] / df["snaps_seen14"].replace(0, np.nan)
    df["share_hot_ban14"] = df["snaps_hot_ban14"] / df["snaps_seen14"].replace(0, np.nan)
    # How far above its own recent normal the hero is right now.
    df["wr_vs_mean7"] = df["win_rate"] - df["wr_mean7"]
    df["ban_vs_mean7"] = df["ban_rate"] - df["ban_mean7"]
    return df


def _add_cross_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gap between this rank and the hero's mean across every rank in the same snapshot."""
    by_hero_ts = df.groupby(["ts", "hero"], sort=False)
    n_ranks = by_hero_ts["rank_bracket"].transform("nunique")
    for col, name in (("win_rate", "wr_rank_spread"), ("ban_rate", "ban_rank_spread"),
                      ("pick_rate", "pick_rank_spread")):
        mean_all = by_hero_ts[col].transform("mean")
        spread = df[col] - mean_all
        df[name] = spread.where(n_ranks > 1)  # meaningless while only one rank is collected
    df["ranks_in_snapshot"] = n_ranks
    return df


def _add_patch_features(df: pd.DataFrame, patches: pd.DataFrame | None) -> pd.DataFrame:
    """Attach the active patch and how old it was when the snapshot was taken."""
    if patches is None or patches.empty:
        df["patch_id"] = ""
        df["patch_release"] = pd.NaT
        df["days_since_patch"] = np.nan
        return df

    cal = patches.sort_values("release_date")[["patch_id", "release_date"]].copy()
    left = df[["ts"]].copy().sort_values("ts")
    merged = pd.merge_asof(left, cal, left_on="ts", right_on="release_date", direction="backward").set_index(left.index)

    df["patch_id"] = merged["patch_id"].fillna("")
    df["patch_release"] = merged["release_date"]
    df["days_since_patch"] = (df["ts"] - df["patch_release"]).dt.days
    return df


def build_features(long_df: pd.DataFrame, patches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long snapshot frame -> feature rows. Pass the whole archive; filter the result afterwards.

    Trend features need history, so slicing the input to a single day produces a row with every
    delta NaN. Always build on everything, then select the snapshot you care about.
    """
    if long_df.empty:
        return pd.DataFrame(columns=ID_COLUMNS)

    df = long_df.copy()
    df = df.sort_values(["rank_bracket", "hero", "ts"]).reset_index(drop=True)

    df["contest_rate"] = df["pick_rate"] + df["ban_rate"]
    df["log_pick"] = np.log1p(df["pick_rate"])
    df["log_ban"] = np.log1p(df["ban_rate"])

    df = _add_cohort_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_cross_rank_features(df)
    df = _add_patch_features(df, patches)

    return df.sort_values(["ts", "rank_bracket", "hero"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model inputs: everything built above, minus ids, labels and bookkeeping."""
    drop = set(ID_COLUMNS) | {
        "patch_id", "patch_release", "nerfed_next", "next_patch_id",
        "snaps_hot_wr14", "snaps_hot_ban14", "snaps_seen14",
    }
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
