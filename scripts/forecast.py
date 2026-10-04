"""Rank nerf candidates using the whole archive, not just today.

Three engines, picked with --engine (default: auto):

  model      the trained classifier from scripts/model.py. Outputs a calibrated-ish probability
             per hero. Needs labeled patch windows to exist, so it is unavailable until enough
             patches are recorded.
  trend      no training needed. Takes the ORIGINAL heuristic from scripts/predict.py exactly as
             written -- same function, same 0.45/0.55 weights, imported not copied -- and adds
             momentum and persistence terms on top of it. Works from day one.
  baseline   scripts/predict.py's heuristic alone, single snapshot, no history. Kept so you can
             always see what the extra machinery is actually adding.
  auto       model if one is trained and loadable, otherwise trend.

The trend engine's extra weights (0.30 / 0.25 / 0.15 / 0.20 / 0.15 below) are unvalidated
guesses, exactly like the baseline's 0.45/0.55 -- that is what the model engine exists to
replace. They are declared here rather than hidden so they can be argued with.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as featmod  # noqa: E402
import model as modelmod  # noqa: E402
import store  # noqa: E402
from predict import score as baseline_score  # the untouched original heuristic  # noqa: E402
from promote import PATCHES, load_patches  # noqa: E402

PREDICTIONS_DIR = "predictions"
TOP_N = 10

# Trend-engine weights on top of the baseline score. Unvalidated.
W_WIN_D7 = 0.30
W_BAN_D7 = 0.25
W_CONTEST_D7 = 0.15
W_HOT_WR = 0.20
W_HOT_BAN = 0.15

# A "share of the last fortnight spent hot" computed from one or two snapshots is not
# persistence, it is just the current reading wearing a persistence label -- and it would
# double-count the level terms it sits next to. Below this many snapshots in the window, the
# persistence terms are dropped and not cited as drivers. This bites after a collection outage:
# the 22-day gap in this archive left exactly one snapshot in the trailing fortnight.
MIN_SNAPS_FOR_PERSISTENCE = 3

# Features the "drivers" column is allowed to cite, with human wording. Restricted on purpose:
# a driver list is meant to be read by a person deciding whether to believe the ranking.
DRIVER_LABELS = {
    "win_rate": "high win rate", "wr_z": "win rate vs field", "wr_z_role": "win rate vs role",
    "wr_mean7": "sustained win rate", "wr_max14": "14d win peak", "wr_vs_mean7": "above own normal",
    "win_d1": "win rate up 1d", "win_d3": "win rate up 3d", "win_d7": "win rate up 7d",
    "win_d14": "win rate up 14d", "wr_slope7": "win rate climbing",
    "ban_rate": "high ban rate", "ban_z": "bans vs field", "ban_z_role": "bans vs role",
    "ban_d3": "bans up 3d", "ban_d7": "bans up 7d", "ban_d14": "bans up 14d",
    "ban_mean7": "sustained bans", "ban_slope7": "bans climbing", "ban_vs_mean7": "bans above normal",
    "pick_rate": "high pick rate", "pick_z": "picks vs field", "pick_d7": "picks up 7d",
    "contest_rate": "heavily contested", "contest_d7": "contest up 7d", "contest_slope7": "contest climbing",
    "share_hot_wr14": "hot win rate all fortnight", "share_hot_ban14": "banned all fortnight",
    "wr_rank_spread": "stronger in this rank", "ban_rank_spread": "banned more in this rank",
    "days_since_patch": "patch age",
}


def _zfill(s: pd.Series) -> pd.Series:
    """z-score with missing history treated as 'no movement' rather than dropped.

    Only the trend engine does this, and only at scoring time: a hero with four days of data
    gets a 0 trend contribution instead of being excluded from the ranking entirely.
    """
    s = s.astype(float)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - s.mean()) / sd).fillna(0.0)


def trend_score(snap_features: pd.DataFrame) -> pd.Series:
    """Baseline heuristic + momentum + persistence, on one snapshot's feature rows."""
    base = baseline_score(snap_features[["hero", "win_rate", "pick_rate", "ban_rate"]])["skor"]
    conf = snap_features["conf"].fillna(0.0)

    momentum = (
        W_WIN_D7 * _zfill(snap_features["win_d7"])
        + W_BAN_D7 * _zfill(snap_features["ban_d7"])
        + W_CONTEST_D7 * _zfill(snap_features["contest_d7"])
    ) * conf
    enough_history = snap_features["snaps_seen14"].fillna(0) >= MIN_SNAPS_FOR_PERSISTENCE
    persistence = (
        W_HOT_WR * snap_features["share_hot_wr14"].fillna(0.0)
        + W_HOT_BAN * snap_features["share_hot_ban14"].fillna(0.0)
    ).where(enough_history, 0.0)
    return base.to_numpy() + momentum + persistence


def drivers(snap_features: pd.DataFrame, weights: dict[str, float], top: int = 3) -> list[str]:
    """Per-hero plain-English reason strings.

    Each candidate feature is turned into a z-score across the snapshot, multiplied by how much
    the engine leans on that feature, and the largest few are named. This describes what the
    ranking is keying on -- it is not a causal claim about Moonton's balance team.
    """
    usable = {f: w for f, w in weights.items() if f in snap_features.columns and f in DRIVER_LABELS}
    if not usable:
        return [""] * len(snap_features)

    # Don't let a feature describe itself as a fortnight-long pattern on one snapshot.
    if "snaps_seen14" in snap_features.columns:
        thin = snap_features["snaps_seen14"].fillna(0) < MIN_SNAPS_FOR_PERSISTENCE
    else:
        thin = pd.Series(False, index=snap_features.index)

    contrib = pd.DataFrame(index=snap_features.index)
    for feat, w in usable.items():
        values = _zfill(snap_features[feat]) * w
        if feat in ("share_hot_wr14", "share_hot_ban14"):
            values = values.where(~thin, 0.0)
        contrib[feat] = values

    out = []
    for idx in snap_features.index:
        row = contrib.loc[idx].sort_values(key=lambda s: s.abs(), ascending=False)
        picked = [DRIVER_LABELS[f] for f, v in row.items() if v > 0][:top]
        out.append(", ".join(picked))
    return out


def resolve_engine(requested: str, model_path: str = modelmod.MODEL_PATH) -> tuple[str, dict | None]:
    if requested in ("model", "auto"):
        bundle = modelmod.load(model_path)
        if bundle is not None:
            return "model", bundle
        if requested == "model":
            raise RuntimeError(
                f"No usable model at {model_path}. Train one (menu option 8) or use "
                "--engine trend, which needs no training."
            )
        return "trend", None
    if requested in ("trend", "baseline"):
        return requested, None
    raise RuntimeError(f"Unknown engine {requested!r}")


def forecast(
    engine: str = "auto",
    rank: str = store.DEFAULT_RANK,
    date: str | None = None,
    top_n: int = TOP_N,
    model_path: str = modelmod.MODEL_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Returns (ranked table, metadata). ``date`` scores a historical snapshot instead of the latest."""
    engine, bundle = resolve_engine(engine, model_path)

    long_df = store.load_snapshots(ranks=(rank,))
    if long_df.empty:
        raise RuntimeError(f"No snapshots for rank {rank!r}. Collect one first (menu option 1).")

    patches = load_patches(PATCHES) if os.path.exists(PATCHES) else None
    feat = featmod.build_features(long_df, patches)

    if date:
        candidates = feat[feat["date"] == date]
        if candidates.empty:
            raise RuntimeError(f"No {rank} snapshot on {date}. Available: {feat['date'].min()}..{feat['date'].max()}")
        target_ts = candidates["ts"].max()
    else:
        target_ts = feat["ts"].max()

    snap = feat[feat["ts"] == target_ts].copy().reset_index(drop=True)

    if engine == "baseline":
        snap["skor"] = baseline_score(snap[["hero", "win_rate", "pick_rate", "ban_rate"]])["skor"].to_numpy()
        snap["driver"] = ""
        value_col, value_label = "skor", "Score"
    elif engine == "trend":
        snap["skor"] = trend_score(snap).to_numpy()
        snap["driver"] = drivers(snap, {
            "win_rate": 0.45, "ban_rate": 0.55, "win_d7": W_WIN_D7, "ban_d7": W_BAN_D7,
            "contest_d7": W_CONTEST_D7, "share_hot_wr14": W_HOT_WR, "share_hot_ban14": W_HOT_BAN,
        })
        value_col, value_label = "skor", "Score"
    else:  # model
        snap["prob"] = modelmod.predict_proba(bundle, snap)
        imp = bundle["importances"].head(25)
        snap["driver"] = drivers(snap, dict(zip(imp["feature"], imp["weight"])))
        value_col, value_label = "prob", "P(nerf)"

    ranked = snap.sort_values(value_col, ascending=False).reset_index(drop=True)

    meta = {
        "engine": engine,
        "rank": rank,
        "ts": target_ts,
        "date": ranked["date"].iloc[0] if len(ranked) else (date or ""),
        "patch_id": ranked["patch_id"].iloc[0] if len(ranked) and "patch_id" in ranked else "",
        "value_col": value_col,
        "value_label": value_label,
        "snapshots_used": int(long_df["ts"].nunique()),
        "history_days": int((long_df["ts"].max() - long_df["ts"].min()).days) + 1,
        "heroes": int(len(ranked)),
        "top_n": top_n,
        "model": None if bundle is None else {
            "name": bundle["estimator_name"],
            "trained_at": bundle["trained_at"],
            "patch_windows": bundle["patch_windows"],
            "cv": bundle["cv"][bundle["estimator_name"]]["mean"],
            "cv_reliable": bundle["cv_reliable"],
        },
    }
    return ranked, meta


DISPLAY = ["hero", "role", "win_rate", "pick_rate", "ban_rate", "win_d7", "ban_d7"]


def format_console(ranked: pd.DataFrame, meta: dict) -> str:
    v = meta["value_col"]
    top = ranked.head(meta["top_n"])
    show = top[DISPLAY + [v, "driver"]].copy()
    show.columns = ["Hero", "Role", "Win%", "Pick%", "Ban%", "d7Win", "d7Ban", meta["value_label"], "Why"]
    show.insert(0, "#", range(1, len(show) + 1))
    for col in ("Win%", "Pick%", "Ban%", "d7Win", "d7Ban"):
        show[col] = show[col].map(lambda x: "   -  " if pd.isna(x) else f"{x:6.2f}")
    show[meta["value_label"]] = show[meta["value_label"]].map(
        lambda x: f"{x:6.1%}" if v == "prob" else f"{x:6.3f}"
    )
    return show.to_string(index=False)


def write_markdown(ranked: pd.DataFrame, meta: dict, out_dir: str = PREDICTIONS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    patch = meta["patch_id"] or "unlabeled"
    out_path = os.path.join(out_dir, f"{meta['date']}_{patch}_{meta['rank']}_{meta['engine']}.md")
    v, label = meta["value_col"], meta["value_label"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Nerf candidates -- {meta['date']} ({meta['rank']}, patch {patch})\n\n")
        f.write(f"- Engine: **{meta['engine']}**\n")
        f.write(f"- History used: {meta['snapshots_used']} snapshot(s) over {meta['history_days']} day(s), "
                f"{meta['heroes']} heroes\n")
        if meta["model"]:
            m = meta["model"]
            cv = ", ".join(f"{k}={val:.3f}" for k, val in m["cv"].items()) or "no usable folds"
            f.write(f"- Model: `{m['name']}` trained {m['trained_at']} on patches {', '.join(m['patch_windows'])}\n")
            f.write(f"- Leave-one-patch-out CV: {cv}\n")
            if not m["cv_reliable"]:
                f.write("- **Caveat:** too few labeled patch windows for those scores to be trustworthy yet.\n")
        else:
            f.write("- Weights are unvalidated heuristics "
                    "(baseline 0.45/0.55 from `predict.py`, plus momentum and persistence terms).\n")
        f.write("\n")
        f.write(f"| # | Hero | Role | Win% | Pick% | Ban% | 7d Win | 7d Ban | {label} | Why |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        def num(x, fmt="{:.2f}"):
            return "–" if pd.isna(x) else fmt.format(x)

        for i, (_, row) in enumerate(ranked.head(meta["top_n"]).iterrows(), start=1):
            val = f"{row[v]:.1%}" if v == "prob" else f"{row[v]:.3f}"
            f.write(
                f"| {i} | {row['hero']} | {row['role']} | {num(row['win_rate'])} | {num(row['pick_rate'])} | "
                f"{num(row['ban_rate'])} | {num(row['win_d7'], '{:+.2f}')} | {num(row['ban_d7'], '{:+.2f}')} | "
                f"{val} | {row['driver']} |\n"
            )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", default="auto", choices=["auto", "model", "trend", "baseline"])
    ap.add_argument("--rank", default=store.DEFAULT_RANK, choices=list(store.RANKS))
    ap.add_argument("--date", help="Score this snapshot date (YYYY-MM-DD) instead of the latest.")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--no-write", action="store_true", help="Print only, don't write predictions/.")
    args = ap.parse_args()

    try:
        ranked, meta = forecast(engine=args.engine, rank=args.rank, date=args.date, top_n=args.top)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Engine: {meta['engine']}  |  {meta['rank']}  |  snapshot {meta['date']}  |  "
          f"{meta['snapshots_used']} snapshot(s) of history")
    if meta["model"] and not meta["model"]["cv_reliable"]:
        print("NOTE: model trained on very few patch windows -- treat the probabilities as rough.")
    print()
    print(format_console(ranked, meta))

    if not args.no_write:
        print(f"\nWrote {write_markdown(ranked, meta)}")


if __name__ == "__main__":
    main()
