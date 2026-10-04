"""Score past patch windows against what actually got nerfed.

This is the only thing that answers "does any of this work". A ranked list is easy to produce
and impossible to trust until you check it against patches that have already shipped.

For each labeled patch window it takes the LAST snapshot inside that window -- the realistic
decision point, right before the next patch landed -- ranks every hero, and reports where the
heroes that actually got nerfed came out.

Engine honesty, which is the whole point of doing this carefully:

  baseline / trend   need no training, so scoring a past window is genuinely out-of-sample.
  model              is fitted leave-one-patch-out: to score window W the model is retrained on
                     every window EXCEPT W. Scoring a window the model trained on would be
                     in-sample and would look far better than it is. A window is skipped when
                     the remaining windows don't contain both classes, which is common early on.

Reported per window: precision@k, recall@k, and the rank of each nerfed hero, so a near-miss
(nerfed hero at rank 12) reads differently from a total miss (rank 95).

    python scripts/backtest.py                       # every engine, k=10
    python scripts/backtest.py --engines trend --k 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds  # noqa: E402
import features as featmod  # noqa: E402
import forecast as fc  # noqa: E402
import model as modelmod  # noqa: E402
import store  # noqa: E402
from predict import score as baseline_score  # noqa: E402
from promote import PATCHES, load_patches  # noqa: E402

DEFAULT_K = 10
ENGINES = ("baseline", "trend", "model")


def _next_patch_nerfs(patches: pd.DataFrame) -> dict[str, tuple[str, set[str]]]:
    cal = patches.sort_values("release_date").reset_index(drop=True)
    out = {}
    for i in range(len(cal) - 1):
        nxt = cal.loc[i + 1]
        if nxt["nerf_set"]:
            out[cal.loc[i, "patch_id"]] = (nxt["patch_id"], nxt["nerf_set"])
    return out


def _model_scores_loo(
    labeled: pd.DataFrame, window: str, cols: list[str], snap: pd.DataFrame
) -> np.ndarray | None:
    """Fit on every window except ``window``, then score exactly the rows in ``snap``.

    Scoring only the snapshot being evaluated (not the whole window) keeps the returned array
    aligned with ``snap`` row-for-row, which is what the ranking below indexes into.
    """
    train = labeled[labeled["patch_id"] != window]
    y = train[ds.LABEL].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    usable, _ = modelmod.usable_features(train[cols])
    if not usable:
        return None
    est = modelmod.build_estimators()["logreg"]
    est.fit(train[usable], y, clf__sample_weight=ds.window_weights(train).to_numpy())
    return est.predict_proba(snap[usable])[:, 1]


def run(engines: tuple[str, ...] = ENGINES, k: int = DEFAULT_K, rank: str = store.DEFAULT_RANK) -> pd.DataFrame:
    long_df = store.load_snapshots(ranks=(rank,))
    if long_df.empty:
        raise RuntimeError(f"No snapshots for rank {rank!r}.")
    patches = load_patches(PATCHES)
    feat = featmod.build_features(long_df, patches)
    data = ds.label_snapshots(feat, patches)
    labeled, _ = ds.labeled_split(data)
    if labeled.empty:
        raise RuntimeError(
            "No labeled patch windows yet, so there is nothing to back-test. Record the patch "
            "after a window you have snapshots for (menu option 4)."
        )

    cols = featmod.feature_columns(data)
    nerfs = _next_patch_nerfs(patches)
    results = []

    for window in sorted(labeled["patch_id"].unique()):
        rows = labeled[labeled["patch_id"] == window]
        last_ts = rows["ts"].max()
        snap = rows[rows["ts"] == last_ts].copy().reset_index(drop=True)
        next_id, nerf_set = nerfs.get(window, ("?", set()))
        actual = sorted(nerf_set & set(snap["hero"]))
        missing = sorted(nerf_set - set(snap["hero"]))
        if not actual:
            continue

        for engine in engines:
            if engine == "baseline":
                scores = baseline_score(snap[["hero", "win_rate", "pick_rate", "ban_rate"]])["skor"].to_numpy()
            elif engine == "trend":
                scores = np.asarray(fc.trend_score(snap))
            else:
                scores = _model_scores_loo(labeled, window, cols, snap)
                if scores is None:
                    results.append({
                        "window": window, "next_patch": next_id, "snapshot": str(last_ts.date()),
                        "engine": engine, "n_nerfed": len(actual), "hits": np.nan,
                        f"precision@{k}": np.nan, f"recall@{k}": np.nan, "median_rank": np.nan,
                        "ranks": "skipped: other windows lack both classes",
                    })
                    continue

            order = np.argsort(-scores)
            rank_of = {snap["hero"].iloc[int(i)]: pos + 1 for pos, i in enumerate(order)}
            positions = sorted(rank_of[h] for h in actual)
            hits = sum(1 for p in positions if p <= k)
            results.append({
                "window": window, "next_patch": next_id, "snapshot": str(last_ts.date()),
                "engine": engine, "n_nerfed": len(actual), "hits": hits,
                f"precision@{k}": hits / k, f"recall@{k}": hits / len(actual),
                "median_rank": float(np.median(positions)),
                "ranks": ", ".join(f"{h}#{rank_of[h]}" for h in sorted(actual, key=lambda x: rank_of[x])),
            })

        if missing:
            print(f"  note: {window}->{next_id} nerfed {missing}, not present in that snapshot "
                  f"(renamed hero, or added to the roster later)", file=sys.stderr)

    out = pd.DataFrame(results)
    out.attrs["k"] = k
    out.attrs["n_heroes"] = int(snap["hero"].nunique())
    return out


def format_report(res: pd.DataFrame) -> str:
    k = res.attrs.get("k", DEFAULT_K)
    n = res.attrs.get("n_heroes", 0)
    lines = [f"Back-test: last snapshot of each labeled window, {n} heroes ranked, k={k}", ""]

    for window in res["window"].unique():
        block = res[res["window"] == window]
        head = block.iloc[0]
        lines.append(f"  {window} -> {head['next_patch']}   snapshot {head['snapshot']}, "
                     f"{head['n_nerfed']} hero(es) actually nerfed")
        for _, row in block.iterrows():
            if pd.isna(row["hits"]):
                lines.append(f"    {row['engine']:<9} {row['ranks']}")
                continue
            lines.append(
                f"    {row['engine']:<9} {int(row['hits'])}/{int(row['n_nerfed'])} in top {k}"
                f"   precision@{k} {row[f'precision@{k}']:.2f}"
                f"   recall@{k} {row[f'recall@{k}']:.2f}"
                f"   median rank {row['median_rank']:.0f}"
            )
            lines.append(f"              {row['ranks']}")
        lines.append("")

    scored = res[res["hits"].notna()]
    if len(scored) and scored["window"].nunique() > 1:
        lines.append("  Mean across windows:")
        for engine, grp in scored.groupby("engine"):
            lines.append(f"    {engine:<9} recall@{k} {grp[f'recall@{k}'].mean():.2f}   "
                         f"precision@{k} {grp[f'precision@{k}'].mean():.2f}")
        lines.append("")

    n_windows = scored["window"].nunique()
    lines.append(f"  Based on {n_windows} patch window(s). With this few, treat every number as a")
    lines.append("  direction of travel, not a measurement -- one hero moving changes it a lot.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--rank", default=store.DEFAULT_RANK, choices=list(store.RANKS))
    args = ap.parse_args()

    engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
    bad = [e for e in engines if e not in ENGINES]
    if bad:
        print(f"ERROR: unknown engine(s) {bad}; choose from {list(ENGINES)}", file=sys.stderr)
        sys.exit(2)

    try:
        res = run(engines=engines, k=args.k, rank=args.rank)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(format_report(res))


if __name__ == "__main__":
    main()
