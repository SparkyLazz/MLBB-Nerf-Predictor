"""End-to-end check of the pipeline on a synthetic archive, with no network needed.

Builds a throwaway archive in a temp directory -- several patch windows, daily snapshots, and a
planted signal where the strongest rising heroes get nerfed in the following patch -- then runs
the real store -> features -> dataset -> model -> forecast path over it and asserts the planted
heroes come back at the top.

This is here because the pipeline has a lot of moving parts that only misbehave on data shapes
the repo doesn't have yet (several patch windows, several rank brackets, gaps in collection,
multiple snapshots in one day). Running it after a change tells you the plumbing still works
without waiting weeks for real patches to accumulate.

    python scripts/selftest.py          # quick
    python scripts/selftest.py -v       # show the intermediate numbers

Nothing it does touches data/, models/ or predictions/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROLES = ("tank", "fighter", "assassin", "mage", "marksman", "support")
N_HEROES = 60
N_PATCHES = 5
PATCH_LEN = 14
SNAPS_PER_PATCH = 10
NERFS_PER_PATCH = 4


def _write_archive(root: str, verbose: bool = False) -> dict:
    """Synthetic daily + multi-rank archive with a known nerf signal. Returns the ground truth."""
    rng = np.random.default_rng(11)
    heroes = [f"Hero{i:02d}" for i in range(N_HEROES)]
    roles = [ROLES[i % len(ROLES)] for i in range(N_HEROES)]

    os.makedirs(os.path.join(root, "data", "daily"), exist_ok=True)
    strength = rng.normal(0, 1, N_HEROES)
    pick_base = np.abs(rng.normal(1.2, 0.8, N_HEROES)) + 0.1
    ban_base = np.abs(rng.normal(8, 9, N_HEROES)) + 0.3

    start = date(2026, 1, 5)
    truth: dict[str, list[str]] = {}

    for p in range(N_PATCHES):
        pid = f"9.0.{p:02d}"
        release = start + timedelta(days=p * PATCH_LEN)
        strength = strength + rng.normal(0, 0.2, N_HEROES)
        order = np.argsort(-(strength + 0.3 * np.log1p(pick_base)))
        targets = [int(i) for i in order if pick_base[i] > 0.5][:NERFS_PER_PATCH]
        truth[pid] = [heroes[i] for i in targets]

        for d in range(SNAPS_PER_PATCH):
            day = release + timedelta(days=d)
            if d == 4:
                continue  # a deliberate collection gap -- the lag features must survive it
            climb = np.zeros(N_HEROES)
            climb[targets] = 0.12 * d
            win = 50 + 2.0 * strength + climb + rng.normal(0, 0.3, N_HEROES)
            win = win - (win.mean() - 50.0)
            pick = np.clip(pick_base * (1 + 0.05 * strength) + rng.normal(0, 0.05, N_HEROES), 0.02, None)
            ban = np.clip(ban_base * (1 + 0.25 * strength) + 2.5 * climb + rng.normal(0, 0.8, N_HEROES), 0.05, 99)

            frame = pd.DataFrame({
                "patch_id": "", "patch_date": "", "hero": heroes, "role": roles,
                "win_rate": win.round(2), "pick_rate": pick.round(2), "ban_rate": ban.round(2),
                "rank_bracket": "mythic", "nerfed_next": "",
            })
            frame.to_csv(os.path.join(root, "data", "daily", f"{day.isoformat()}.csv"), index=False)

            if d == 2:  # a second rank, and two snapshots in one day, to exercise both paths
                for rank, offset in (("mythical_glory", 0.4), ("epic", -0.3)):
                    alt = frame.copy()
                    alt["win_rate"] = (alt["win_rate"] + offset * strength).round(2)
                    alt["rank_bracket"] = rank
                    out_dir = os.path.join(root, "data", "archive", rank)
                    os.makedirs(out_dir, exist_ok=True)
                    alt.to_csv(os.path.join(out_dir, f"{day.isoformat()}T120000Z.csv"), index=False)

        strength[targets] -= 1.5
        ban_base[targets] *= 0.7

    rows = []
    for p in range(N_PATCHES + 1):
        pid = f"9.0.{p:02d}"
        prev = f"9.0.{p-1:02d}"
        rows.append({
            "patch_id": pid,
            "release_date": (start + timedelta(days=p * PATCH_LEN)).isoformat(),
            "heroes_nerfed": "|".join(truth[prev]) if p > 0 else "",
            "heroes_buffed": "", "notes": "selftest fixture",
        })
    pd.DataFrame(rows).to_csv(os.path.join(root, "data", "patches.csv"), index=False)

    if verbose:
        n = len(os.listdir(os.path.join(root, "data", "daily")))
        print(f"  fixture: {n} daily files, {N_PATCHES + 1} patches, {N_HEROES} heroes")
    return truth


def run(verbose: bool = False) -> list[str]:
    """Returns a list of failure messages; empty means everything passed."""
    import dataset as ds
    import features as featmod
    import forecast
    import model as modelmod
    import store
    import trends

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and verbose else ""))
        if not ok:
            failures.append(f"{name}{': ' + detail if detail else ''}")

    tmp = tempfile.mkdtemp(prefix="mlbb-selftest-")
    original_cwd = os.getcwd()
    try:
        truth = _write_archive(tmp, verbose=verbose)
        os.chdir(tmp)

        # --- store ---------------------------------------------------------------------
        refs = store.iter_refs()
        long_df = store.load_snapshots(use_cache=False)
        check("store reads both layouts", len(refs) > 0 and not long_df.empty,
              f"{len(refs)} snapshots, {len(long_df)} rows")
        check("store sees multiple ranks", long_df["rank_bracket"].nunique() == 3,
              f"ranks={sorted(long_df['rank_bracket'].unique())}")
        check("store dedups the legacy mirror",
              len(refs) == len({(r.ts, r.rank) for r in refs}))

        cached = store.load_snapshots(use_cache=True)
        cached2 = store.load_snapshots(use_cache=True)
        check("snapshot cache is consistent", cached.equals(cached2) and len(cached) == len(long_df))

        # --- features ------------------------------------------------------------------
        patches = ds.load_patches("data/patches.csv")
        feat = featmod.build_features(long_df, patches)
        cols = featmod.feature_columns(feat)
        check("features build", len(feat) == len(long_df) and len(cols) > 40, f"{len(cols)} features")
        check("lag features populate", feat["win_d7"].notna().mean() > 0.5,
              f"{feat['win_d7'].notna().mean():.0%} non-null")
        check("cross-rank spread populates where >1 rank",
              feat.loc[feat["ranks_in_snapshot"] > 1, "wr_rank_spread"].notna().all())
        check("cross-rank spread is NaN where 1 rank",
              feat.loc[feat["ranks_in_snapshot"] == 1, "wr_rank_spread"].isna().all())
        check("patch windows attach", feat["patch_id"].ne("").all())

        # --- dataset -------------------------------------------------------------------
        data = ds.build(use_cache=False)
        labeled, unlabeled = ds.labeled_split(data)
        info = ds.describe(data)
        check("dataset labels multiple windows", len(info["labeled_patches"]) >= N_PATCHES - 1,
              f"windows={info['labeled_patches']}")
        check("dataset has positives and negatives",
              info["positives"] > 0 and info["positives"] < info["labeled_rows"],
              f"{info['positives']}/{info['labeled_rows']} positive")
        weights = ds.window_weights(labeled)
        per_window = weights.groupby(labeled["patch_id"]).sum().round(6)
        check("row weights equalise patch windows", bool((per_window == 1.0).all()),
              f"sums={sorted(set(per_window))}")

        # --- model ---------------------------------------------------------------------
        bundle = modelmod.train(data=data, model_path=os.path.join(tmp, "m.joblib"), save=True)
        cv = bundle["cv"][bundle["estimator_name"]]["mean"]
        check("model trains", bool(cv), f"estimator={bundle['estimator_name']}")
        check("leave-one-patch-out CV beats random",
              cv.get("roc_auc", 0) > 0.6, f"roc_auc={cv.get('roc_auc', float('nan')):.3f}")
        check("model reloads", modelmod.load(os.path.join(tmp, "m.joblib")) is not None)

        # --- forecast ------------------------------------------------------------------
        for engine in ("baseline", "trend", "model"):
            ranked, meta = forecast.forecast(
                engine=engine, rank="mythic", top_n=10,
                model_path=os.path.join(tmp, "m.joblib"),
            )
            ok = len(ranked) == N_HEROES and ranked[meta["value_col"]].notna().all()
            check(f"forecast engine '{engine}' ranks every hero", ok, f"{len(ranked)} rows")

        # Planted signal: the heroes nerfed in the final recorded patch should surface near the
        # top of the last window's snapshot. The model saw this window in training, so this is a
        # plumbing check (is the signal wired through at all), not a performance claim.
        last_window = sorted(truth)[N_PATCHES - 1]
        expected = set(truth[last_window])
        ranked, meta = forecast.forecast(engine="model", rank="mythic", top_n=10,
                                         model_path=os.path.join(tmp, "m.joblib"))
        hits = len(expected & set(ranked.head(10)["hero"]))
        check("model surfaces planted nerf targets in top 10", hits >= NERFS_PER_PATCH - 1,
              f"{hits}/{len(expected)} found")

        path = forecast.write_markdown(ranked, meta, out_dir=os.path.join(tmp, "predictions"))
        check("forecast writes a report", os.path.exists(path) and os.path.getsize(path) > 200)

        # --- trends --------------------------------------------------------------------
        hist = trends.hero_history("Hero00", rank="mythic")
        check("hero history reads back", len(hist) > SNAPS_PER_PATCH, f"{len(hist)} snapshots")
        check("sparkline renders", len(trends.sparkline(hist["win_rate"])) > 0)
        health = trends.health()
        check("health spots the planted collection gap",
              any("no snapshot" in p for p in health["problems"]), f"{len(health['problems'])} problems")
        cmp = trends.compare(hist["date"].iloc[0], hist["date"].iloc[-1], rank="mythic")
        check("date comparison works", len(cmp) == N_HEROES and "d_win" in cmp.columns)

    finally:
        os.chdir(original_cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print("Self-test on a synthetic archive (nothing in data/ is touched)\n")
    try:
        failures = run(verbose=args.verbose)
    except Exception as exc:
        print(f"\nSelf-test could not run: {exc}")
        if args.verbose:
            raise
        sys.exit(1)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
