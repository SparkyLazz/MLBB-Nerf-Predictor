"""Read the archive directly: one hero's history, what moved between two dates, data health.

These answer the questions the ranked prediction can't. "Fredrinn is 4th on the list" is only
useful if you can also see that his ban rate has climbed every day for two weeks, and that the
archive has no four-day hole in the middle of that window.

  hero     one hero's full tracked history, with sparklines and the biggest jumps
  compare  every hero's movement between two snapshot dates, biggest first
  health   collection gaps, duplicate dates, roster changes, per-rank coverage
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store  # noqa: E402

SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float] | pd.Series, width: int = 24) -> str:
    """Tiny inline chart. Flat series render flat rather than as noise."""
    v = pd.Series(list(values), dtype=float).dropna()
    if v.empty:
        return ""
    if len(v) > width:  # thin out evenly so a long history still fits one line
        idx = np.linspace(0, len(v) - 1, width).round().astype(int)
        v = v.iloc[idx]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-9:
        return SPARK[len(SPARK) // 2] * len(v)
    scaled = ((v - lo) / (hi - lo) * (len(SPARK) - 1)).round().astype(int)
    return "".join(SPARK[i] for i in scaled)


def hero_history(hero: str, rank: str = store.DEFAULT_RANK, last: int | None = None) -> pd.DataFrame:
    long_df = store.load_snapshots(ranks=(rank,))
    if long_df.empty:
        raise RuntimeError(f"No snapshots for rank {rank!r}.")

    matches = long_df[long_df["hero"].str.lower() == hero.lower()]
    if matches.empty:
        near = sorted({h for h in long_df["hero"].unique() if hero.lower() in h.lower()})
        hint = f" Did you mean: {', '.join(near[:6])}?" if near else ""
        raise RuntimeError(f"No hero named {hero!r} in the {rank} archive.{hint}")

    out = matches.sort_values("ts")[["date", "ts", "role", "win_rate", "pick_rate", "ban_rate"]].copy()
    out["contest_rate"] = out["pick_rate"] + out["ban_rate"]
    return out.tail(last) if last else out


def format_hero_history(hist: pd.DataFrame, hero: str, rank: str) -> str:
    lines = [f"{hero} ({hist['role'].iloc[-1]}) -- {rank}, {len(hist)} snapshot(s) "
             f"{hist['date'].iloc[0]} -> {hist['date'].iloc[-1]}", ""]

    for col, label in (("win_rate", "Win %  "), ("pick_rate", "Pick % "),
                       ("ban_rate", "Ban %  "), ("contest_rate", "Contest")):
        s = hist[col]
        change = s.iloc[-1] - s.iloc[0]
        lines.append(
            f"  {label} {sparkline(s)}  now {s.iloc[-1]:6.2f}   "
            f"min {s.min():6.2f}  max {s.max():6.2f}  since start {change:+6.2f}"
        )

    lines.append("")
    recent = hist.tail(10)
    table = recent[["date", "win_rate", "pick_rate", "ban_rate"]].copy()
    table.columns = ["Date", "Win%", "Pick%", "Ban%"]
    lines.append("  Last snapshots:")
    lines.append("\n".join("  " + ln for ln in table.to_string(index=False).splitlines()))
    return "\n".join(lines)


def compare(date_a: str, date_b: str, rank: str = store.DEFAULT_RANK) -> pd.DataFrame:
    """Per-hero movement between two snapshot dates. Later snapshot wins if a date has several."""
    long_df = store.load_snapshots(ranks=(rank,))
    if long_df.empty:
        raise RuntimeError(f"No snapshots for rank {rank!r}.")

    def one(date: str) -> pd.DataFrame:
        sel = long_df[long_df["date"] == date]
        if sel.empty:
            available = sorted(long_df["date"].unique())
            raise RuntimeError(
                f"No {rank} snapshot on {date}. Archive covers {available[0]} .. {available[-1]} "
                f"({len(available)} dates)."
            )
        return sel[sel["ts"] == sel["ts"].max()].set_index("hero")

    a, b = one(date_a), one(date_b)
    heroes = a.index.intersection(b.index)
    out = pd.DataFrame({
        "hero": heroes,
        "role": b.loc[heroes, "role"].to_numpy(),
        "win_then": a.loc[heroes, "win_rate"].to_numpy(),
        "win_now": b.loc[heroes, "win_rate"].to_numpy(),
        "ban_then": a.loc[heroes, "ban_rate"].to_numpy(),
        "ban_now": b.loc[heroes, "ban_rate"].to_numpy(),
        "pick_then": a.loc[heroes, "pick_rate"].to_numpy(),
        "pick_now": b.loc[heroes, "pick_rate"].to_numpy(),
    })
    out["d_win"] = out["win_now"] - out["win_then"]
    out["d_ban"] = out["ban_now"] - out["ban_then"]
    out["d_pick"] = out["pick_now"] - out["pick_then"]

    only_a = sorted(set(a.index) - set(b.index))
    only_b = sorted(set(b.index) - set(a.index))
    out.attrs["left_out"] = only_a
    out.attrs["new_heroes"] = only_b
    return out


def health() -> dict:
    """Collection coverage and anything that looks wrong about the archive."""
    refs = store.iter_refs()
    if not refs:
        return {"ok": False, "problems": ["archive is empty -- no snapshots collected yet"], "per_rank": {}}

    problems: list[str] = []
    per_rank: dict[str, dict] = {}

    for rank in store.RANKS:
        rank_refs = [r for r in refs if r.rank == rank]
        if not rank_refs:
            continue
        dates = sorted({r.date for r in rank_refs})
        span = pd.date_range(dates[0], dates[-1], freq="D").strftime("%Y-%m-%d").tolist()
        missing = [d for d in span if d not in set(dates)]
        per_rank[rank] = {
            "snapshots": len(rank_refs),
            "dates": len(dates),
            "first": dates[0],
            "last": dates[-1],
            "span_days": len(span),
            "missing_days": missing,
            "coverage": len(dates) / len(span) if span else 0.0,
        }
        if missing:
            problems.append(
                f"{rank}: {len(missing)} day(s) with no snapshot between {dates[0]} and {dates[-1]}"
                + (f" (e.g. {', '.join(missing[:5])})" if missing else "")
            )

    # Roster drift: hero count changing over time is worth surfacing, not hiding.
    long_df = store.load_snapshots()
    counts = long_df.groupby(["ts", "rank_bracket"])["hero"].nunique()
    if counts.nunique() > 1:
        problems.append(
            f"hero count varies across snapshots ({counts.min()}..{counts.max()}) -- "
            "a new hero, or some scrapes were incomplete"
        )

    nulls = int(long_df[["win_rate", "pick_rate", "ban_rate"]].isna().sum().sum())
    if nulls:
        problems.append(f"{nulls} missing rate value(s) across the archive")

    means = long_df.groupby(["ts", "rank_bracket"])["win_rate"].mean()
    off_band = means[(means < 48) | (means > 52)]
    if len(off_band):
        problems.append(f"{len(off_band)} snapshot(s) with a mean win rate outside 48-52% -- suspect scrapes")

    return {
        "ok": not problems,
        "problems": problems,
        "per_rank": per_rank,
        "snapshots": len(refs),
        "rows": len(long_df),
        "heroes_latest": int(long_df[long_df["ts"] == long_df["ts"].max()]["hero"].nunique()),
    }


def format_health(h: dict) -> str:
    lines = []
    if not h["per_rank"]:
        return "Archive is empty -- collect a snapshot first."
    lines.append(f"{h['snapshots']} snapshot(s), {h['rows']} rows, {h['heroes_latest']} heroes in the latest snapshot")
    lines.append("")
    lines.append(f"  {'rank':<16} {'snaps':>6} {'dates':>6} {'span':>6} {'cover':>7}  range")
    for rank, s in h["per_rank"].items():
        lines.append(
            f"  {rank:<16} {s['snapshots']:>6} {s['dates']:>6} {s['span_days']:>6} "
            f"{s['coverage']:>6.0%}  {s['first']} .. {s['last']}"
        )
    lines.append("")
    if h["ok"]:
        lines.append("  No problems found.")
    else:
        lines.append("  Worth a look:")
        for p in h["problems"]:
            lines.append(f"   - {p}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hero", help="One hero's tracked history.")
    h.add_argument("name")
    h.add_argument("--rank", default=store.DEFAULT_RANK, choices=list(store.RANKS))
    h.add_argument("--last", type=int, help="Only the last N snapshots.")

    c = sub.add_parser("compare", help="What moved between two snapshot dates.")
    c.add_argument("date_a")
    c.add_argument("date_b")
    c.add_argument("--rank", default=store.DEFAULT_RANK, choices=list(store.RANKS))
    c.add_argument("--by", default="d_win", choices=["d_win", "d_ban", "d_pick"])
    c.add_argument("--top", type=int, default=15)

    sub.add_parser("health", help="Collection coverage and data problems.")

    args = ap.parse_args()
    try:
        if args.cmd == "hero":
            hist = hero_history(args.name, rank=args.rank, last=args.last)
            canonical = store.load_snapshots(ranks=(args.rank,))
            canonical = canonical[canonical["hero"].str.lower() == args.name.lower()]["hero"].iloc[0]
            print(format_hero_history(hist, canonical, args.rank))
        elif args.cmd == "compare":
            cmp = compare(args.date_a, args.date_b, rank=args.rank)
            print(f"{args.rank}: {args.date_a} -> {args.date_b}, {len(cmp)} heroes in both\n")
            show = cmp.reindex(cmp[args.by].abs().sort_values(ascending=False).index).head(args.top)
            cols = ["hero", "role", "win_then", "win_now", "d_win", "ban_then", "ban_now", "d_ban", "d_pick"]
            print(show[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
            if cmp.attrs["new_heroes"]:
                print(f"\nNew since {args.date_a}: {', '.join(cmp.attrs['new_heroes'])}")
            if cmp.attrs["left_out"]:
                print(f"Gone since {args.date_a}: {', '.join(cmp.attrs['left_out'])}")
        else:
            print(format_health(health()))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
