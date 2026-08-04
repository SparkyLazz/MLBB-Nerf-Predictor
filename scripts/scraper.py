"""Scrape live MLBB hero win/pick/ban rates from mlbbhub.com/statistics via headless Playwright.

Moved from the original mlbb_scraper.py prototype -- fetch_stats() and the extraction JS are
unchanged from the version already verified against the live site. Only the surrounding
orchestration changed, to fit this repo's architecture:

- `daily`: writes an unlabeled raw snapshot to data/daily/<UTC date>.csv. data/daily/ is
  append-only -- this NEVER overwrites or deletes an existing file for that date, and there is no
  --force escape hatch for it (unlike the old prototype). This is the only thing the scheduled
  cron trigger ever runs.
- `labeled`: the ONLY way a new row gets added to data/patches.csv, which is the hand-curated
  patch calendar (release dates + nerf/buff hero lists) that scripts/promote.py joins against
  data/daily/ to build data/train.csv. patch_id/patch_date (and heroes_nerfed/heroes_buffed, if
  known yet) are caller-supplied strings -- this script never infers or guesses them. Also takes
  a same-day daily snapshot (append-only, same rule as above) so there's real data to anchor the
  new patch's window to. This is the only thing workflow_dispatch ever runs.

Neither subcommand writes to data/train.csv. That file is a pure derived artifact -- see
scripts/promote.py.

The mean-win-rate validation lives in fetch_stats() itself (the fetch path), not in either
save function, so there is no way to save data without going through it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from playwright.sync_api import Page, sync_playwright

STATS_URL = "https://mlbbhub.com/statistics"
EXPECTED_HERO_COUNT = 133  # sanity check; bump this if Moonton ships a new hero
WIN_RATE_MEAN_BOUNDS = (48.0, 52.0)  # every match has one winner and one loser -> average should hover ~50%

DAILY_DIR_DEFAULT = "data/daily"
PATCHES_DEFAULT = "data/patches.csv"
PATCHES_COLUMNS = ["patch_id", "release_date", "heroes_nerfed", "heroes_buffed", "notes"]

RankKey = Literal["epic", "legend", "mythic", "mythical_honor", "mythical_glory"]

RANK_LABELS: dict[str, str] = {
    "epic": "Epic",
    "legend": "Legend",
    "mythic": "Mythic",
    "mythical_honor": "Mythical Honor",
    "mythical_glory": "Mythical Glory",
}

# Verified interactively against the live site -- kept as one block so it's trivial to diff
# against that transcript if the site layout ever changes.
_EXTRACT_JS = r"""
() => {
  const container = document.querySelector('.md\\:hidden.space-y-\\[2px\\]');
  if (!container) {
    return { error: 'stats table container not found', bodyLen: document.body.innerText.length };
  }
  const rows = Array.from(container.children);
  const data = rows.map(row => {
    const nameImg = row.querySelector('img[alt$=" hero icon"]');
    const name = nameImg ? nameImg.alt.replace(' hero icon', '') : null;
    const roleImgs = Array.from(row.querySelectorAll('img[alt$=" icon"]'))
      .filter(i => !i.alt.includes(' hero icon'))
      .map(i => i.alt.replace(' icon', ''));
    const winSpan = Array.from(row.querySelectorAll('div')).find(d => d.className.includes('grid-cols-3'));
    const statsText = winSpan ? winSpan.innerText : row.innerText;
    const win = (statsText.match(/Win([\d.]+)%/) || [])[1];
    const pick = (statsText.match(/Pick([\d.]+)%/) || [])[1];
    const ban = (statsText.match(/Ban([\d.]+)%/) || [])[1];
    return { name, role: roleImgs[0] || null, win, pick, ban };
  });
  return { count: data.length, data };
}
"""


def _click_rank_filter(page: Page, rank_label: str, timeout_ms: int) -> None:
    button = page.get_by_role("button", name=rank_label, exact=True)
    button.wait_for(state="visible", timeout=timeout_ms)
    button.click()


def fetch_stats(
    rank: RankKey = "mythic",
    headless: bool = True,
    timeout_ms: int = 30_000,
) -> pd.DataFrame:
    """Scrape mlbbhub.com/statistics for one rank bracket.

    Opens the page, clicks the requested rank filter, waits for the table to re-render, runs the
    same extraction JS used interactively, and returns a validated DataFrame with columns:
    hero, role, win_rate, pick_rate, ban_rate, rank_bracket.

    Raises RuntimeError -- and therefore writes nothing anywhere, since callers only touch disk
    after this returns -- if the scrape looks wrong in any way: missing container, wrong hero
    count, nulls, duplicates, or a mean win_rate outside WIN_RATE_MEAN_BOUNDS.
    """
    if rank not in RANK_LABELS:
        raise ValueError(f"rank must be one of {list(RANK_LABELS)}, got {rank!r}")
    rank_label = RANK_LABELS[rank]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            # "networkidle" never fires on this page (continuous ad/analytics traffic), so wait for
            # DOM content instead and rely on explicit element waits below for real readiness.
            page.goto(STATS_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            _click_rank_filter(page, rank_label, timeout_ms)

            # Wait for the filtered table to actually contain rows, not just for the click to register.
            page.wait_for_function(
                r"""() => {
                    const c = document.querySelector('.md\\:hidden.space-y-\\[2px\\]');
                    return !!c && c.children.length > 0;
                }""",
                timeout=timeout_ms,
            )
            # The filter re-render is client-side and can briefly show stale rows mid-transition.
            page.wait_for_timeout(500)

            result = page.evaluate(_EXTRACT_JS)
        finally:
            browser.close()

    if "error" in result:
        raise RuntimeError(f"Could not read the stats table: {result}")

    df = pd.DataFrame(result["data"])

    if df.empty:
        raise RuntimeError("Scrape returned zero hero rows -- page structure may have changed.")
    if len(df) != EXPECTED_HERO_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_HERO_COUNT} heroes, got {len(df)}. "
            "Hero roster or DOM structure may have changed -- check before trusting this data "
            "(update EXPECTED_HERO_COUNT once you've confirmed it's a real roster change, not a scrape bug)."
        )
    if df[["name", "role", "win", "pick", "ban"]].isnull().any().any():
        bad = df[df[["name", "role", "win", "pick", "ban"]].isnull().any(axis=1)]
        raise RuntimeError(f"Scrape returned incomplete rows, refusing to proceed:\n{bad}")
    if df["name"].duplicated().any():
        dupes = df[df["name"].duplicated(keep=False)]
        raise RuntimeError(f"Scrape returned duplicate heroes, refusing to proceed:\n{dupes}")

    df["role"] = df["role"].str.lower()
    df[["win", "pick", "ban"]] = df[["win", "pick", "ban"]].astype(float)

    # Validation lives here -- the fetch path -- on purpose, so no save function can bypass it.
    mean_win = df["win"].mean()
    lo, hi = WIN_RATE_MEAN_BOUNDS
    if not (lo <= mean_win <= hi):
        raise RuntimeError(
            f"Mean win_rate is {mean_win:.2f}%, outside the expected {lo}-{hi}% band. "
            "Every ranked match has exactly one winner and one loser, so the hero-average win rate "
            "should hover near 50% -- this reading means the scrape is probably broken (wrong rank "
            "filter applied, stale/cached page, or the wrong column got parsed). Refusing to write anything."
        )

    df["rank_bracket"] = rank
    df = df.rename(columns={"name": "hero", "win": "win_rate", "pick": "pick_rate", "ban": "ban_rate"})
    return df[["hero", "role", "win_rate", "pick_rate", "ban_rate", "rank_bracket"]].reset_index(drop=True)


def save_daily_snapshot(
    output_dir: str = DAILY_DIR_DEFAULT,
    rank: RankKey = "mythic",
    headless: bool = True,
) -> str:
    """Write one UNLABELED raw snapshot to output_dir/<UTC date>.csv.

    Append-only: if today's file already exists, this is a no-op (prints and returns the existing
    path). There is deliberately no overwrite option here -- data/daily/ must never be mutated by
    a script once a file has landed.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{today}.csv")
    if os.path.exists(out_path):
        print(f"{out_path} already exists -- data/daily/ is append-only, leaving it untouched.")
        return out_path

    df = fetch_stats(rank=rank, headless=headless)
    df.insert(0, "patch_date", "")
    df.insert(0, "patch_id", "")
    df["nerfed_next"] = ""
    df = df[
        ["patch_id", "patch_date", "hero", "role", "win_rate", "pick_rate", "ban_rate", "rank_bracket", "nerfed_next"]
    ]
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return out_path


def append_patch_record(
    patches_csv: str,
    patch_id: str,
    patch_date: str,
    heroes_nerfed: str = "",
    heroes_buffed: str = "",
    notes: str = "",
    force: bool = False,
) -> None:
    """Append one row to data/patches.csv. patch_id/patch_date/hero lists are all caller-supplied
    -- never inferred. Idempotent by default: skips if patch_id is already present, unless force=True.
    """
    os.makedirs(os.path.dirname(patches_csv) or ".", exist_ok=True)

    if os.path.exists(patches_csv) and os.path.getsize(patches_csv) > 0:
        existing = pd.read_csv(patches_csv, dtype=str).fillna("")
        if patch_id in existing["patch_id"].values:
            if not force:
                print(f"patch_id={patch_id} already present in {patches_csv}, skipping (use --force to overwrite).")
                return
            existing = existing[existing["patch_id"] != patch_id]
    else:
        existing = pd.DataFrame(columns=PATCHES_COLUMNS)

    new_row = pd.DataFrame([{
        "patch_id": patch_id,
        "release_date": patch_date,
        "heroes_nerfed": heroes_nerfed,
        "heroes_buffed": heroes_buffed,
        "notes": notes,
    }])
    combined = pd.concat([existing, new_row], ignore_index=True)
    combined.to_csv(patches_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Recorded patch {patch_id} ({patch_date}) -> {patches_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape MLBB hero stats. `daily` for the scheduled cron archive, "
        "`labeled` for recording a new patch via workflow_dispatch."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    daily_parser = subparsers.add_parser(
        "daily",
        help="Write an unlabeled raw snapshot to data/daily/<UTC date>.csv. Append-only, no overwrite option.",
    )
    daily_parser.add_argument("--rank", default="mythic", choices=list(RANK_LABELS))
    daily_parser.add_argument("--output-dir", default=DAILY_DIR_DEFAULT)
    daily_parser.add_argument("--no-headless", action="store_true", help="Run with a visible browser (local debugging only).")

    labeled_parser = subparsers.add_parser(
        "labeled",
        help="Record a new patch in data/patches.csv, plus a same-day daily snapshot. "
        "patch_id/patch_date must be human-confirmed -- never guessed.",
    )
    labeled_parser.add_argument("--patch-id", required=True, help='e.g. "2.1.92"')
    labeled_parser.add_argument("--patch-date", required=True, help='e.g. "2026-07-29" (YYYY-MM-DD)')
    labeled_parser.add_argument("--nerfed", default="", help='Pipe-separated hero list, e.g. "Baxia|Akai". Leave blank to fill in later.')
    labeled_parser.add_argument("--buffed", default="", help="Pipe-separated hero list.")
    labeled_parser.add_argument("--notes", default="")
    labeled_parser.add_argument("--patches-csv", default=PATCHES_DEFAULT)
    labeled_parser.add_argument("--daily-dir", default=DAILY_DIR_DEFAULT)
    labeled_parser.add_argument("--rank", default="mythic", choices=list(RANK_LABELS))
    labeled_parser.add_argument("--no-daily-snapshot", action="store_true", help="Skip taking a same-day daily snapshot.")
    labeled_parser.add_argument("--force", action="store_true", help="Overwrite this patch_id's row in patches.csv if already present.")
    labeled_parser.add_argument("--no-headless", action="store_true", help="Run with a visible browser (local debugging only).")

    args = parser.parse_args()

    try:
        if args.mode == "daily":
            save_daily_snapshot(output_dir=args.output_dir, rank=args.rank, headless=not args.no_headless)
        else:  # labeled
            if not args.no_daily_snapshot:
                save_daily_snapshot(output_dir=args.daily_dir, rank=args.rank, headless=not args.no_headless)
            append_patch_record(
                patches_csv=args.patches_csv,
                patch_id=args.patch_id,
                patch_date=args.patch_date,
                heroes_nerfed=args.nerfed,
                heroes_buffed=args.buffed,
                notes=args.notes,
                force=args.force,
            )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
