"""Read the official patch notes and propose rows for data/patches.csv.

Why this exists: nothing in this repo can be trained until patch windows are *labeled*, and a
window is only labeled once the patch after it is recorded with its ``heroes_nerfed`` list. Doing
that by hand means the model stays untrainable until someone transcribes a patch note. As of
writing, data/patches.csv has two rows and zero labeled windows -- so zero training rows -- even
though a month of snapshots has been collected.

How it respects the repo's rule that patch identity is "never auto-detected":

  * It defaults to a DRY RUN. It prints the rows it would add and changes nothing. Writing needs
    an explicit --apply, which is a person deciding, same as typing the row by hand.
  * It never invents a hero name. Every name parsed out of a patch page is checked against the
    hero roster in the snapshot archive, and anything that isn't a hero the scraper has actually
    seen is reported as unmatched and dropped rather than guessed at.
  * It never edits or removes a row that already exists; --apply only appends new patch_ids
    unless you also pass --force.
  * Every row it writes carries its source URL and fetch time in the ``notes`` column, so a
    parsed row is always distinguishable from a hand-entered one.
  * Nothing on a schedule calls it. The daily cron still only ever collects stats.

It reads the live-server notes (``/patch-notes/original/<version>``) by default. The Advance
Server list (``/patch-notes/advanced/<version>``) runs weeks ahead of live and its numbers are
not what the stats archive reflects, so those are opt-in via --channel advanced.

HEADS UP ON THE PARSER: the DOM selectors here are written defensively but have NOT been
verified against a live page -- unlike the stats scraper, whose extraction JS was confirmed
interactively. Mlbbhub was unreachable from where this was written. Read the dry-run output
before trusting it, and use --dump-html to inspect a page when a parse comes back empty or odd.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store  # noqa: E402
from scraper import PATCHES_COLUMNS, PATCHES_DEFAULT  # noqa: E402

INDEX_URL = "https://mlbbhub.com/patch-notes"
PATCH_URL = "https://mlbbhub.com/patch-notes/{channel}/{version}"
CHANNELS = ("original", "advanced")  # original = live server, advanced = test server

NERF_WORDS = ("nerf", "weaken", "reduced", "decrease")
BUFF_WORDS = ("buff", "strengthen", "enhanced", "increase")

_VERSION_RE = re.compile(r"/patch-notes/(original|advanced)/([0-9][0-9A-Za-z.\-]*)")
_DATE_PATTERNS = (
    re.compile(r"(\d{4}-\d{2}-\d{2})"),
    re.compile(r"([A-Z][a-z]+ \d{1,2},? \d{4})"),
    re.compile(r"(\d{1,2} [A-Z][a-z]+ \d{4})"),
)

# Pulls the patch index: every patch link on the page, in document order.
_INDEX_JS = r"""
() => {
  const links = Array.from(document.querySelectorAll('a[href*="/patch-notes/"]'));
  const seen = new Set();
  const out = [];
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/patch-notes\/(original|advanced)\/([0-9][0-9A-Za-z.\-]*)/);
    if (!m) continue;
    const key = m[1] + '/' + m[2];
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ channel: m[1], version: m[2], href, text: (a.innerText || '').trim().slice(0, 200) });
  }
  return { count: out.length, patches: out, bodyLen: document.body.innerText.length };
}
"""

# Pulls one patch page as (heading, text-under-heading) blocks plus the raw body text. Parsing
# which heroes were nerfed is done in Python against the known roster, not with guesswork here.
_PATCH_JS = r"""
() => {
  const body = document.body.innerText || '';
  const nodes = Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,strong,b,[class*="title"],[class*="heading"]'));
  const blocks = [];
  for (const node of nodes) {
    const heading = (node.innerText || '').trim();
    if (!heading || heading.length > 120) continue;
    let text = '';
    let el = node;
    for (let i = 0; i < 40 && el; i++) {
      el = el.nextElementSibling;
      if (!el) break;
      if (/^H[1-5]$/.test(el.tagName)) break;
      text += ' ' + (el.innerText || '');
    }
    const parent = node.parentElement;
    if (!text.trim() && parent) text = parent.innerText || '';
    blocks.push({ heading, text: text.trim().slice(0, 4000) });
  }
  const heroImgs = Array.from(document.querySelectorAll('img[alt]'))
    .map(i => i.alt.replace(/ (hero )?icon$/i, '').trim())
    .filter(Boolean);
  return { body: body.slice(0, 60000), blocks, heroImgs, title: document.title };
}
"""


def known_heroes() -> list[str]:
    """Hero vocabulary from the snapshot archive -- the only names allowed into patches.csv."""
    long_df = store.load_snapshots()
    if long_df.empty:
        raise RuntimeError(
            "The snapshot archive is empty, so there is no hero roster to validate patch-note "
            "names against. Collect at least one snapshot first (menu option 1)."
        )
    latest = long_df[long_df["ts"] == long_df["ts"].max()]
    return sorted(latest["hero"].unique().tolist())


def _find_heroes(text: str, roster: list[str]) -> list[str]:
    """Hero names present in ``text``, longest-first so "Popol and Kupa" wins over "Kupa"."""
    found: list[str] = []
    haystack = text.lower()
    for hero in sorted(roster, key=len, reverse=True):
        needle = hero.lower()
        if re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", haystack):
            if not any(hero.lower() in f.lower() and hero != f for f in found):
                found.append(hero)
    return found


def _classify_blocks(blocks: list[dict], roster: list[str]) -> tuple[list[str], list[str]]:
    """Walk heading blocks, bucket heroes into nerfed/buffed by the heading's wording."""
    nerfed: list[str] = []
    buffed: list[str] = []
    for block in blocks:
        head = block["heading"].lower()
        is_nerf = any(w in head for w in NERF_WORDS)
        is_buff = any(w in head for w in BUFF_WORDS)
        if is_nerf == is_buff:  # neither, or ambiguously both -- don't guess
            continue
        target = nerfed if is_nerf else buffed
        for hero in _find_heroes(block["heading"] + " " + block["text"], roster):
            if hero not in target:
                target.append(hero)
    # A hero in both lists had its sections parsed ambiguously; drop it rather than mislabel.
    both = set(nerfed) & set(buffed)
    return [h for h in nerfed if h not in both], [h for h in buffed if h not in both]


def _parse_date(text: str) -> str | None:
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            return pd.to_datetime(m.group(1)).date().isoformat()
        except Exception:
            continue
    return None


def fetch_index(channel: str = "original", headless: bool = True, timeout_ms: int = 30_000) -> list[dict]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(INDEX_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            result = page.evaluate(_INDEX_JS)
        finally:
            browser.close()

    patches = [p for p in result["patches"] if p["channel"] == channel]
    if not patches:
        raise RuntimeError(
            f"No {channel!r} patch links found at {INDEX_URL} (page had {result['bodyLen']} chars of text, "
            f"{result['count']} patch links across all channels). The page layout probably changed -- "
            "rerun with --dump-html and check the selectors in _INDEX_JS."
        )
    return patches


def fetch_patch(version: str, channel: str = "original", headless: bool = True,
                timeout_ms: int = 30_000, dump_html: str | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    url = PATCH_URL.format(channel=channel, version=version)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(1500)
            result = page.evaluate(_PATCH_JS)
            if dump_html:
                os.makedirs(os.path.dirname(dump_html) or ".", exist_ok=True)
                with open(dump_html, "w", encoding="utf-8") as f:
                    f.write(page.content())
                print(f"Dumped page HTML -> {dump_html}")
        finally:
            browser.close()

    roster = known_heroes()
    nerfed, buffed = _classify_blocks(result["blocks"], roster)
    mentioned = _find_heroes(result["body"], roster)
    unmatched = [h for h in mentioned if h not in nerfed and h not in buffed]

    return {
        "version": version,
        "channel": channel,
        "url": url,
        "release_date": _parse_date(result["body"]),
        "heroes_nerfed": nerfed,
        "heroes_buffed": buffed,
        "mentioned_unclassified": unmatched,
        "title": result["title"],
    }


def load_existing(patches_csv: str = PATCHES_DEFAULT) -> pd.DataFrame:
    if os.path.exists(patches_csv) and os.path.getsize(patches_csv) > 0:
        return pd.read_csv(patches_csv, dtype=str).fillna("")
    return pd.DataFrame(columns=PATCHES_COLUMNS)


def apply_rows(rows: list[dict], patches_csv: str = PATCHES_DEFAULT, force: bool = False) -> int:
    """Append parsed rows to patches.csv. Existing patch_ids are left alone unless force."""
    existing = load_existing(patches_csv)
    have = set(existing["patch_id"]) if len(existing) else set()

    new_rows = []
    for row in rows:
        if row["patch_id"] in have and not force:
            print(f"  skip {row['patch_id']}: already in {patches_csv} (use --force to replace)")
            continue
        if row["patch_id"] in have:
            existing = existing[existing["patch_id"] != row["patch_id"]]
        new_rows.append(row)

    if not new_rows:
        return 0

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.sort_values("release_date").reset_index(drop=True)
    os.makedirs(os.path.dirname(patches_csv) or ".", exist_ok=True)
    combined[PATCHES_COLUMNS].to_csv(patches_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return len(new_rows)


def sync(
    channel: str = "original",
    limit: int = 8,
    only: list[str] | None = None,
    apply: bool = False,
    force: bool = False,
    patches_csv: str = PATCHES_DEFAULT,
    headless: bool = True,
    dump_html: str | None = None,
) -> list[dict]:
    versions = only or [p["version"] for p in fetch_index(channel=channel, headless=headless)[:limit]]
    print(f"Reading {len(versions)} {channel} patch page(s): {', '.join(versions)}\n")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    for version in versions:
        try:
            info = fetch_patch(version, channel=channel, headless=headless, dump_html=dump_html)
        except Exception as exc:
            print(f"  {version}: FAILED ({exc})", file=sys.stderr)
            continue

        if not info["release_date"]:
            print(f"  {version}: no release date found on the page -- skipping "
                  f"(record it by hand with scripts/scraper.py labeled)", file=sys.stderr)
            continue
        if not info["heroes_nerfed"] and not info["heroes_buffed"]:
            print(f"  {version}: no hero changes parsed -- skipping. "
                  f"({len(info['mentioned_unclassified'])} hero name(s) appear on the page but none sat "
                  f"under a buff/nerf heading.)", file=sys.stderr)
            continue

        print(f"  {version}  released {info['release_date']}")
        print(f"      nerfed : {'|'.join(info['heroes_nerfed']) or '(none)'}")
        print(f"      buffed : {'|'.join(info['heroes_buffed']) or '(none)'}")
        if info["mentioned_unclassified"]:
            print(f"      unclear: {'|'.join(info['mentioned_unclassified'][:12])}"
                  f"{' ...' if len(info['mentioned_unclassified']) > 12 else ''}")

        rows.append({
            "patch_id": version,
            "release_date": info["release_date"],
            "heroes_nerfed": "|".join(info["heroes_nerfed"]),
            "heroes_buffed": "|".join(info["heroes_buffed"]),
            "notes": f"parsed from {info['url']} at {fetched_at}",
        })

    if not rows:
        print("\nNothing parsed. Record the patch by hand instead (menu option 4) -- that path is "
              "always reliable.", file=sys.stderr)
        return rows

    if apply:
        added = apply_rows(rows, patches_csv=patches_csv, force=force)
        print(f"\nAdded {added} row(s) -> {patches_csv}")
        if added:
            print("Next: rebuild the datasets (menu option 7), then retrain (option 8).")
    else:
        print(f"\nDRY RUN -- {patches_csv} unchanged. Re-run with --apply to write these "
              f"{len(rows)} row(s), after checking the lists above against the patch notes.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--channel", default="original", choices=list(CHANNELS),
                    help="original = live server (default), advanced = Advance Server (runs ahead of live).")
    ap.add_argument("--limit", type=int, default=8, help="How many of the newest patches to read.")
    ap.add_argument("--only", help="Comma-separated versions to read instead of the newest N.")
    ap.add_argument("--apply", action="store_true", help="Actually write to data/patches.csv.")
    ap.add_argument("--force", action="store_true", help="Replace rows whose patch_id already exists.")
    ap.add_argument("--patches-csv", default=PATCHES_DEFAULT)
    ap.add_argument("--dump-html", help="Save each fetched page's HTML here for debugging.")
    ap.add_argument("--no-headless", action="store_true")
    args = ap.parse_args()

    only = [v.strip() for v in args.only.split(",")] if args.only else None
    try:
        sync(channel=args.channel, limit=args.limit, only=only, apply=args.apply, force=args.force,
             patches_csv=args.patches_csv, headless=not args.no_headless, dump_html=args.dump_html)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
