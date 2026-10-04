"""Scrape live MLBB hero win/pick/ban rates from mlbbhub.com/statistics via headless Playwright.

Moved from the original mlbb_scraper.py prototype -- fetch_stats() and the extraction JS are
unchanged from the version already verified against the live site. Only the surrounding
orchestration changed, to fit this repo's architecture:

- `daily`: writes an unlabeled raw snapshot to data/daily/<UTC date>.csv. data/daily/ is
  append-only -- this NEVER overwrites or deletes an existing file for that date, and there is no
  --force escape hatch for it (unlike the old prototype). This is the only thing the scheduled
  cron trigger ever runs.
- `snapshot`: the unlimited-collection mode. Scrapes one or many rank brackets in a single run
  and writes each to data/archive/<rank>/<UTC timestamp>.csv -- any number of snapshots per day,
  any rank, kept forever. With --every it keeps collecting on an interval until stopped. For the
  Mythic bracket it also mirrors to data/daily/<date>.csv when that file doesn't exist yet, so
  the legacy layout and everything reading it stays correct. Still append-only: a file that has
  landed is never rewritten.
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

Roster-size check: a count BELOW MIN_HERO_COUNT still fails hard -- that is the signature of a
half-rendered page or a broken selector, and writing it would poison the archive. A count ABOVE
it that merely differs from EXPECTED_HERO_COUNT now warns instead of failing, because Moonton
shipping hero 134 should not silently stop daily collection for however long it takes someone to
notice and bump a constant. Override the floor with MLBB_MIN_HERO_COUNT if you need to.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store  # noqa: E402

if TYPE_CHECKING:  # Playwright is imported lazily inside fetch_stats() -- see below.
    from playwright.sync_api import Page

STATS_URL = "https://mlbbhub.com/statistics"
EXPECTED_HERO_COUNT = 133  # informational: a different count warns, it no longer halts collection
MIN_HERO_COUNT = int(os.environ.get("MLBB_MIN_HERO_COUNT", "125"))  # below this the scrape is broken, not new
WIN_RATE_MEAN_BOUNDS = (48.0, 52.0)  # every match has one winner and one loser -> average should hover ~50%

DAILY_DIR_DEFAULT = "data/daily"
ARCHIVE_DIR_DEFAULT = store.ARCHIVE_DIR
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
assert tuple(RANK_LABELS) == store.RANKS, "RANK_LABELS must stay in sync with store.RANKS"

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


# Dumps whatever the page currently offers as a rank control. This exists because when the
# filter selector broke, the scheduled job failed with nothing but "button 'Mythic' not visible"
# -- true, but useless: it never said what the page DID have. Every failure now ends with this,
# so one log read is enough to write the new selector.
_PROBE_JS = r"""
() => {
  const LABELS = ['Epic','Legend','Mythic','Mythical Honor','Mythical Glory'];
  const squash = t => (t || '').replace(/\s+/g, ' ').trim();
  const describe = el => ({
    tag: el.tagName.toLowerCase(),
    role: el.getAttribute('role'),
    text: squash(el.innerText).slice(0, 40),
    aria: el.getAttribute('aria-label'),
    cls: squash((el.className || '').toString()).slice(0, 90),
    data: Object.keys(el.dataset || {}).slice(0, 6),
  });

  const buttons = Array.from(document.querySelectorAll('button')).map(describe);
  const roles = Array.from(document.querySelectorAll('[role="tab"],[role="option"],[role="menuitem"],[role="radio"]')).map(describe);
  const selects = Array.from(document.querySelectorAll('select')).map(s => ({
    cls: squash((s.className || '').toString()).slice(0, 90),
    options: Array.from(s.options).map(o => squash(o.textContent)).slice(0, 12),
  }));

  // Anything whose own text is exactly a rank name -- the likeliest new control, whatever its tag.
  const exact = [];
  for (const el of document.querySelectorAll('*')) {
    if (el.children.length) continue;
    const t = squash(el.textContent);
    if (LABELS.includes(t)) exact.push({ ...describe(el), parentTag: el.parentElement ? el.parentElement.tagName.toLowerCase() : null,
                                          parentCls: el.parentElement ? squash((el.parentElement.className || '').toString()).slice(0, 90) : null });
  }

  const container = document.querySelector('.md\\:hidden.space-y-\\[2px\\]');
  // Fallback: any element with many same-shaped children looks like the stats table.
  const bigLists = Array.from(document.querySelectorAll('div,ul,tbody'))
    .filter(e => e.children.length >= 50)
    .map(e => ({ cls: squash((e.className || '').toString()).slice(0, 90), children: e.children.length, tag: e.tagName.toLowerCase() }))
    .slice(0, 8);

  return {
    title: document.title, url: location.href, bodyLen: document.body.innerText.length,
    buttons: buttons.slice(0, 40), roles: roles.slice(0, 40), selects,
    exactRankText: exact.slice(0, 20),
    knownContainerFound: !!container,
    knownContainerChildren: container ? container.children.length : 0,
    bigLists,
  };
}
"""


def _format_probe(probe: dict) -> str:
    lines = [
        f"  page title : {probe['title']!r}",
        f"  url        : {probe['url']}",
        f"  body text  : {probe['bodyLen']} chars",
        f"  stats container ('.md:hidden.space-y-[2px]'): "
        f"{'FOUND, ' + str(probe['knownContainerChildren']) + ' rows' if probe['knownContainerFound'] else 'NOT FOUND'}",
    ]
    if probe["exactRankText"]:
        lines.append("  elements whose text is exactly a rank name:")
        for e in probe["exactRankText"]:
            lines.append(f"    <{e['tag']} role={e['role']!r} class={e['cls']!r}> {e['text']!r} "
                         f"(parent <{e['parentTag']} class={e['parentCls']!r}>)")
    else:
        lines.append("  NO element has a rank name as its exact text -- the labels themselves changed.")
    if probe["selects"]:
        lines.append("  <select> elements:")
        for sel in probe["selects"]:
            lines.append(f"    class={sel['cls']!r} options={sel['options']}")
    lines.append(f"  buttons on page ({len(probe['buttons'])} shown):")
    for b in probe["buttons"][:18]:
        lines.append(f"    text={b['text']!r} aria={b['aria']!r} class={b['cls']!r}")
    if probe["roles"]:
        lines.append(f"  tab/option/menuitem roles ({len(probe['roles'])}):")
        for r in probe["roles"][:12]:
            lines.append(f"    <{r['tag']} role={r['role']!r}> {r['text']!r} class={r['cls']!r}")
    if probe["bigLists"]:
        lines.append("  candidate stats containers (50+ children):")
        for c in probe["bigLists"]:
            lines.append(f"    <{c['tag']} class={c['cls']!r}> {c['children']} children")
    return "\n".join(lines)


def _select_rank_listbox(page: "Page", rank_label: str, timeout_ms: int) -> bool:
    """Select a rank from the redesigned page's listbox dropdown.

    The 2026 redesign replaced the row of rank buttons with a combobox:

        <button aria-haspopup="listbox" aria-controls="...-list">
          <span class="sr-only">Rank</span><span>All Ranks</span>
        <ul role="listbox" aria-label="Rank">
          <li role="option" aria-selected="true"><span>All Ranks</span>
          <li role="option"><span>Epic</span> ... <span>Mythical Glory</span>

    Two consequences beyond the selector change: the control has to be *opened* before any rank
    is clickable (which is why every single-step strategy missed it), and the page now defaults
    to "All Ranks" -- so not selecting a rank silently yields blended data rather than Mythic.
    """
    trigger = None
    for candidate in (
        'button[aria-haspopup="listbox"]:has(span.sr-only:text-is("Rank"))',
        'button[aria-haspopup="listbox"]:has-text("Rank")',
        '[role="combobox"]:has-text("Rank")',
    ):
        try:
            locator = page.locator(candidate).first
            locator.wait_for(state="visible", timeout=max(2_000, timeout_ms // 6))
            trigger = locator
            break
        except Exception:
            continue
    if trigger is None:
        return False

    # Matching the option is where this is easy to get silently wrong. The option's text sits in
    # a nested <span>, so Playwright's :text-is() binds to the span rather than the <li>; and
    # :has-text() is a substring match, so "Mythic" also matches "Mythical Honor" and "Mythical
    # Glory" -- which would scrape a different bracket and label it mythic. Exact accessible-name
    # matching is the only form verified to select Mythic and nothing else.
    for build_option in (
        lambda: page.get_by_role("option", name=rank_label, exact=True),
        lambda: page.locator(f'[role="option"]:has(span:text-is("{rank_label}"))'),
    ):
        try:
            trigger.click()
            option = build_option().first
            option.wait_for(state="visible", timeout=max(2_000, timeout_ms // 6))
            option.click()
            page.wait_for_timeout(400)
            return True
        except Exception:
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            continue

    # Leave the dropdown closed so the legacy strategies start from a clean page.
    try:
        page.keyboard.press("Escape")
    except Exception:
        pass
    return False


# The redesigned stats table. Resolves columns by their header text rather than by position,
# so adding a column (the redesign added TIER and TREND) doesn't shift the numbers being read.
_EXTRACT_TABLE_JS = r"""
() => {
  const squash = t => (t || '').replace(/\s+/g, ' ').trim();
  const ROLES = ['tank', 'fighter', 'assassin', 'mage', 'marksman', 'support'];

  let best = null;
  for (const t of document.querySelectorAll('table')) {
    const rows = Array.from(t.querySelectorAll('tbody tr'));
    if (!best || rows.length > best.rows.length) best = { table: t, rows };
  }
  if (!best || !best.rows.length) {
    return { error: 'no <table> with tbody rows found', tables: document.querySelectorAll('table').length };
  }

  const heads = Array.from(best.table.querySelectorAll('thead th, thead td'))
    .map(c => squash(c.innerText).toUpperCase());
  const col = needle => heads.findIndex(h => h.includes(needle));
  const iHero = col('HERO'), iWin = col('WIN'), iPick = col('PICK'), iBan = col('BAN');
  if (iWin < 0 || iPick < 0 || iBan < 0) {
    return { error: 'could not find WIN/PICK/BAN columns by header text', heads };
  }

  // "53.2%" / "53.2" / "+0.4" -> first number. Rejects a cell that holds no number at all.
  const num = text => { const m = squash(text).match(/-?\d+(?:\.\d+)?/); return m ? m[0] : null; };

  const data = best.rows.map(tr => {
    const cells = Array.from(tr.children);
    const at = i => (i >= 0 && cells[i]) ? squash(cells[i].innerText) : '';
    const heroCell = cells[iHero >= 0 ? iHero : 0];

    let name = null;
    if (heroCell) {
      const link = heroCell.querySelector('a');
      name = squash((link || heroCell).innerText).split(' ').length > 6
        ? squash((link || heroCell).innerText).slice(0, 40)
        : squash((link || heroCell).innerText);
    }

    // Role is no longer its own column. Look for it in an image alt, then in the hero cell's
    // text, then in any cell whose whole text is a role name.
    let role = null;
    if (heroCell) {
      for (const img of heroCell.querySelectorAll('img[alt]')) {
        const cand = img.alt.toLowerCase().replace(/ (hero )?icon$/, '').trim();
        if (ROLES.includes(cand)) { role = cand; break; }
      }
      if (!role) {
        const lower = squash(heroCell.innerText).toLowerCase();
        role = ROLES.find(r => lower.includes(r)) || null;
      }
    }
    if (!role) {
      for (const c of cells) {
        const lower = squash(c.innerText).toLowerCase();
        if (ROLES.includes(lower)) { role = lower; break; }
      }
    }

    return { name, role, win: num(at(iWin)), pick: num(at(iPick)), ban: num(at(iBan)) };
  });

  return { count: data.length, data, heads, layout: 'table' };
}
"""


def _roles_from_archive() -> dict[str, str]:
    """hero -> role from the most recent snapshot that has roles. Empty dict if unavailable."""
    try:
        long_df = store.load_snapshots()
        if long_df.empty:
            return {}
        latest = long_df[long_df["ts"] == long_df["ts"].max()]
        roles = latest.loc[latest["role"].notna() & (latest["role"] != "unknown"), ["hero", "role"]]
        return dict(zip(roles["hero"], roles["role"]))
    except Exception:
        return {}


def _click_rank_filter(page: "Page", rank_label: str, timeout_ms: int) -> None:
    """Select the rank-bracket filter, trying several markup shapes before giving up.

    The original single strategy (role=button with an exact accessible name) broke when the site
    changed its filter markup, and the scheduled job then failed identically every day for three
    weeks. Each strategy below is a different plausible shape for the same control; the first one
    that becomes visible wins. If all of them miss, the error carries a dump of what the page
    actually offers, so the fix is a log read rather than a guessing game.
    """
    # The current site shape first: a listbox that must be opened before its options exist.
    if _select_rank_listbox(page, rank_label, timeout_ms):
        return

    per_try = max(2_000, timeout_ms // 5)
    attempts = [
        ("role=button exact", lambda: page.get_by_role("button", name=rank_label, exact=True)),
        ("role=tab exact", lambda: page.get_by_role("tab", name=rank_label, exact=True)),
        ("role=option exact", lambda: page.get_by_role("option", name=rank_label, exact=True)),
        ("role=radio exact", lambda: page.get_by_role("radio", name=rank_label, exact=True)),
        ("aria-label", lambda: page.locator(f'[aria-label="{rank_label}"]')),
        ("exact text", lambda: page.get_by_text(rank_label, exact=True)),
    ]

    errors = []
    for name, build in attempts:
        try:
            locator = build().first
            locator.wait_for(state="visible", timeout=per_try)
            locator.click()
            print(f"NOTE: rank filter {rank_label!r} matched via the legacy '{name}' strategy, not "
                  f"the listbox dropdown -- the page markup changed again. Worth revisiting "
                  f"_select_rank_listbox.", file=sys.stderr)
            return
        except Exception as exc:  # locator miss, not visible, intercepted click, detached node
            errors.append(f"{name}: {type(exc).__name__}")

    # A <select> is a different interaction, so it gets its own attempt rather than a click.
    try:
        select = page.locator("select").first
        select.wait_for(state="visible", timeout=per_try)
        select.select_option(label=rank_label)
        print(f"NOTE: rank filter {rank_label!r} set via a <select>, not a button.", file=sys.stderr)
        return
    except Exception as exc:
        errors.append(f"select_option: {type(exc).__name__}")

    try:
        probe = _format_probe(page.evaluate(_PROBE_JS))
    except Exception as exc:
        probe = f"  (page probe also failed: {exc})"

    raise RuntimeError(
        f"Could not find the {rank_label!r} rank filter. Tried: {', '.join(errors)}.\n"
        f"What the page actually offers right now:\n{probe}\n"
        f"Update _click_rank_filter / _EXTRACT_JS in {os.path.basename(__file__)} to match, then "
        f"re-run. Nothing was written."
    )



# Second-stage probe: the table's real shape, and what the rank dropdown contains once opened.
# The first probe says "the markup changed"; these two say "and here is exactly what to parse".
_TABLE_JS = r"""
() => {
  const squash = t => (t || '').replace(/\s+/g, ' ').trim();
  return Array.from(document.querySelectorAll('table')).map(t => {
    const head = Array.from(t.querySelectorAll('thead th, thead td')).map(c => squash(c.innerText));
    const bodyRows = Array.from(t.querySelectorAll('tbody tr'));
    return {
      cls: squash((t.className || '').toString()).slice(0, 80),
      rows: bodyRows.length,
      head,
      sample: bodyRows.slice(0, 3).map(tr => ({
        cells: Array.from(tr.children).map(c => squash(c.innerText)),
        imgs: Array.from(tr.querySelectorAll('img[alt]')).map(i => i.alt),
        html: tr.outerHTML.slice(0, 900),
      })),
    };
  });
}
"""

_MENU_JS = r"""
() => {
  const squash = t => (t || '').replace(/\s+/g, ' ').trim();
  const visible = el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  // Only rank-relevant text. The previous version listed every visible button and hero link,
  // which buried the answer under 60 lines of noise.
  const KEY = /epic|legend|mythic|mythical|honor|glory|all ranks/i;
  const items = [];
  const sel = '[role="option"],[role="menuitem"],[role="menuitemradio"],[role="radio"],li,button,a,div[data-value],span';
  for (const el of document.querySelectorAll(sel)) {
    if (!visible(el) || el.children.length > 2) continue;
    const t = squash(el.innerText);
    if (!t || t.length > 30 || !KEY.test(t)) continue;
    items.push({
      tag: el.tagName.toLowerCase(), role: el.getAttribute('role'), text: t,
      cls: squash((el.className || '').toString()).slice(0, 60),
      data: Object.keys(el.dataset || {}).slice(0, 5),
    });
  }
  // Whatever container is currently open -- Radix/Headless UI style popovers live in a portal.
  const open = Array.from(document.querySelectorAll(
      '[role="listbox"],[role="menu"],[role="dialog"],[data-state="open"],[aria-expanded="true"]'))
    .map(el => ({ tag: el.tagName.toLowerCase(), role: el.getAttribute('role'),
                  state: el.getAttribute('data-state'),
                  text: squash(el.innerText).slice(0, 200),
                  html: el.outerHTML.slice(0, 600) }))
    .slice(0, 6);
  return { items: items.slice(0, 25), open };
}
"""

# The redesigned page puts each filter behind a dropdown whose trigger shows "<name> <current
# value>" -- e.g. "Rank All Ranks". Opening it is a separate step from choosing a value.
_FILTER_TRIGGERS = (
    'button:has-text("{name}")',
    '[aria-label*="{name}"]',
    '[role="combobox"]:has-text("{name}")',
)


def _open_filter_menu(page: "Page", name: str, timeout_ms: int = 4_000) -> bool:
    """Click the dropdown trigger for a named filter (Rank / Window / Role / Lane)."""
    for pattern in _FILTER_TRIGGERS:
        try:
            trigger = page.locator(pattern.format(name=name)).first
            trigger.wait_for(state="visible", timeout=timeout_ms)
            trigger.click()
            page.wait_for_timeout(700)
            return True
        except Exception:
            continue
    return False


def diagnose(headless: bool = True, timeout_ms: int = 30_000, dump_html: str | None = None) -> dict:
    """Open the stats page and report what it offers, without writing anything.

    This is the thing to run when collection has been failing: it answers "what does the page
    look like now" in one go, including whether the stats table container still matches.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install -r requirements.txt && "
            "playwright install --with-deps chromium` (menu option 16)."
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(STATS_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(2500)
            probe = page.evaluate(_PROBE_JS)
            probe["tables"] = page.evaluate(_TABLE_JS)
            probe["rank_menu_opened"] = _open_filter_menu(page, "Rank")
            probe["rank_menu"] = page.evaluate(_MENU_JS) if probe["rank_menu_opened"] else []
            if dump_html:
                os.makedirs(os.path.dirname(dump_html) or ".", exist_ok=True)
                with open(dump_html, "w", encoding="utf-8") as f:
                    f.write(page.content())
                print(f"Wrote page HTML -> {dump_html}")
        finally:
            browser.close()

    # Deliberate ordering: the two things needed to write new selectors (the table's real shape
    # and the rank dropdown's contents) print LAST, because a CI log is usually read by its tail.
    print(f"Diagnostic probe of {STATS_URL}")
    print(f"  title={probe['title']!r}  bodyLen={probe['bodyLen']}  "
          f"old container={'FOUND' if probe['knownContainerFound'] else 'GONE'}")
    filters = [b for b in probe["buttons"] if any(
        k in (b["text"] or "") for k in ("Rank", "Window", "Role", "Lane"))]
    print(f"  filter-looking buttons: {[b['text'] for b in filters]}")

    print("\n=== TABLES ===")
    for i, table in enumerate(probe.get("tables", [])):
        print(f"  <table {i}> class={table['cls']!r}  {table['rows']} body row(s)")
        print(f"    headers : {table['head']}")
        for j, row in enumerate(table["sample"][:2]):
            print(f"    row{j} cells: {row['cells']}")
            print(f"    row{j} alts : {row['imgs']}")
        if table["sample"]:
            print(f"    row0 html : {table['sample'][0]['html'][:600]}")

    print("\n=== RANK DROPDOWN ===")
    menu = probe.get("rank_menu") or {}
    print(f"  trigger clicked: {probe.get('rank_menu_opened')}")
    for item in (menu.get("items") or []):
        print(f"    <{item['tag']} role={item['role']!r}> {item['text']!r} "
              f"class={item['cls']!r} data={item['data']}")
    if not (menu.get("items") or []):
        print("    no rank-named items visible after the click")
    for o in (menu.get("open") or []):
        print(f"    OPEN <{o['tag']} role={o['role']!r} state={o['state']!r}> text={o['text']!r}")
        print(f"         html={o['html'][:400]}")
    return probe


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

    # Imported here, not at module scope, so recording a patch by hand and reading the archive
    # work on a machine that has never installed Playwright or its browsers.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed, so stats cannot be scraped. Run "
            "`pip install -r requirements.txt && playwright install --with-deps chromium` "
            "(menu option 16)."
        ) from exc

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

            # Current layout first, then the pre-redesign container, so an old mirror or a
            # rollback still scrapes rather than failing outright.
            result = page.evaluate(_EXTRACT_TABLE_JS)
            if "error" in result:
                table_error = result
                result = page.evaluate(_EXTRACT_JS)
                if "error" in result:
                    result = {"error": f"table layout: {table_error}; legacy layout: {result}"}
                else:
                    print("NOTE: scraped via the pre-redesign container, not the <table> layout.",
                          file=sys.stderr)
        finally:
            browser.close()

    if "error" in result:
        raise RuntimeError(f"Could not read the stats table: {result}")

    df = pd.DataFrame(result["data"])

    if df.empty:
        raise RuntimeError("Scrape returned zero hero rows -- page structure may have changed.")
    if len(df) < MIN_HERO_COUNT:
        raise RuntimeError(
            f"Only {len(df)} heroes scraped, below the floor of {MIN_HERO_COUNT}. That is a "
            "half-rendered page or a changed selector, not a roster change -- refusing to write. "
            "(Override the floor with MLBB_MIN_HERO_COUNT if the roster really did shrink.)"
        )
    if len(df) != EXPECTED_HERO_COUNT:
        print(
            f"NOTE: scraped {len(df)} heroes, expected {EXPECTED_HERO_COUNT} -- probably a new hero. "
            f"Collection continues; bump EXPECTED_HERO_COUNT in {os.path.basename(__file__)} to silence this.",
            file=sys.stderr,
        )
    # Role moved out of its own column in the redesign, so it can come back empty even when
    # every rate parsed fine. Rather than discard a good scrape over a field that is constant
    # per hero, fill it from the roles already in the archive; only a genuinely new hero stays
    # unknown. The rates -- the actual measurements -- are never filled in like this.
    if df["role"].isnull().any():
        filled = _roles_from_archive()
        if filled:
            df["role"] = df["role"].fillna(df["name"].map(filled))
        still_missing = df["role"].isnull()
        if still_missing.any():
            print(
                f"WARNING: no role for {sorted(df.loc[still_missing, 'name'])} -- not on the page "
                "and not in the archive (new hero?). Recording them as 'unknown'; role-relative "
                "features will treat them as their own cohort until it is corrected.",
                file=sys.stderr,
            )
            df["role"] = df["role"].fillna("unknown")

    if df[["name", "win", "pick", "ban"]].isnull().any().any():
        bad = df[df[["name", "win", "pick", "ban"]].isnull().any(axis=1)]
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


def _write_snapshot_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out = df.copy()
    out.insert(0, "patch_date", "")
    out.insert(0, "patch_id", "")
    out["nerfed_next"] = ""
    out[store.SNAPSHOT_COLUMNS].to_csv(out_path, index=False)


def save_snapshot(
    ranks: list[str] | tuple[str, ...] = (store.DEFAULT_RANK,),
    archive_dir: str = ARCHIVE_DIR_DEFAULT,
    daily_dir: str = DAILY_DIR_DEFAULT,
    headless: bool = True,
    mirror_daily: bool = True,
    once_per_day: bool = False,
) -> list[str]:
    """Scrape each requested rank and archive it. Returns the paths written.

    One archive file per rank per run, stamped to the second, so there is no per-day cap on how
    often this can run -- that is the point of this mode. Mythic additionally mirrors into the
    legacy data/daily/<date>.csv when that day has no file yet, which keeps promote.py,
    train.csv and the existing archive readable exactly as before.

    A rank that fails to scrape does not abort the others: its error is reported and the run
    continues, so one flaky bracket never costs a whole day of collection across the rest.
    """
    written: list[str] = []
    failures: list[tuple[str, str]] = []
    when = store.utc_now()
    today = when.date().isoformat()

    for rank in ranks:
        if once_per_day:
            already = [r for r in store.iter_refs(ranks=(rank,), include_legacy=False) if r.date == today]
            if already:
                print(f"[{rank}] already archived today ({already[-1].path}), skipping (--once-per-day).")
                continue
        try:
            df = fetch_stats(rank=rank, headless=headless)
        except (RuntimeError, ValueError) as exc:
            print(f"[{rank}] ERROR: {exc}", file=sys.stderr)
            failures.append((rank, str(exc)))
            continue

        out_path = store.archive_path(rank, when=when, archive_dir=archive_dir)
        if os.path.exists(out_path):
            print(f"[{rank}] {out_path} already exists, leaving it untouched (append-only).")
        else:
            _write_snapshot_csv(df, out_path)
            written.append(out_path)
            print(f"[{rank}] wrote {len(df)} rows -> {out_path}")

        if mirror_daily and rank == store.DEFAULT_RANK:
            legacy = os.path.join(daily_dir, f"{today}.csv")
            if os.path.exists(legacy):
                print(f"[{rank}] {legacy} already exists -- data/daily/ is append-only, leaving it untouched.")
            else:
                _write_snapshot_csv(df, legacy)
                written.append(legacy)
                print(f"[{rank}] mirrored -> {legacy}")

    if failures and not written:
        raise RuntimeError(
            "Every rank failed: " + "; ".join(f"{r}: {e.splitlines()[0]}" for r, e in failures)
        )
    return written


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

    diag_parser = subparsers.add_parser(
        "diagnose",
        help="Open the stats page and report what rank controls / containers it has. Writes no data.",
    )
    diag_parser.add_argument("--dump-html", help="Also save the page HTML here.")
    diag_parser.add_argument("--no-headless", action="store_true")

    snap_parser = subparsers.add_parser(
        "snapshot",
        help="Unlimited collection: archive one or many ranks to data/archive/, optionally on a repeating interval.",
    )
    snap_parser.add_argument(
        "--ranks", default=store.DEFAULT_RANK,
        help='Comma-separated ranks, or "all" for every bracket. Default: mythic.',
    )
    snap_parser.add_argument("--archive-dir", default=ARCHIVE_DIR_DEFAULT)
    snap_parser.add_argument("--daily-dir", default=DAILY_DIR_DEFAULT)
    snap_parser.add_argument(
        "--no-mirror-daily", action="store_true",
        help="Don't also write the legacy data/daily/<date>.csv for the Mythic bracket.",
    )
    snap_parser.add_argument(
        "--once-per-day", action="store_true",
        help="Skip a rank that already has an archive snapshot today (for idempotent cron runs).",
    )
    snap_parser.add_argument(
        "--every", type=float, metavar="MINUTES",
        help="Keep collecting every MINUTES minutes instead of exiting after one pass.",
    )
    snap_parser.add_argument(
        "--count", type=int, default=0, metavar="N",
        help="With --every: stop after N passes. 0 (default) means run until interrupted.",
    )
    snap_parser.add_argument("--no-headless", action="store_true", help="Run with a visible browser (local debugging only).")

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
        if args.mode == "diagnose":
            diagnose(headless=not args.no_headless, dump_html=args.dump_html)
        elif args.mode == "daily":
            save_daily_snapshot(output_dir=args.output_dir, rank=args.rank, headless=not args.no_headless)
        elif args.mode == "snapshot":
            ranks = list(store.RANKS) if args.ranks == "all" else [r.strip() for r in args.ranks.split(",") if r.strip()]
            unknown = [r for r in ranks if r not in RANK_LABELS]
            if unknown:
                raise RuntimeError(f"Unknown rank(s) {unknown}; choose from {list(RANK_LABELS)} or 'all'.")

            passes = 0
            while True:
                passes += 1
                if args.every:
                    print(f"--- pass {passes}{f' of {args.count}' if args.count else ''} "
                          f"at {store.utc_now().strftime('%Y-%m-%d %H:%M:%S')} UTC ---")
                save_snapshot(
                    ranks=ranks,
                    archive_dir=args.archive_dir,
                    daily_dir=args.daily_dir,
                    headless=not args.no_headless,
                    mirror_daily=not args.no_mirror_daily,
                    once_per_day=args.once_per_day,
                )
                if not args.every or (args.count and passes >= args.count):
                    break
                print(f"Sleeping {args.every:g} min; Ctrl-C to stop.")
                try:
                    time.sleep(args.every * 60)
                except KeyboardInterrupt:
                    print("\nStopped. Everything already collected is on disk.")
                    break
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
