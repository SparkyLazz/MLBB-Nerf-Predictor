#!/usr/bin/env python3
"""MLBB Nerf Predictor -- interactive menu.

    python mlbb.py            # menu
    python mlbb.py --run 10   # jump straight to one option and exit
    python mlbb.py --list     # print the options and exit

Everything here is a front end over the scripts in scripts/, which all still work on their own:
nothing moved behind the menu. The status block at the top is read fresh from disk each time the
menu is drawn, so it always reflects the real archive rather than what happened earlier in the
session.

Heavy and optional imports (pandas, scikit-learn, Playwright) are loaded inside the individual
actions, so the menu itself opens instantly and still opens on a machine where nothing is
installed yet -- option 16 is how you fix that from in here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from typing import Callable

REPO = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO, "scripts")

# Every script in this repo assumes the repo root is the working directory.
os.chdir(REPO)
sys.path.insert(0, SCRIPTS)

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def cyan(t: str) -> str:
    return c(t, "36")


def green(t: str) -> str:
    return c(t, "32")


def yellow(t: str) -> str:
    return c(t, "33")


def red(t: str) -> str:
    return c(t, "31")


# --------------------------------------------------------------------------------------
# input helpers
# --------------------------------------------------------------------------------------

def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        print()
        return default
    return value or default


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print(red(f"  '{raw}' is not a whole number."))


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print(red(f"  '{raw}' is not a number."))


def confirm(prompt: str, default: bool = False) -> bool:
    raw = ask(f"{prompt} (y/n)", "y" if default else "n").lower()
    return raw.startswith("y")


def choose(prompt: str, options: list[str], default: str | None = None) -> str:
    """Pick from a list by number or by typing the value."""
    default = default or options[0]
    print(f"  {prompt}")
    for i, opt in enumerate(options, start=1):
        marker = dim(" (default)") if opt == default else ""
        print(f"    {i}) {opt}{marker}")
    while True:
        raw = ask("  choice", default)
        if raw in options:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(red(f"  pick 1-{len(options)} or type one of: {', '.join(options)}"))


def ask_date(prompt: str, default: str = "") -> str:
    """A YYYY-MM-DD date, re-prompted until it parses. Empty is allowed when a default is empty."""
    import pandas as pd
    while True:
        raw = ask(prompt, default)
        if not raw:
            return ""
        try:
            return pd.to_datetime(raw).date().isoformat()
        except Exception:
            print(red(f"  '{raw}' isn't a date I can read -- use YYYY-MM-DD."))


def ask_ranks() -> list[str]:
    import store
    print("  Which rank bracket(s)?")
    for i, rank in enumerate(store.RANKS, start=1):
        print(f"    {i}) {rank}")
    print(f"    a) all {len(store.RANKS)}")
    raw = ask("  choice", store.DEFAULT_RANK)
    if raw.lower() in ("a", "all"):
        return list(store.RANKS)
    picked = []
    for token in raw.replace(",", " ").split():
        if token.isdigit() and 1 <= int(token) <= len(store.RANKS):
            picked.append(store.RANKS[int(token) - 1])
        elif token in store.RANKS:
            picked.append(token)
    return picked or [store.DEFAULT_RANK]


# --------------------------------------------------------------------------------------
# status header
# --------------------------------------------------------------------------------------

def _trainable_windows(patches) -> int:
    """Patch windows that can actually produce training rows.

    A window needs two things, and the count people care about is where both hold: the NEXT
    patch's nerf list must be recorded (that is the label), and the window must contain at least
    one snapshot (that is the data). Counting only the first is how you end up reporting a
    labeled window that no collection ever covered -- which is exactly this repo's situation for
    patch 2.1.88, whose window closed weeks before the first snapshot was taken.
    """
    import pandas as pd

    import store
    dates = sorted({ref.date for ref in store.iter_refs()})
    if not dates or len(patches) < 2:
        return 0
    stamps = [pd.Timestamp(d) for d in dates]

    count = 0
    for i in range(len(patches) - 1):
        if not str(patches.iloc[i + 1]["heroes_nerfed"]).strip():
            continue
        start = pd.Timestamp(patches.iloc[i]["release_date"])
        end = pd.Timestamp(patches.iloc[i + 1]["release_date"])
        if any(start <= s < end for s in stamps):
            count += 1
    return count


def status_lines() -> list[str]:
    """Three lines of current state. Deliberately cheap -- no dataset build, no model load."""
    lines = []
    try:
        import store
        summary = store.summary()
        if summary["snapshots"] == 0:
            lines.append(f"  Archive  {yellow('empty')} -- nothing collected yet (start with option 1)")
        else:
            ranks = ", ".join(summary["ranks"])
            lines.append(
                f"  Archive  {green(str(summary['snapshots']) + ' snapshots')} · {summary['days']} days "
                f"({summary['first']} → {summary['last']}) · {summary['heroes']} heroes · ranks: {ranks}"
            )
    except Exception as exc:
        lines.append(f"  Archive  {red('unreadable')}: {exc}")

    try:
        import pandas as pd
        from promote import PATCHES
        if os.path.exists(PATCHES):
            patches = pd.read_csv(PATCHES, dtype=str).fillna("").sort_values("release_date").reset_index(drop=True)
            trainable = _trainable_windows(patches)
            latest = patches.iloc[-1] if len(patches) else None
            latest_txt = f"latest {latest['patch_id']} ({latest['release_date']})" if latest is not None else "none"
            flag = (green if trainable >= 2 else yellow)(f"{trainable} trainable window(s)")
            lines.append(f"  Patches  {len(patches)} recorded · {latest_txt} · {flag}")
        else:
            lines.append(f"  Patches  {yellow('no data/patches.csv')} -- record a patch with option 4")
    except Exception as exc:
        lines.append(f"  Patches  {red('unreadable')}: {exc}")

    try:
        import model as modelmod
        if os.path.exists(modelmod.MODEL_PATH):
            bundle = modelmod.load()
            if bundle:
                cv = bundle["cv"][bundle["estimator_name"]]["mean"]
                ap = cv.get("average_precision")
                score = f"AP {ap:.3f}" if ap is not None else "no CV folds"
                warn = "" if bundle["cv_reliable"] else yellow(" (few patches -- rough)")
                lines.append(
                    f"  Model    {green(bundle['estimator_name'])} trained {bundle['trained_at'][:10]} "
                    f"on {len(bundle['patch_windows'])} window(s) · {score}{warn}"
                )
            else:
                lines.append(f"  Model    {red('present but unreadable')} ({modelmod.MODEL_PATH})")
        else:
            lines.append(f"  Model    {yellow('not trained')} -- the trend engine (option 10) works without one")
    except Exception as exc:
        lines.append(f"  Model    {dim(f'unavailable ({exc})')}")

    return lines


# --------------------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------------------

def act_snapshot_pick() -> None:
    """Collect one snapshot for the rank(s) you choose."""
    import scraper
    ranks = ask_ranks()
    print()
    scraper.save_snapshot(ranks=ranks)


def act_snapshot_all() -> None:
    """Collect one snapshot for every rank bracket."""
    import scraper
    import store
    scraper.save_snapshot(ranks=list(store.RANKS))


def act_watch() -> None:
    """Keep collecting on an interval until you stop it (Ctrl-C)."""
    import scraper
    ranks = ask_ranks()
    every = ask_float("  Minutes between snapshots", 60.0)
    count = ask_int("  How many passes? (0 = until stopped)", 0)
    if every < 5:
        print(yellow("  Note: the underlying stats only refresh every few hours. Anything under "
                     "~60 min mostly archives identical numbers."))
        if not confirm("  Continue anyway?", default=False):
            return
    print(dim("\n  Ctrl-C stops it; everything already collected stays on disk.\n"))
    passes = 0
    import time
    import store
    while True:
        passes += 1
        print(f"--- pass {passes}{f' of {count}' if count else ''} at "
              f"{store.utc_now().strftime('%Y-%m-%d %H:%M:%S')} UTC ---")
        try:
            scraper.save_snapshot(ranks=ranks)
        except RuntimeError as exc:
            print(red(f"  pass failed: {exc}"))
        if count and passes >= count:
            break
        try:
            time.sleep(every * 60)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break


def act_record_patch() -> None:
    """Type in a patch by hand -- the always-reliable path for labeling."""
    import scraper
    print(dim("  From the official patch notes. Hero lists are pipe-separated: Baxia|Akai\n"))
    patch_id = ask("  Patch version (e.g. 2.2.16)")
    if not patch_id:
        print(red("  Cancelled -- a patch version is required."))
        return
    patch_date = ask_date("  Release date (YYYY-MM-DD)")
    if not patch_date:
        print(red("  Cancelled -- a release date is required."))
        return
    nerfed = ask("  Heroes NERFED (blank to fill in later)")
    buffed = ask("  Heroes BUFFED (blank to fill in later)")
    notes = ask("  Notes")

    unknown = _unknown_heroes(nerfed) + _unknown_heroes(buffed)
    if unknown:
        print(yellow(f"  Not hero names the scraper has seen: {', '.join(unknown)}"))
        print(yellow("  A misspelled name silently becomes a wrong label, so check it."))
        if not confirm("  Record anyway?", default=False):
            return

    print()
    print(f"  patch_id      {patch_id}")
    print(f"  release_date  {patch_date}")
    print(f"  heroes_nerfed {nerfed or '(blank)'}")
    print(f"  heroes_buffed {buffed or '(blank)'}")
    if not confirm("  Write this to data/patches.csv?", default=True):
        print("  Cancelled.")
        return

    scraper.append_patch_record(
        patches_csv=scraper.PATCHES_DEFAULT, patch_id=patch_id, patch_date=patch_date,
        heroes_nerfed=nerfed, heroes_buffed=buffed, notes=notes,
    )
    print(dim("\n  Next: rebuild the datasets (option 7), then train (option 8)."))


def _unknown_heroes(pipe_list: str) -> list[str]:
    """Names in a pipe-separated list that don't appear in the archive's roster."""
    names = [h.strip() for h in pipe_list.split("|") if h.strip()]
    if not names:
        return []
    try:
        import store
        long_df = store.load_snapshots()
        if long_df.empty:
            return []
        roster = {h.lower() for h in long_df[long_df["ts"] == long_df["ts"].max()]["hero"]}
        return [n for n in names if n.lower() not in roster]
    except Exception:
        return []


def act_sync_patchnotes() -> None:
    """Read the official patch notes and offer to add the rows."""
    import patchnotes
    print(dim("  Reads mlbbhub's patch notes, matches hero names against the archive roster,\n"
              "  and shows what it would add. Nothing is written until you confirm.\n"
              "  NOTE: this parser has not been verified against a live page -- check its output.\n"))
    channel = choose("Which channel?", list(patchnotes.CHANNELS), default="original")
    if channel == "advanced":
        print(yellow("  Advance Server patches run weeks ahead of live and don't match the stats\n"
                     "  archive. Useful as a heads-up, not as training labels."))
    limit = ask_int("  How many of the newest patches to read", 8)

    rows = patchnotes.sync(channel=channel, limit=limit, apply=False)
    if not rows:
        return
    if confirm(f"\n  Write these {len(rows)} row(s) to data/patches.csv?", default=False):
        added = patchnotes.apply_rows(rows)
        print(green(f"  Added {added} row(s)."))
        print(dim("  Next: rebuild the datasets (option 7), then train (option 8)."))
    else:
        print("  Nothing written.")


def act_show_patches() -> None:
    """Print the patch calendar as recorded."""
    import pandas as pd
    from promote import PATCHES
    if not os.path.exists(PATCHES):
        print(yellow(f"  No {PATCHES} yet -- record a patch with option 4."))
        return
    import store
    patches = pd.read_csv(PATCHES, dtype=str).fillna("").sort_values("release_date").reset_index(drop=True)
    snap_dates = sorted({ref.date for ref in store.iter_refs()})
    print(f"  {len(patches)} patch(es) in {PATCHES}\n")
    for i, (_, row) in enumerate(patches.iterrows()):
        nxt = patches.iloc[i + 1] if i + 1 < len(patches) else None
        labeled = bool(nxt is not None and nxt["heroes_nerfed"].strip())
        start = row["release_date"]
        end = nxt["release_date"] if nxt is not None else None
        in_window = [d for d in snap_dates if d >= start and (end is None or d < end)]
        if labeled and in_window:
            tag = green(f"trainable · {len(in_window)} snapshot day(s)")
        elif labeled:
            tag = yellow("labeled, but no snapshots in its window")
        elif in_window:
            tag = yellow(f"{len(in_window)} snapshot day(s), awaiting the next patch's nerf list")
        else:
            tag = dim("no label, no snapshots")
        print(f"  {bold(row['patch_id']):<14} {row['release_date']}  [{tag}]")
        print(f"    nerfed  {row['heroes_nerfed'] or dim('(none recorded)')}")
        print(f"    buffed  {row['heroes_buffed'] or dim('(none recorded)')}")
        if row["notes"]:
            print(f"    notes   {dim(row['notes'])}")
    print(dim("\n  A window is 'trainable' only when BOTH hold: the next patch's nerf list is recorded\n"
              "  (the label) and snapshots were collected during it (the data)."))


def act_rebuild() -> None:
    """Rebuild train.csv (promote.py) and features.csv (dataset.py)."""
    print(bold("  data/train.csv  (one row per hero per patch -- the original contract)"))
    rc = subprocess.call([sys.executable, os.path.join("scripts", "promote.py")])
    if rc != 0:
        print(yellow("  promote.py reported a problem; see above."))
    print()
    print(bold("  data/features.csv  (every snapshot, with trend features)"))
    subprocess.call([sys.executable, os.path.join("scripts", "dataset.py")])


def act_train() -> None:
    """Cross-validate and save the nerf model."""
    import model as modelmod
    only = choose("Which estimator(s)?", ["both", "logreg", "gbm"], default="both")
    print()
    try:
        bundle = modelmod.train(only=None if only == "both" else only)
    except RuntimeError as exc:
        print(red(f"  Cannot train yet:\n\n{exc}"))
        return
    print(modelmod.format_report(bundle))
    print(green(f"\n  Saved -> {bundle.get('path', modelmod.MODEL_PATH)}"))


def act_model_report() -> None:
    """Print the saved model's CV scores and what it keys on."""
    import model as modelmod
    bundle = modelmod.load()
    if bundle is None:
        print(yellow("  No model trained yet -- option 8 trains one."))
        return
    print(modelmod.format_report(bundle, top_features=20))


def act_forecast_quick() -> None:
    """Rank nerf candidates with the best available engine."""
    _run_forecast(engine="auto")


def act_forecast_custom() -> None:
    """Rank nerf candidates, choosing engine, rank, date and list length."""
    import forecast
    import store
    engine = choose("Engine?", ["auto", "model", "trend", "baseline"], default="auto")
    print()
    rank = ask_ranks()[0]
    date = ask_date("  Snapshot date (blank = latest)", "")
    top = ask_int("  How many heroes to list", forecast.TOP_N)
    write = confirm("  Write a markdown report to predictions/?", default=True)
    print()
    _run_forecast(engine=engine, rank=rank, date=date or None, top=top, write=write)


def act_forecast_baseline() -> None:
    """The original predict.py heuristic, unchanged, for comparison."""
    print(dim("  scripts/predict.py's heuristic exactly as written: one snapshot, two terms,\n"
              "  no history. Useful as the thing to beat.\n"))
    _run_forecast(engine="baseline")


def _run_forecast(engine: str, rank: str | None = None, date: str | None = None,
                  top: int | None = None, write: bool = True) -> None:
    import forecast
    import store
    try:
        ranked, meta = forecast.forecast(
            engine=engine, rank=rank or store.DEFAULT_RANK, date=date, top_n=top or forecast.TOP_N,
        )
    except RuntimeError as exc:
        print(red(f"  {exc}"))
        return

    print(f"  Engine {bold(meta['engine'])} · {meta['rank']} · snapshot {meta['date']} · "
          f"{meta['snapshots_used']} snapshot(s) of history over {meta['history_days']} day(s)")
    if meta["model"] and not meta["model"]["cv_reliable"]:
        print(yellow(f"  Model saw only {len(meta['model']['patch_windows'])} labeled patch window(s) -- "
                     "treat the probabilities as rough."))
    if meta["engine"] in ("trend", "baseline"):
        print(dim("  Scores come from unvalidated heuristic weights, not a fitted model."))
    print()
    for line in forecast.format_console(ranked, meta).splitlines():
        print("  " + line)
    if write:
        print(green(f"\n  Wrote {forecast.write_markdown(ranked, meta)}"))


def act_backtest() -> None:
    """Score past patch windows against the heroes that actually got nerfed."""
    import backtest
    print(dim("  Takes the last snapshot before each patch landed, ranks every hero, and shows\n"
              "  where the heroes that really got nerfed came out. The model is scored\n"
              "  leave-one-patch-out, so it never grades a window it trained on.\n"))
    k = ask_int("  Top-k to score", backtest.DEFAULT_K)
    print()
    try:
        res = backtest.run(k=k)
    except RuntimeError as exc:
        print(red(f"  {exc}"))
        return
    for line in backtest.format_report(res).splitlines():
        print("  " + line)


def act_health() -> None:
    """Collection coverage, gaps and anything odd in the archive."""
    import trends
    print(trends.format_health(trends.health()))


def act_hero() -> None:
    """One hero's tracked history, with sparklines."""
    import store
    import trends
    name = ask("  Hero name")
    if not name:
        return
    rank = ask_ranks()[0]
    try:
        hist = trends.hero_history(name, rank=rank)
    except RuntimeError as exc:
        print(red(f"  {exc}"))
        return
    long_df = store.load_snapshots(ranks=(rank,))
    canonical = long_df[long_df["hero"].str.lower() == name.lower()]["hero"].iloc[0]
    print()
    for line in trends.format_hero_history(hist, canonical, rank).splitlines():
        print("  " + line)


def act_compare() -> None:
    """What moved between two snapshot dates."""
    import store
    import trends
    long_df = store.load_snapshots()
    if long_df.empty:
        print(yellow("  Archive is empty."))
        return
    dates = sorted(long_df["date"].unique())
    print(dim(f"  Archive covers {dates[0]} .. {dates[-1]} ({len(dates)} dates)\n"))
    date_a = ask_date("  Earlier date", dates[0])
    date_b = ask_date("  Later date", dates[-1])
    rank = ask_ranks()[0]
    by = choose("Sort by biggest change in?", ["d_win", "d_ban", "d_pick"], default="d_win")
    top = ask_int("  How many rows", 15)
    try:
        cmp = trends.compare(date_a, date_b, rank=rank)
    except RuntimeError as exc:
        print(red(f"  {exc}"))
        return
    print(f"\n  {rank}: {date_a} → {date_b}, {len(cmp)} heroes in both\n")
    show = cmp.reindex(cmp[by].abs().sort_values(ascending=False).index).head(top)
    cols = ["hero", "role", "win_then", "win_now", "d_win", "ban_then", "ban_now", "d_ban", "d_pick"]
    for line in show[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}").splitlines():
        print("  " + line)
    if cmp.attrs.get("new_heroes"):
        print(f"\n  New since {date_a}: {', '.join(cmp.attrs['new_heroes'])}")


def act_install() -> None:
    """Install Python dependencies and the Chromium build Playwright needs."""
    print(dim("  Runs: pip install -r requirements.txt\n"))
    if not confirm("  Go ahead?", default=True):
        return
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if rc != 0:
        print(red("  pip failed; see above."))
        return
    print()
    if confirm("  Also install the Chromium build Playwright needs (a few hundred MB)?", default=True):
        subprocess.call([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"])
    print(green("\n  Done. Scraping (options 1-3) should work now."))


def act_selftest() -> None:
    """Check the whole pipeline against a synthetic archive -- no network, no real data touched."""
    import selftest
    failures = selftest.run(verbose=True)
    print()
    if failures:
        print(red(f"  {len(failures)} check(s) failed:"))
        for f in failures:
            print(red(f"   - {f}"))
    else:
        print(green("  All checks passed."))


# --------------------------------------------------------------------------------------
# menu wiring
# --------------------------------------------------------------------------------------

@dataclass
class Item:
    key: str
    label: str
    action: Callable[[], None]
    needs_net: bool = False


@dataclass
class Section:
    title: str
    items: list[Item]


SECTIONS = [
    Section("COLLECT DATA", [
        Item("1", "Take a snapshot now (choose rank)", act_snapshot_pick, needs_net=True),
        Item("2", "Take a snapshot now (all 5 ranks)", act_snapshot_all, needs_net=True),
        Item("3", "Keep collecting on an interval (unlimited)", act_watch, needs_net=True),
    ]),
    Section("PATCH CALENDAR", [
        Item("4", "Record a patch by hand", act_record_patch),
        Item("5", "Sync patches from official patch notes", act_sync_patchnotes, needs_net=True),
        Item("6", "Show the patch calendar", act_show_patches),
    ]),
    Section("TRAIN", [
        Item("7", "Rebuild datasets from the archive", act_rebuild),
        Item("8", "Train the nerf model", act_train),
        Item("9", "Model report (scores, what it keys on)", act_model_report),
    ]),
    Section("PREDICT", [
        Item("10", "Forecast nerf candidates", act_forecast_quick),
        Item("11", "Forecast with options (engine, rank, date)", act_forecast_custom),
        Item("12", "Baseline heuristic only (original predict.py)", act_forecast_baseline),
    ]),
    Section("VALIDATE", [
        Item("18", "Back-test against patches that already shipped", act_backtest),
    ]),
    Section("INSPECT", [
        Item("13", "Archive health and coverage", act_health),
        Item("14", "Hero deep-dive", act_hero),
        Item("15", "Compare two dates", act_compare),
    ]),
    Section("SETUP", [
        Item("16", "Install dependencies", act_install),
        Item("17", "Run the self-test (no network needed)", act_selftest),
    ]),
]

ITEMS: dict[str, Item] = {item.key: item for section in SECTIONS for item in section.items}


def render_menu() -> str:
    width = 78
    out = [
        "",
        cyan("╭" + "─" * width + "╮"),
        cyan("│") + bold("  MLBB NERF PREDICTOR".ljust(width)) + cyan("│"),
        cyan("╰" + "─" * width + "╯"),
    ]
    out += status_lines()
    for section in SECTIONS:
        out.append("")
        out.append("  " + bold(section.title))
        for item in section.items:
            net = dim("  ↯") if item.needs_net else ""
            out.append(f"   {cyan(item.key.rjust(2))}  {item.label}{net}")
    out.append("")
    out.append(f"    {cyan('q')}  Quit")
    out.append("")
    out.append(dim("  ↯ needs internet and Playwright"))
    return "\n".join(out)


def run_item(item: Item) -> None:
    print()
    print(bold(f"── {item.label} " + "─" * max(0, 60 - len(item.label))))
    print()
    try:
        item.action()
    except KeyboardInterrupt:
        print(yellow("\n  Interrupted."))
    except RuntimeError as exc:
        print(red(f"\n  {exc}"))
    except ImportError as exc:
        print(red(f"\n  Missing dependency: {exc}"))
        print(yellow("  Option 16 installs everything this repo needs."))
    except Exception:
        print(red("\n  Unexpected error:"))
        traceback.print_exc()
        print(yellow("  The archive is append-only, so nothing already collected was affected."))


def interactive() -> int:
    print(render_menu())
    while True:
        try:
            choice = input(f"\n{bold('Pick an option')} (q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if choice in ("q", "quit", "exit"):
            print("Bye.")
            return 0
        if not choice:
            print(render_menu())
            continue
        if choice in ("m", "menu", "?", "h", "help"):
            print(render_menu())
            continue
        item = ITEMS.get(choice)
        if item is None:
            print(red(f"  No option {choice!r}. Options: {', '.join(ITEMS)} or q."))
            continue
        run_item(item)
        try:
            input(dim("\n  Enter for the menu, or Ctrl-C to quit "))
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        print(render_menu())


def main() -> int:
    ap = argparse.ArgumentParser(description="MLBB Nerf Predictor menu.")
    ap.add_argument("--run", metavar="N", help="Run option N and exit, without showing the menu.")
    ap.add_argument("--list", action="store_true", help="Print the options and exit.")
    ap.add_argument("--status", action="store_true", help="Print the status block and exit.")
    args = ap.parse_args()

    if args.status:
        print("\n".join(status_lines()))
        return 0
    if args.list:
        print(render_menu())
        return 0
    if args.run:
        item = ITEMS.get(args.run.strip())
        if item is None:
            print(f"No option {args.run!r}. Options: {', '.join(ITEMS)}", file=sys.stderr)
            return 2
        run_item(item)
        return 0

    if not sys.stdin.isatty():
        print(render_menu())
        print(dim("\n  Not a terminal, so there's nothing to read a choice from.\n"
                  "  Use `python mlbb.py --run <option>` instead, e.g. `--run 10`."))
        return 0

    return interactive()


if __name__ == "__main__":
    sys.exit(main())
