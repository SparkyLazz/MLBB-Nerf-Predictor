# MLBB Nerf Predictor

Mobile Legends: Bang Bang's own stat sites (mlbbhub, mlbb.io, mobadraft, and friends) only ever
show the *current* live win/pick/ban rate — none of them archive it per patch, and the Wayback
Machine doesn't have usable snapshots either. So there is no way to look back and ask "what did
this hero's numbers look like right before it got nerfed" unless someone is capturing that data
themselves, every day, before it scrolls off.

That's what this repo does: it scrapes hero stats on a schedule and keeps every snapshot
permanently, with the goal of predicting which heroes are likely to get nerfed in the *next*
patch, from the numbers visible in the *current* one. Daily collection started **2026-08-04**.
Before that date, no historical stats exist for this game and none can be reconstructed — the
training set only grows forward from here.

## Start here

```bash
pip install -r requirements.txt
playwright install --with-deps chromium

python mlbb.py
```

That opens a menu — pick a number, it runs, you come back to the menu. Everything the repo can
do is in there, and the header always shows the real state of the archive:

```
╭──────────────────────────────────────────────────────────────────────────────╮
│  MLBB NERF PREDICTOR                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
  Archive  30 snapshots · 30 days (2026-08-04 → 2026-09-12) · 133 heroes · ranks: mythic
  Patches  2 recorded · latest 2.1.90 (2026-07-08) · 0 trainable window(s)
  Model    not trained -- the trend engine (option 10) works without one

  COLLECT DATA          1  snapshot now (choose rank)      2  snapshot now (all 5 ranks)
                        3  keep collecting on an interval
  PATCH CALENDAR        4  record a patch by hand          5  sync from official patch notes
                        6  show the patch calendar
  TRAIN                 7  rebuild datasets                8  train the nerf model
                        9  model report
  PREDICT              10  forecast nerf candidates       11  forecast with options
                       12  baseline heuristic only
  VALIDATE             18  back-test against shipped patches
  INSPECT              13  archive health and coverage     14  hero deep-dive
                       15  compare two dates
  SETUP                16  install dependencies           17  run the self-test
```

The menu is only a front end. Every script still runs on its own from the repo root, and
`python mlbb.py --run 10` goes straight to one option without the menu (handy for cron or CI).
`python mlbb.py --status` prints just the header.

No dependencies installed yet? The menu still opens — option 16 installs everything. Option 17
then checks the whole pipeline against a synthetic archive, offline, without touching real data.

## What it predicts, and how honest it is

**Where this actually stands.** The patch calendar now covers 2.1.95 (2026-08-04), 2.1.95a
(2026-08-26) and 2.2.16 (2026-09-16), which splits the archive into **2 trainable windows** —
just enough to train. A model exists. But its honest scores are weak, and the back-test
(option 18) says something uncomfortable that is worth reading before trusting any of this:

```
2.1.95a -> 2.2.16   snapshot 2026-09-12, 5 heroes actually nerfed
  baseline  3/5 in top 10   Hanabi#1, Miya#2, Paquito#6, Melissa#14, Yi Sun-shin#16
  trend     2/5 in top 10   Miya#1, Hanabi#2, Melissa#12, Paquito#23, Yi Sun-shin#24
  model     2/5 in top 10   Hanabi#6, Paquito#8, Miya#12, Melissa#29, Yi Sun-shin#36
```

On the only window with enough nerfs to measure, the **original two-term heuristic beat both of
the additions**. Three findings follow, and none of them are flattering:

1. The signal is real. All five nerfed heroes land in the top ~24 of 133 under either heuristic,
   from a snapshot taken four days before the patch, with no knowledge of it.
2. The `trend` engine's momentum term is miscalibrated. It demoted Paquito from #6 to #23
   because his ban rate was *falling* — and he was nerfed anyway. A cooling hero is not a safe
   hero, and that term currently assumes otherwise.
3. The model is underpowered, not wrong in principle. Leave-one-patch-out on two windows means
   it trains on a single window containing one positive hero (Atlas, a high-ban tank), so it
   learned "high ban rate" and little else.

The fix for all three is the same: more recorded patches. Two windows is the minimum to train at
all; four or more before the cross-validation numbers mean anything. **Collection also stopped
on 2026-09-12**, so the current patch (2.2.16) has zero snapshots and the tool cannot predict
the *next* patch until collection resumes — start with option 1 or 2.

Three prediction engines, chosen with option 11 or `--engine`:

| Engine | Needs labels? | What it uses |
|---|---|---|
| `baseline` | no | `scripts/predict.py` exactly as written: one snapshot, win-rate and ban-rate z-scores, 0.45/0.55 weights |
| `trend` | no | the baseline score (imported, not copied) plus 7-day momentum in win/ban/contest rate and how long the hero has been hot |
| `model` | **yes** | a classifier fitted on the archive across ~60 trend features, returning a probability per hero |
| `auto` | — | `model` if one is trained, otherwise `trend` |

`baseline` is kept deliberately so you can always see what the extra machinery adds — and right
now, on the one measurable window, it adds nothing. Keep comparing with option 12 and option 18
rather than assuming the more complex engine is better.

The `trend` engine's extra weights are unvalidated guesses, exactly like the baseline's
0.45/0.55. That's the point of the `model` engine: learn them instead of guessing. Both engines
print a note saying so, and every report says which engine produced it.

### What the model is, and how it's validated

Two candidates are cross-validated and the better one is kept: a class-balanced **logistic
regression** (readable, works with very little data) and a **histogram gradient-boosting
classifier** (handles NaN natively, finds interactions, needs more data to pay off).

Validation is **leave-one-patch-out**: each fold holds out a whole patch window. Anything else
leaks — twenty daily snapshots of the same hero inside one patch are nearly the same row, so a
random split would score ~0.99 and mean nothing. `precision@10` / `recall@10` are computed **one
snapshot at a time and then averaged**, because that's how the tool is used: rank today's 133
heroes, read the top 10. Rows are weighted so every patch window counts equally regardless of
how many days were collected during it.

Option 9 prints the scores, the caveat when there are too few windows, and which features the
model keys on. **Option 18 back-tests every engine against patches that already shipped** —
taking the last snapshot before each patch landed and reporting where the heroes that really got
nerfed came out. The model is scored leave-one-patch-out there, so it never grades a window it
trained on. That option, not the CV printout, is the one to believe.

### A label trap worth knowing about

A patch row with an empty `heroes_nerfed` list is ambiguous: it can mean "this patch nerfed
nobody" or "nobody has transcribed it yet". `dataset.py` treats it as the second, leaves the
window **unlabeled**, and warns — because reading it the other way turns a whole window of
heroes into confident negatives that teach the model the opposite of the truth, silently. Heroes
listed as *adjusted* in the patch notes go in neither list for the same reason: the notes
distinguish adjust from nerf, so inventing a label either way would be guessing.

## Unlimited tracking

The original pipeline had four hard ceilings on how much data it could hold or use. All four are
gone:

| Was | Now |
|---|---|
| One snapshot per day, ever | Any number per day. `data/archive/<rank>/<date>T<time>Z.csv` is timestamped to the second, and option 3 collects on a repeating interval until stopped |
| Mythic only | All five brackets in one run (option 2). This also switches on the cross-rank features, which are NaN until a second bracket exists |
| A hard `EXPECTED_HERO_COUNT = 133` that **failed the scrape** when it didn't match | A *floor* (`MIN_HERO_COUNT`, default 125). Below it still fails hard — that's a half-rendered page, and writing it would poison the archive. Above it, a new hero only prints a note, so Moonton shipping hero 134 no longer silently stops collection |
| `promote.py` kept **one snapshot per patch window** and discarded the rest | `dataset.py` keeps **every** snapshot. Three weeks of daily Mythic collection in one window is 21 rows per hero instead of 1 — and with all five brackets, 105 |

`promote.py` and `data/train.csv` are untouched and still mean exactly what they used to; the
unlimited path is a second, parallel artifact. Nothing was migrated and nothing was deleted.

Reading thousands of small CSVs gets slow, so the merged archive is cached in `data/.cache/`
behind a manifest of file sizes and mtimes. It's pure derived data — delete it any time.

## Folder structure

```
mlbb-nerf-predictor/
├── mlbb.py                        # THE MENU -- start here
├── .github/workflows/scrape.yml   # cron (daily archive, all ranks) + workflow_dispatch (patch labeling)
├── scripts/
│   ├── scraper.py                 # headless Playwright scraper (daily / snapshot / labeled modes)
│   ├── store.py                   # unified reader over both snapshot layouts, + cache
│   ├── features.py                # trend/cohort/rolling/cross-rank feature engineering
│   ├── dataset.py                 # data/features.csv: every snapshot, labeled
│   ├── promote.py                 # data/train.csv: one row per hero per patch (unchanged)
│   ├── model.py                   # train / cross-validate / save the classifier
│   ├── forecast.py                # ranked nerf candidates, model|trend|baseline
│   ├── predict.py                 # the original heuristic (unchanged -- see note below)
│   ├── patchnotes.py              # read official patch notes -> proposed patches.csv rows
│   ├── backtest.py                # score past windows against what really got nerfed
│   ├── trends.py                  # hero history, date comparison, data health
│   └── selftest.py                # end-to-end check on a synthetic archive, offline
├── data/
│   ├── daily/                     # legacy: one CSV per day, Mythic, append-only, never edited
│   ├── archive/<rank>/            # unlimited: any rank, any cadence, append-only
│   ├── patches.csv                # hand-curated patch calendar: release dates + nerf/buff lists
│   ├── train.csv                  # derived from daily/ + patches.csv -- delete & rebuild anytime
│   ├── features.csv               # derived, gitignored -- every snapshot + features + label
│   └── .cache/                    # derived, gitignored -- merged-archive cache
├── models/                        # derived, gitignored -- the trained model
├── predictions/                   # dated top-10 output, one .md per run
└── notebooks/
```

Every script assumes it's run from the repo root, e.g. `python scripts/scraper.py daily`.
`mlbb.py` changes to the repo root itself, so it works from anywhere.

## Data model

- **`data/daily/<date>.csv`** — raw, unlabeled (patch_id/patch_date blank). One file per UTC
  date, Mythic. Append-only: no script in this repo ever overwrites or deletes a file in here.
- **`data/archive/<rank>/<date>T<time>Z.csv`** — same columns, same append-only rule, but one
  directory per rank and a timestamp instead of a date, so there's no cap on cadence. A Mythic
  snapshot also mirrors into `data/daily/` when that day has no file yet, which keeps the legacy
  layout and everything reading it correct. Readers drop the mirror so a day collected by both
  layouts is never counted twice.
- **`data/patches.csv`** — the only place patch identity lives. Columns:
  `patch_id, release_date, heroes_nerfed, heroes_buffed, notes`, hero lists pipe-separated
  (`Baxia|Akai`). Rows are added by a human (option 4) or proposed by `patchnotes.py` and
  confirmed by a human (option 5) — **never auto-detected on a schedule**. A patch's "active
  window" runs from its `release_date` up to the next patch's `release_date`.
- **`data/train.csv`** — pure derived output of `promote.py`. For each patch window it keeps the
  *latest* daily snapshot inside that window, and computes `nerfed_next` from the *following*
  patch's `heroes_nerfed`. Safe to delete; `python scripts/promote.py` rebuilds it. Note this
  file legitimately changes as collection continues, because "latest snapshot in the window"
  moves forward each day.
- **`data/features.csv`** — pure derived output of `dataset.py`. Every snapshot × hero, ~60
  features, labeled the same way. Gitignored because it's large and fully reproducible.

Every scrape validates that the hero-average `win_rate` falls within 48–52% (every ranked match
has exactly one winner and one loser, so it should hover near 50%) before anything is written to
disk. Outside that band, the script exits non-zero and writes nothing — that check lives in the
fetch path, not the save path, so there's no way to bypass it.

## On `predict.py` and `patches.csv`: two rules this repo had, and how they were kept

**`scripts/predict.py` says not to change, reweight or extend its heuristic without discussing
first.** It hasn't been. The file is byte-identical. `forecast.py` *imports* `predict.score` and
builds on top of it, so the baseline stays single-sourced and you can always compare against it
with option 12.

**`patches.csv` says patch identity is never auto-detected.** `patchnotes.py` reads the official
notes, but it defaults to a dry run, every hero name it parses is checked against the roster the
scraper has actually seen (anything else is reported and dropped, never guessed), it only ever
appends new `patch_id`s, every row it writes records its source URL and fetch time in `notes`,
and **nothing on a schedule calls it**. Writing still takes a human saying yes.

One caveat on that parser: unlike the stats scraper — whose extraction JS was verified against
the live page — `patchnotes.py`'s selectors have **not** been checked against a live patch-notes
page, because the site was unreachable from where it was written. It's written defensively and
fails loudly rather than writing garbage, and `--dump-html` saves a page for inspection. Read
its dry-run output before accepting it. Option 4 (by hand) is always reliable.

## Running it from the command line

```bash
# Unlimited collection: every rank, archived; Mythic also mirrors to data/daily/
python scripts/scraper.py snapshot --ranks all
python scripts/scraper.py snapshot --ranks mythic,mythical_glory --every 60   # until stopped
python scripts/scraper.py daily                                              # legacy single-day mode

# Record a patch (the only way a row enters patches.csv)
python scripts/scraper.py labeled --patch-id 2.2.16 --patch-date 2026-09-30 \
  --nerfed "Karina|Lukas|Melissa" --buffed "Argus|Aulus|Gloo"
python scripts/patchnotes.py --limit 8            # dry run: propose rows from official notes
python scripts/patchnotes.py --limit 8 --apply    # write them

# Rebuild derived data
python scripts/promote.py      # data/train.csv
python scripts/dataset.py      # data/features.csv

# Train and inspect
python scripts/model.py
python scripts/model.py --report

# Predict
python scripts/forecast.py                                  # auto engine, latest snapshot
python scripts/forecast.py --engine trend --rank mythical_glory --top 20
python scripts/forecast.py --engine baseline --date 2026-08-22
python scripts/predict.py                                   # the original, untouched

# Validate against patches that already shipped
python scripts/backtest.py
python scripts/backtest.py --engines baseline,trend --k 5

# Inspect the archive
python scripts/trends.py health
python scripts/trends.py hero Fredrinn
python scripts/trends.py compare 2026-08-04 2026-09-12 --by d_ban

# Verify everything works, offline
python scripts/selftest.py -v
```

## When collection stops

It already happened once, and the way it happened is worth knowing about: between 2026-09-12 and
2026-10-04 the scheduled job **ran every single day and failed every single day** — ten
consecutive `failure` runs — always at the same line:

```
TimeoutError: Locator.wait_for: Timeout 30000ms exceeded.
  - waiting for get_by_role("button", name="Mythic", exact=True) to be visible
```

mlbbhub changed its rank-filter markup. The page still loaded; only the one hard-coded selector
stopped matching. Three weeks of data was lost to that, and nothing surfaced it — the archive
just quietly stopped growing while the cron kept green-lighting itself into the same crash.

Two changes so this is cheaper next time:

- **The scraper tries seven markup shapes** for the rank filter (`role=button`, `role=tab`,
  `role=option`, `role=radio`, `aria-label`, exact text, and `<select>` option) and logs which
  one matched when it isn't the primary, so you know the page drifted before it breaks outright.
- **When all of them miss, the error dumps the live page**: title, whether the stats container
  still matches, every element whose text is exactly a rank name, all buttons with their classes
  and aria-labels, tab/option roles, selects with their options, and any 50+ child container
  that could be the new stats table. The next breakage should be a one-log-read fix.

To investigate without writing anything:

```bash
python scripts/scraper.py diagnose                  # what does the page offer right now?
python scripts/scraper.py diagnose --dump-html /tmp/stats.html
```

Or run it on GitHub: **Actions → Scrape MLBB Patch Stats → Run workflow → mode: `diagnose`**.
That prints the same probe and uploads the page HTML as an artifact, which is the fastest way to
see the live DOM if you can't reach the site locally.

`workflow_dispatch` now takes a **`mode`**:

| mode | what it does |
|---|---|
| `snapshot` | collect now, every rank — use this to restart collection or confirm a fix |
| `diagnose` | report what the page offers, write nothing |
| `labeled` | record a patch in `data/patches.csv` (needs `patch_id` + `patch_date`) |

Before, `labeled` was the *only* manual run and both patch fields were mandatory — so there was
no way to collect on demand, or to test a scraper fix, without also recording a patch.

Note the daily cron runs on the **default branch**, so a scraper fix only reaches scheduled
collection once it's merged there.

## Per-patch workflow

0. **Check the Action is actually passing.** It is the single point of failure for everything
   else here, it fails silently, and option 13 (archive health) is what tells you the archive
   stopped growing. See "When collection stops" above.
1. **Every day**, the scheduled GitHub Action (`0 15 * * *` UTC / 22:00 WIB) collects a snapshot
   for every rank bracket, rebuilds the derived datasets, and commits. Narrow `DAILY_RANKS` in
   the workflow if five brackets a day is more than you want.
2. **When a new patch drops**, record it: menu option 4, or the workflow's manual
   `workflow_dispatch` trigger, or option 5 to read it off the patch notes and confirm. This is
   the step that turns collected snapshots into training data, so it's the one that matters most.
3. **Then rebuild and retrain** — options 7 and 8. Option 9 tells you whether the scores are yet
   worth anything.
4. **For a read on the current meta**, option 10. It works whether or not a model exists.
