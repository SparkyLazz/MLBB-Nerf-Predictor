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

**The headline caveat, read this first:** as of the latest commit there are **zero trainable
patch windows**, so the machine-learning model *cannot be trained yet*. The archive holds a
month of snapshots, but all of them fall inside patch 2.1.90's window, and a window only becomes
training data once the patch *after* it is recorded with its nerf list. Option 6 shows this
plainly per patch. Until that changes:

- **Option 10 still works.** It falls back to the `trend` engine, which needs no labels.
- **To unlock the model**, record the patches that have shipped since 2.1.90 (option 4 by hand,
  or option 5 to read them off the official notes). Each recorded patch with a nerf list turns
  the window before it into training rows. Two windows is the minimum to train; four or more
  before the cross-validation scores mean much.

Three prediction engines, chosen with option 11 or `--engine`:

| Engine | Needs labels? | What it uses |
|---|---|---|
| `baseline` | no | `scripts/predict.py` exactly as written: one snapshot, win-rate and ban-rate z-scores, 0.45/0.55 weights |
| `trend` | no | the baseline score (imported, not copied) plus 7-day momentum in win/ban/contest rate and how long the hero has been hot |
| `model` | **yes** | a classifier fitted on the archive across ~60 trend features, returning a probability per hero |
| `auto` | — | `model` if one is trained, otherwise `trend` |

`baseline` is kept deliberately so you can always see what the extra machinery adds. On the
current archive, the trend engine promotes Hilda (ban rate +4.8 in a week) and Hirara into the
top 10 and demotes Paquito and Rafaela, whose numbers are falling — movement the single-day
heuristic is blind to by construction.

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
model keys on.

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

# Inspect the archive
python scripts/trends.py health
python scripts/trends.py hero Fredrinn
python scripts/trends.py compare 2026-08-04 2026-09-12 --by d_ban

# Verify everything works, offline
python scripts/selftest.py -v
```

## Per-patch workflow

1. **Every day**, the scheduled GitHub Action (`0 15 * * *` UTC / 22:00 WIB) collects a snapshot
   for every rank bracket, rebuilds the derived datasets, and commits. Narrow `DAILY_RANKS` in
   the workflow if five brackets a day is more than you want.
2. **When a new patch drops**, record it: menu option 4, or the workflow's manual
   `workflow_dispatch` trigger, or option 5 to read it off the patch notes and confirm. This is
   the step that turns collected snapshots into training data, so it's the one that matters most.
3. **Then rebuild and retrain** — options 7 and 8. Option 9 tells you whether the scores are yet
   worth anything.
4. **For a read on the current meta**, option 10. It works whether or not a model exists.
