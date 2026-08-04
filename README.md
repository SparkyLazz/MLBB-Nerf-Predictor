# MLBB Nerf Predictor

Mobile Legends: Bang Bang's own stat sites (mlbbhub, mlbb.io, mobadraft, and friends) only ever
show the *current* live win/pick/ban rate — none of them archive it per patch, and the Wayback
Machine doesn't have usable snapshots either. So there is no way to look back and ask "what did
this hero's numbers look like right before it got nerfed" unless someone is capturing that data
themselves, every day, before it scrolls off.

That's what this repo does: it scrapes Mythic-rank hero stats daily and keeps every snapshot
permanently, with the goal of eventually predicting which heroes are likely to get nerfed in the
*next* patch, from the numbers visible in the *current* one. Daily collection started
**2026-08-04**. Before that date, no historical stats exist for this game and none can be
reconstructed — the training set only grows forward from here.

## Folder structure

```
mlbb-nerf-predictor/
├── .github/workflows/scrape.yml   # cron (daily archive) + workflow_dispatch (patch labeling)
├── scripts/
│   ├── scraper.py                 # headless Playwright scraper (daily / labeled modes)
│   ├── promote.py                 # data/daily/ + data/patches.csv -> data/train.csv
│   └── predict.py                 # heuristic scoring -> top 10 nerf candidates
├── data/
│   ├── daily/                     # one CSV per day, unlabeled, append-only, never edited
│   ├── patches.csv                # hand-curated patch calendar: release dates + nerf/buff lists
│   └── train.csv                  # derived from daily/ + patches.csv -- delete & rebuild anytime
├── predictions/                   # predict.py's dated top-10 output, one .md per run
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

Every script assumes it's run from the repo root, e.g. `python scripts/scraper.py daily`.

## Data model

- **`data/daily/<date>.csv`** — raw, unlabeled (patch_id/patch_date blank). One file per UTC
  date. Append-only: no script in this repo ever overwrites or deletes a file already in here.
- **`data/patches.csv`** — the only place patch identity lives. Columns:
  `patch_id, release_date, heroes_nerfed, heroes_buffed, notes`, hero lists pipe-separated
  (`Baxia|Akai`). Rows are added by hand or via `scraper.py labeled` — **never auto-detected**.
  A patch's "active window" runs from its `release_date` up to the next patch's `release_date`.
- **`data/train.csv`** — pure derived output of `promote.py`. For each patch window, it keeps the
  *latest* daily snapshot inside that window (closest to end-of-patch, not day one), and computes
  `nerfed_next` from whether each hero appears in the *following* patch's `heroes_nerfed`. Safe to
  delete; `python scripts/promote.py` rebuilds it byte-for-byte from the two inputs above.

## Running it

```bash
pip install -r requirements.txt
playwright install --with-deps chromium

# Daily archive snapshot (safe to run repeatedly -- skips if today's file already exists)
python scripts/scraper.py daily

# Record a new patch once you've confirmed it from official patch notes
python scripts/scraper.py labeled --patch-id 2.1.92 --patch-date 2026-07-29 \
  --nerfed "HeroA|HeroB" --buffed "HeroC"

# Rebuild the training set from data/daily/ + data/patches.csv
python scripts/promote.py

# Score the latest snapshot for nerf candidates -> prints top 10, writes predictions/<date>_<patch_id>.md
python scripts/predict.py
```

Every scrape validates that the hero-average `win_rate` falls within 48–52% (every ranked match
has exactly one winner and one loser, so it should hover near 50%) before anything is written to
disk. Outside that band, the script exits non-zero and writes nothing — that check lives in the
fetch path, not the save path, so there's no way to bypass it.

## Per-patch workflow

1. **Every day**, the scheduled GitHub Action (`0 15 * * *` UTC / 22:00 WIB) runs
   `scraper.py daily` and commits whatever lands in `data/daily/`.
2. **When a new patch drops**, trigger the workflow manually (`workflow_dispatch`) with the
   patch's version and release date — and the nerf/buff hero lists too, if you already have them
   from the patch notes. This appends one row to `data/patches.csv` (and takes a same-day daily
   snapshot if one doesn't exist yet).
3. **Whenever you want a training set**, run `promote.py`. It re-derives `data/train.csv` from
   scratch, so there is never a risk of it drifting out of sync with the raw archive.
4. **For a quick read on the current meta**, run `predict.py`. It scores whatever the latest
   `data/daily/` snapshot is, independent of whether that day's patch window has been closed out
   yet in `patches.csv`.
