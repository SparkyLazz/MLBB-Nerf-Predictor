"""Unified snapshot store -- the "track unlimited data" layer.

Two physical stores, one logical view:

- ``data/daily/<YYYY-MM-DD>.csv``
      Legacy layout: one file per UTC day, Mythic only, written by ``scraper.py daily``.
      Every snapshot collected before this module existed lives here. Read-only from here.

- ``data/archive/<rank>/<YYYY-MM-DD>T<HHMMSS>Z.csv``
      Unlimited layout: any rank bracket, any number of snapshots per day, kept forever.
      Written by ``scraper.py snapshot``.

Both stores are append-only. Nothing in this module ever overwrites or deletes a snapshot that
has already landed -- same rule the rest of the repo follows.

``load_snapshots()`` returns one long DataFrame (one row per snapshot x hero) across both
stores, which is what scripts/features.py, scripts/dataset.py and scripts/forecast.py all read.
If an archive snapshot exists for the same (date, rank) as a legacy daily file, the archive
copy wins and the legacy file is skipped, so a day collected by both layouts is never
double-counted.

Reading thousands of small CSVs gets slow, so the long frame is cached in
``data/.cache/snapshots.pkl.gz`` behind a manifest of (path, size, mtime). The cache is pure
derived data: delete it any time and it rebuilds. It is gitignored.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

DAILY_DIR = "data/daily"
ARCHIVE_DIR = "data/archive"
CACHE_DIR = "data/.cache"
CACHE_FILE = os.path.join(CACHE_DIR, "snapshots.pkl.gz")

# Rank brackets mlbbhub exposes, ordered weakest -> strongest. Keep in sync with
# scraper.RANK_LABELS (scraper imports this list so there is only one source of truth).
RANKS: tuple[str, ...] = ("epic", "legend", "mythic", "mythical_honor", "mythical_glory")
DEFAULT_RANK = "mythic"

# Columns every snapshot file carries, in order.
SNAPSHOT_COLUMNS = [
    "patch_id", "patch_date", "hero", "role",
    "win_rate", "pick_rate", "ban_rate", "rank_bracket", "nerfed_next",
]
# Columns load_snapshots() returns.
LONG_COLUMNS = ["ts", "date", "rank_bracket", "hero", "role", "win_rate", "pick_rate", "ban_rate"]

_DAILY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")
_ARCHIVE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:T(\d{6})Z)?\.csv$")


@dataclass(frozen=True)
class SnapshotRef:
    """One snapshot file on disk."""

    ts: pd.Timestamp  # UTC, tz-naive (midnight for legacy daily files)
    date: str  # YYYY-MM-DD
    rank: str
    path: str
    store: str  # "daily" (legacy) or "archive"

    @property
    def label(self) -> str:
        stamp = self.ts.strftime("%Y-%m-%d %H:%M") if self.store == "archive" else self.date
        return f"{stamp} UTC [{self.rank}]"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def archive_path(rank: str, when: datetime | None = None, archive_dir: str = ARCHIVE_DIR) -> str:
    """Path a snapshot taken at ``when`` for ``rank`` should be written to."""
    if rank not in RANKS:
        raise ValueError(f"rank must be one of {list(RANKS)}, got {rank!r}")
    when = when or utc_now()
    stamp = when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return os.path.join(archive_dir, rank, f"{stamp}.csv")


def iter_refs(
    ranks: tuple[str, ...] | list[str] | None = None,
    daily_dir: str = DAILY_DIR,
    archive_dir: str = ARCHIVE_DIR,
    include_legacy: bool = True,
) -> list[SnapshotRef]:
    """Every snapshot on disk, oldest first, across both stores.

    A legacy daily file is dropped when the archive already holds a Mythic snapshot for that
    same date, so the two layouts never contribute the same reading twice.
    """
    wanted = tuple(ranks) if ranks else RANKS
    refs: list[SnapshotRef] = []
    archived_mythic_dates: set[str] = set()

    for rank in wanted:
        pattern = os.path.join(archive_dir, rank, "*.csv")
        for path in sorted(glob.glob(pattern)):
            m = _ARCHIVE_RE.match(os.path.basename(path))
            if not m:
                continue
            date_str, time_str = m.group(1), m.group(2) or "000000"
            ts = pd.Timestamp(f"{date_str} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}")
            refs.append(SnapshotRef(ts=ts, date=date_str, rank=rank, path=path, store="archive"))
            if rank == DEFAULT_RANK:
                archived_mythic_dates.add(date_str)

    if include_legacy and DEFAULT_RANK in wanted:
        for path in sorted(glob.glob(os.path.join(daily_dir, "*.csv"))):
            m = _DAILY_RE.match(os.path.basename(path))
            if not m:
                continue
            date_str = m.group(1)
            if date_str in archived_mythic_dates:
                continue
            refs.append(
                SnapshotRef(ts=pd.Timestamp(date_str), date=date_str, rank=DEFAULT_RANK, path=path, store="daily")
            )

    refs.sort(key=lambda r: (r.ts, r.rank))
    return refs


def read_snapshot(ref: SnapshotRef) -> pd.DataFrame:
    """One snapshot file -> long rows. Trusts the file's own rank_bracket column when present."""
    df = pd.read_csv(ref.path)
    for col in ("hero", "role", "win_rate", "pick_rate", "ban_rate"):
        if col not in df.columns:
            raise RuntimeError(f"{ref.path} is missing required column {col!r}")
    out = pd.DataFrame({
        "ts": ref.ts,
        "date": ref.date,
        "rank_bracket": df["rank_bracket"].fillna(ref.rank) if "rank_bracket" in df.columns else ref.rank,
        "hero": df["hero"].astype(str).str.strip(),
        "role": df["role"].astype(str).str.strip().str.lower(),
        "win_rate": pd.to_numeric(df["win_rate"], errors="coerce"),
        "pick_rate": pd.to_numeric(df["pick_rate"], errors="coerce"),
        "ban_rate": pd.to_numeric(df["ban_rate"], errors="coerce"),
    })
    return out[LONG_COLUMNS]


def _manifest(refs: list[SnapshotRef]) -> list[tuple]:
    out = []
    for r in refs:
        try:
            st = os.stat(r.path)
        except OSError:
            continue
        out.append((r.path, r.rank, st.st_size, int(st.st_mtime)))
    return out


def load_snapshots(
    ranks: tuple[str, ...] | list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    daily_dir: str = DAILY_DIR,
    archive_dir: str = ARCHIVE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Every snapshot, as one long frame sorted by (ts, rank, hero).

    ``since``/``until`` are inclusive YYYY-MM-DD bounds applied after loading, so the cache stays
    valid across different windows.
    """
    refs = iter_refs(ranks=ranks, daily_dir=daily_dir, archive_dir=archive_dir)
    if not refs:
        return pd.DataFrame(columns=LONG_COLUMNS)

    manifest = _manifest(refs)
    frame: pd.DataFrame | None = None

    if use_cache and os.path.exists(CACHE_FILE):
        try:
            cached = pd.read_pickle(CACHE_FILE)
            if cached.attrs.get("manifest") == manifest:
                frame = cached
        except Exception:
            frame = None  # a corrupt or stale-format cache is never fatal -- just rebuild it

    if frame is None:
        frame = pd.concat([read_snapshot(r) for r in refs], ignore_index=True)
        frame = frame.dropna(subset=["win_rate", "pick_rate", "ban_rate"])
        frame = frame.sort_values(["ts", "rank_bracket", "hero"]).reset_index(drop=True)
        if use_cache:
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                frame.attrs["manifest"] = manifest
                frame.to_pickle(CACHE_FILE)
            except Exception:
                pass  # cache is an optimisation, never a requirement

    out = frame
    if since:
        out = out[out["date"] >= since]
    if until:
        out = out[out["date"] <= until]
    return out.reset_index(drop=True)


def latest_ref(ranks: tuple[str, ...] | list[str] | None = None) -> SnapshotRef | None:
    refs = iter_refs(ranks=ranks)
    return refs[-1] if refs else None


def summary() -> dict:
    """Archive health, for the menu's status header and the data-health screen."""
    refs = iter_refs()
    if not refs:
        return {"snapshots": 0, "ranks": [], "first": None, "last": None, "days": 0, "per_rank": {}, "heroes": 0}

    per_rank: dict[str, int] = {}
    for r in refs:
        per_rank[r.rank] = per_rank.get(r.rank, 0) + 1

    dates = sorted({r.date for r in refs})
    try:
        heroes = int(read_snapshot(refs[-1])["hero"].nunique())
    except Exception:
        heroes = 0

    return {
        "snapshots": len(refs),
        "ranks": sorted(per_rank, key=lambda r: RANKS.index(r) if r in RANKS else 99),
        "per_rank": per_rank,
        "first": dates[0],
        "last": dates[-1],
        "days": len(dates),
        "span_days": (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days + 1,
        "heroes": heroes,
        "legacy": sum(1 for r in refs if r.store == "daily"),
        "archive": sum(1 for r in refs if r.store == "archive"),
    }
