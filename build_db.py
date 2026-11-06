"""CSVs in data/csv/ -> data/broadway.db, rebuilt from scratch every time.

    uv run build_db.py

Productions come straight from grosses_runs.csv (one per run of weekly grosses), and Tony nominees,
cast recordings and hand-curated cast changes are linked to them by title and date. There is no
canonical production list behind this: IBDB, the obvious source, doesn't allow automated access. So
type (original/revival) and opening dates stay guesses from the grosses, and broadway_data.py says so.

Cast changes come only from data/csv/cast_changes_manual.csv, written by hand: one row per performer
joining or leaving, each with a source_url. The file is optional; without it the cast table is empty.

The database is disposable: delete it and rerun. Fix data in the CSVs or the ingest scripts,
never in the .db file.
"""

import re
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "ingest"))
from titles import fold  # noqa: E402  (shared with the ingest scripts; broadway_data imports it from here)

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "csv"
DB = ROOT / "data" / "broadway.db"
SCHEMA = ROOT / "schema.sql"

# ---------------------------------------------------------------------------
# THE closed_early RULE. One place, so you can change it and rebuild.
# A production that has closed (not still open at the data's cutoff) after fewer than this many
# weeks of grosses (previews included) counts as having closed early.
# Closed runs in the 1985-2020 data have a median of 15 weeks, because many are limited-run plays
# and special engagements. So this flags a lot at 26. Only productions running from 2018 on have a
# category (musical or play, from IST719), so the rule can't be limited to musicals across the data.
CLOSED_EARLY_WEEKS = 26
# ---------------------------------------------------------------------------


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def provisional_productions(runs: pd.DataFrame) -> pd.DataFrame:
    """One production per grosses run. IDs are '<title-slug>-<year of first week>'."""
    runs = runs.sort_values(["show", "first_week"]).reset_index(drop=True)
    df = pd.DataFrame(
        {
            "run_id": runs["run_id"],
            "title": runs["show"],
            # The first run of a title in this data is called the original, later ones revivals.
            # Wrong for anything that opened before 1985 (the 1996 'Chicago' is a revival). A guess
            # that stays: no source with full production records allows automated access.
            "type": runs.groupby("show").cumcount().map(lambda n: "original" if n == 0 else "revival"),
            "theater": runs["theatres"].str.split("; ").str[0],
            # Not known from grosses alone: the first week may be a preview week, and a run that
            # begins in 1985-06 may have opened years before. So this stays unknown.
            "opening_date": None,
            # musical / play / special, from IST719: only for runs with weeks from 2018-08 on, else unknown
            "category": runs["category"] if "category" in runs else None,
            "closing_date": runs["last_week"].where(runs["open_at_cutoff"] == 0),  # a week ending, not the last show
            "closed_early": None,
        }
    )
    df["closed_early"] = [
        None if open_ else int(weeks < CLOSED_EARLY_WEEKS)
        for open_, weeks in zip(runs["open_at_cutoff"], runs["weeks"])
    ]

    year = runs["first_week"].str[:4]
    base = df["title"].map(slug) + "-" + year
    # Two runs of one title starting in the same year would collide: number the later ones
    n = base.groupby(base).cumcount()
    df["production_id"] = base.where(n == 0, base + "-" + (n + 1).astype(str))
    return df


# ---------------------------------------------------------------------------
# TONY MATCH. One place, so you can change it and rebuild.
# A nominee links to a production when the folded titles are equal and the run was on Broadway
# during that ceremony's season: it started on or before the ceremony and its last grosses week
# is after the previous ceremony, less SEASON_SLACK_WEEKS (a show that opened in May and closed
# before the June ceremony a year earlier still counts). If several runs fit, the latest-starting
# one wins, so a 1997 'Chicago' revival nominee gets the 1996 revival, not an older run.
# Using the previous ceremony instead of "one year before" matters for 2021: that ceremony was in
# September and covered shows that closed at the March 2020 shutdown.
SEASON_SLACK_WEEKS = 8
# ---------------------------------------------------------------------------


def match_awards_provisional(awards: pd.DataFrame, prods: pd.DataFrame, cer: pd.DataFrame) -> pd.Series:
    """production_id for each award row (None if nothing fits), aligned to awards.index."""
    cer = cer.sort_values("ceremony_date").reset_index(drop=True)
    dates = pd.to_datetime(cer["ceremony_date"])
    prev = dates.shift(1).fillna(dates.iloc[0] - pd.Timedelta(days=365))
    season_start = dict(zip(cer["year"], prev - pd.Timedelta(weeks=SEASON_SLACK_WEEKS)))
    ceremony = dict(zip(cer["year"], dates))

    by_title: dict[str, list] = {}
    for p in prods.itertuples():
        by_title.setdefault(fold(p.title), []).append(
            (pd.Timestamp(p.first_week), pd.Timestamp(p.last_week), p.production_id)
        )

    # The grosses data starts 1985-06-09, a week after the 1985 ceremony. A run already open in the
    # data's first week may have started any time before, so it passes the "started by the ceremony" test.
    data_start = prods["first_week"].min()

    out = {}
    for i, a in awards.iterrows():
        fits = [
            (first, pid)
            for first, last, pid in by_title.get(fold(a["show_title"]), [])
            if (first <= ceremony[a["year"]] or first == pd.Timestamp(data_start))
            and last >= season_start[a["year"]]
        ]
        out[i] = max(fits)[1] if fits else None
    return pd.Series(out)


# ---------------------------------------------------------------------------
# RECORDING MATCH. One place, so you can change it and rebuild.
# A cast recording links to a production of the same (folded) title when its cast_year (from the
# MusicBrainz disambiguation, e.g. '1996 Broadway revival cast', else the release year) is within
# RECORDING_SLACK_YEARS of the run: albums are often recorded in previews, or come out after closing.
# If several runs fit, the one that started closest to cast_year wins. Recordings of productions
# that closed before the grosses data starts (the 1975 'Chicago') match nothing and are left out.
RECORDING_SLACK_YEARS = 1
# ---------------------------------------------------------------------------


def match_recordings_provisional(rec: pd.DataFrame, prods: pd.DataFrame) -> pd.Series:
    """production_id for each recording row (None if nothing fits), aligned to rec.index."""
    by_title: dict[str, list] = {}
    for p in prods.itertuples():
        by_title.setdefault(fold(p.title), []).append((int(p.first_week[:4]), int(p.last_week[:4]), p.production_id))
    out = {}
    for i, r in rec.iterrows():
        year = r["cast_year"]
        fits = [] if pd.isna(year) else [
            (abs(first - year), -first, pid)
            for first, last, pid in by_title.get(fold(r["show_title"]), [])
            if first - RECORDING_SLACK_YEARS <= year <= last + RECORDING_SLACK_YEARS
        ]
        out[i] = min(fits)[2] if fits else None
    return pd.Series(out, dtype=object)


# ---------------------------------------------------------------------------
# CAST MATCH. One place, so you can change it and rebuild.
# A hand-curated cast change links to the production of the same (folded) title whose grosses run
# includes its start_date, give or take CAST_SLACK_WEEKS (a star announced for opening night may start
# in previews; a start_date can fall in a dark week). If several runs fit, the closest one wins.
# A row that matches nothing stops the build: it's hand-written, so it's a typo to fix, not data to drop.
CAST_SLACK_WEEKS = 2
CAST_FILE = CSV / "cast_changes_manual.csv"
CAST_COLUMNS = ["show_title", "person", "role", "start_date", "end_date", "is_star", "source_url"]
# ---------------------------------------------------------------------------


def load_cast_changes(prods: pd.DataFrame) -> pd.DataFrame:
    """cast_changes_manual.csv -> rows for the cast_changes table, each with its production_id."""
    cast = pd.read_csv(CAST_FILE, dtype=str).fillna("")
    missing = [c for c in CAST_COLUMNS if c not in cast.columns]
    assert not missing, f"{CAST_FILE.name} is missing columns {missing}; it needs {CAST_COLUMNS}"

    by_title: dict[str, list] = {}
    for p in prods.itertuples():
        by_title.setdefault(fold(p.title), []).append(
            (pd.Timestamp(p.first_week), pd.Timestamp(p.last_week), p.production_id)
        )
    slack = pd.Timedelta(weeks=CAST_SLACK_WEEKS)

    problems, out = [], []
    for i, r in cast.iterrows():
        line = f"{CAST_FILE.name} line {i + 2} ({r['person']} in {r['show_title']})"  # +2: header, 1-based
        row_problems = [f"{line}: {col} is empty"
                        for col in ("show_title", "person", "start_date", "source_url") if not r[col].strip()]
        try:
            # Strictly YYYY-MM-DD: pandas would also take 'July 2000' and quietly pick July 1
            start = pd.Timestamp(date.fromisoformat(r["start_date"].strip()))
            end = pd.Timestamp(date.fromisoformat(r["end_date"].strip())) if r["end_date"].strip() else None
        except ValueError:
            problems += row_problems + [
                f"{line}: dates must be YYYY-MM-DD, got {r['start_date']!r} / {r['end_date']!r}"]
            continue
        if end is not None and end < start:
            row_problems.append(f"{line}: end_date is before start_date")
        if r["is_star"].strip() not in ("", "0", "1"):
            row_problems.append(f"{line}: is_star must be 0, 1 or empty, got {r['is_star']!r}")
        if row_problems:
            problems += row_problems
            continue

        runs = by_title.get(fold(r["show_title"]), [])
        fits = [(0 if first <= start <= last else min(abs(start - first), abs(start - last)), pid)
                for first, last, pid in runs if first - slack <= start <= last + slack]
        if not fits:
            spans = ", ".join(f"{pid} ({first.date()} to {last.date()})" for first, last, pid in runs) or "no production"
            problems.append(f"{line}: no run of that title includes {start.date()}. Title matches: {spans}")
            continue
        pid = min(fits)[1]
        first_week = next(first for first, _, p in runs if p == pid)
        out.append({
            "production_id": pid,
            "person": r["person"].strip(),
            "role": r["role"].strip() or None,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat() if end is not None else None,
            # Someone starting more than CAST_SLACK_WEEKS after the run's first week replaced someone
            "is_replacement": int(start > first_week + slack),
            "is_star": int(r["is_star"]) if r["is_star"].strip() else None,
            "source_url": r["source_url"].strip(),
        })
    assert not problems, "fix these rows in the cast file:\n  " + "\n  ".join(problems)
    return pd.DataFrame(out, columns=["production_id", "person", "role", "start_date", "end_date",
                                      "is_replacement", "is_star", "source_url"])


def main() -> None:
    weekly = pd.read_csv(CSV / "grosses_weekly.csv")
    runs = pd.read_csv(CSV / "grosses_runs.csv")
    cer = pd.read_csv(CSV / "tony_ceremonies.csv")
    awards = pd.read_csv(CSV / "tony_awards.csv")

    prods = provisional_productions(runs)
    prods = prods.merge(runs[["run_id", "first_week", "last_week"]], on="run_id")  # for the Tony match only
    run_to_prod = dict(zip(prods["run_id"], prods["production_id"]))

    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite; without it REFERENCES checks nothing
    con.executescript(SCHEMA.read_text())

    # productions
    cols = ["production_id", "title", "type", "category", "theater", "opening_date", "closing_date", "closed_early"]
    prods[cols].to_sql("productions", con, if_exists="append", index=False)

    # aliases: the grosses file's own spelling of each title
    aliases = prods[["title", "production_id"]].rename(columns={"title": "alias"})
    aliases["source"] = "grosses"
    aliases["match_method"] = "exact"
    aliases.to_sql("aliases", con, if_exists="append", index=False)

    # grosses
    g = weekly.copy()
    g["production_id"] = g["run_id"].map(run_to_prod)
    assert g["production_id"].notna().all(), "weekly rows with a run_id that has no production"
    gcols = [
        "production_id", "week_ending", "gross", "potential_gross", "attendance", "seats_available",
        "capacity_pct", "avg_ticket", "top_ticket", "performances", "previews", "source_url",
    ]
    g[gcols].to_sql("grosses", con, if_exists="append", index=False)

    con.execute(
        "INSERT INTO data_sources (source, description, covers_from, covers_to, fetched_at, notes) "
        "VALUES ('grosses', 'Weekly Broadway grosses, from Playbill via two datasets: Tidy Tuesday (to 2020-03-01) "
        "and IST719 BroadwayGrosses (2020-03-08 on)', ?, ?, date('now'), "
        "'No grosses during the COVID shutdown: Broadway closed 2020-03-12 and shows reopened from August 2021, "
        "so there are no weeks between 2020-03-08 and 2021-08. Shows playing at the shutdown that reopened "
        "continue as one production. potential_gross (and so gross-to-potential) is missing in the 1980s, most "
        "of the 1990s, and from 2020 on. Broadway only: tours are not included. Preview weeks are in the data.')",
        (g["week_ending"].min(), g["week_ending"].max()),
    )

    # Tonys. Ceremonies first: awards.year references them.
    cer[["year", "ceremony_number", "ceremony_date", "source_url"]].to_sql(
        "tony_ceremonies", con, if_exists="append", index=False
    )
    awards["production_id"] = match_awards_provisional(awards, prods, cer)
    acols = ["year", "category", "nominee", "production_id", "won", "source_url"]
    awards[acols].to_sql("awards", con, if_exists="append", index=False)

    tony_alias = awards.dropna(subset=["production_id"]).drop_duplicates(["show_title", "production_id"])
    tony_alias = pd.DataFrame(
        {"alias": tony_alias["show_title"], "source": "tonys", "production_id": tony_alias["production_id"],
         "match_method": "exact"}  # exact after case/accent/punctuation folding, see fold()
    )
    # aliases is COLLATE NOCASE, so two spellings differing only in case collide on the primary key
    tony_alias = tony_alias[~tony_alias.assign(a=tony_alias["alias"].str.lower()).duplicated(["a", "production_id"])]
    tony_alias.to_sql("aliases", con, if_exists="append", index=False)

    con.execute(
        "INSERT INTO data_sources (source, description, covers_from, covers_to, fetched_at, notes) "
        "VALUES ('tonys', 'Tony Award winners and nominees, competitive categories (Wikipedia ceremony pages)', ?, ?, "
        "date('now'), 'Ceremonies 1985-2026 (no 2020 ceremony; the 2021 one was in September). Special Tonys not included. "
        "Nominees are linked to productions by a provisional title-and-season match, so some have no production.')",
        (cer["ceremony_date"].min(), cer["ceremony_date"].max()),
    )

    # Cast recordings and tracks (MusicBrainz)
    rec = pd.read_csv(CSV / "recordings.csv")
    tracks = pd.read_csv(CSV / "tracks.csv")
    rec["production_id"] = match_recordings_provisional(rec, prods)
    linked = rec.dropna(subset=["production_id"])
    pd.DataFrame(
        {"recording_id": linked["mbid"], "production_id": linked["production_id"], "title": linked["title"],
         "year": linked["cast_year"].astype("Int64"), "mbid": linked["mbid"], "source_url": linked["source_url"]}
    ).to_sql("recordings", con, if_exists="append", index=False)
    tr = tracks[tracks["mbid"].isin(linked["mbid"])].rename(columns={"mbid": "recording_id"})
    tr[["recording_id", "disc", "position", "title"]].to_sql("tracks", con, if_exists="append", index=False)
    con.execute(
        "INSERT INTO data_sources (source, description, covers_from, covers_to, fetched_at, notes) "
        "VALUES ('musicbrainz', 'Broadway cast recordings and track lists (MusicBrainz, CC0)', NULL, NULL, date('now'), "
        "'Only recordings MusicBrainz labels as a Broadway cast, of productions in the grosses data. "
        "Track lists are of one official release per recording, so bonus tracks on other editions may be missing. "
        "Off-Broadway, London and tour recordings are not included.')"
    )

    # Cast changes (hand-curated; optional)
    n_cast = 0
    if CAST_FILE.exists():
        cast = load_cast_changes(prods)
        cast.to_sql("cast_changes", con, if_exists="append", index=False)
        n_cast = len(cast)
        if n_cast:
            con.execute(
                "INSERT INTO data_sources (source, description, covers_from, covers_to, fetched_at, notes) "
                "VALUES ('cast', 'Cast changes: performers joining or leaving, hand-curated from news reports and "
                "Wikipedia (each row has its own source_url)', ?, ?, date('now'), ?)",
                (cast["start_date"].min(), cast["start_date"].max(),
                 f"Only {n_cast} changes in {cast['production_id'].nunique()} productions, chosen by hand: most "
                 "productions have none recorded, and a missing change doesn't mean it didn't happen."),
            )
    con.commit()

    # Checks. Fail loudly rather than ship a database that quietly disagrees with its CSVs.
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    assert not bad, f"foreign key violations: {bad[:5]}"
    n_g = con.execute("SELECT COUNT(*) FROM grosses").fetchone()[0]
    assert n_g == len(weekly), f"grosses rows {n_g} != CSV rows {len(weekly)}"

    n_p = con.execute("SELECT COUNT(*) FROM productions").fetchone()[0]
    flagged, closed = con.execute(
        "SELECT SUM(closed_early = 1), COUNT(closed_early) FROM productions"
    ).fetchone()
    print(f"{n_p:,} productions, {n_g:,} gross rows -> {DB}")
    print(f"closed_early: {flagged:,} of {closed:,} closed productions flagged (rule: under {CLOSED_EARLY_WEEKS} weeks)")

    n_a = con.execute("SELECT COUNT(*) FROM awards").fetchone()[0]
    assert n_a == len(awards), f"awards rows {n_a} != CSV rows {len(awards)}"
    # Grosses end in March 2023, so ceremonies through 2023 can match (for shows running by then).
    # Later ones wait for new grosses.
    for label, where in (("1985-2023", "year <= 2023"), ("2024-2026", "year > 2023")):
        linked, total = con.execute(
            f"SELECT COUNT(production_id), COUNT(*) FROM awards WHERE {where}"
        ).fetchone()
        print(f"awards {label}: {linked:,} of {total:,} nominee rows linked to a production ({linked / total:.1%})")

    n_r, n_t = con.execute("SELECT (SELECT COUNT(*) FROM recordings), (SELECT COUNT(*) FROM tracks)").fetchone()
    assert n_t == len(tr), f"tracks rows {n_t} != linked CSV rows {len(tr)}"
    print(f"recordings: {n_r:,} of {len(rec):,} linked to a production, {n_t:,} tracks "
          f"(unlinked are mostly casts from before the grosses data)")

    n_c = con.execute("SELECT COUNT(*) FROM cast_changes").fetchone()[0]
    assert n_c == n_cast, f"cast_changes rows {n_c} != linked CSV rows {n_cast}"
    print(f"cast changes: {n_c:,}" + ("" if CAST_FILE.exists() else f" ({CAST_FILE.name} not found: optional)"))
    con.close()


if __name__ == "__main__":
    main()