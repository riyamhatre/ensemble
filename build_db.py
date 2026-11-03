"""CSVs in data/csv/ -> data/broadway.db, rebuilt from scratch every time.

    uv run build_db.py

Only grosses and Tonys exist so far, so this version creates *provisional* productions straight from
grosses_runs.csv, and links Tony nominees to them with a provisional title-and-date match
(match_awards_provisional). When link_shows.py exists it will write a canonical productions.csv
(IBDB-backed) and the award links, and both provisional steps below go away. The rest stays the same.

The database is disposable: delete it and rerun. Fix data in the CSVs or the ingest scripts,
never in the .db file.
"""

import re
import sqlite3
import sys
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
# and special engagements. So this flags a lot at 26. Once IBDB gives each production a category,
# apply the rule to musicals only, and pick the number from their distribution.
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
            # Wrong for anything that opened before 1985 (the 1996 'Chicago' is a revival).
            # link_shows.py replaces this from IBDB.
            "type": runs.groupby("show").cumcount().map(lambda n: "original" if n == 0 else "revival"),
            "theater": runs["theatres"].str.split("; ").str[0],
            # Not known from grosses alone: the first week may be a preview week, and a run that
            # begins in 1985-06 may have opened years before. IBDB fills these.
            "opening_date": None,
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
# PROVISIONAL TONY MATCH. One place; link_shows.py replaces it.
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
# PROVISIONAL RECORDING MATCH. One place; link_shows.py replaces it.
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
    cols = ["production_id", "title", "type", "theater", "opening_date", "closing_date", "closed_early"]
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
        "VALUES ('grosses', 'Weekly Broadway grosses (Tidy Tuesday dataset, credited to Playbill)', ?, ?, "
        "date('now'), 'Ends at the 2020 shutdown. potential_gross missing in the 1980s and most of the 1990s. "
        "Broadway only: tours are not included. Preview weeks are in the data.')",
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
    # Grosses end in March 2020, so only ceremonies through 2021 can match. Later ones wait for new grosses.
    for label, where in (("1985-2021", "year <= 2021"), ("2022-2026", "year > 2021")):
        linked, total = con.execute(
            f"SELECT COUNT(production_id), COUNT(*) FROM awards WHERE {where}"
        ).fetchone()
        print(f"awards {label}: {linked:,} of {total:,} nominee rows linked to a production ({linked / total:.1%})")

    n_r, n_t = con.execute("SELECT (SELECT COUNT(*) FROM recordings), (SELECT COUNT(*) FROM tracks)").fetchone()
    assert n_t == len(tr), f"tracks rows {n_t} != linked CSV rows {len(tr)}"
    print(f"recordings: {n_r:,} of {len(rec):,} linked to a production, {n_t:,} tracks "
          f"(unlinked are mostly casts from before the grosses data)")
    con.close()


if __name__ == "__main__":
    main()