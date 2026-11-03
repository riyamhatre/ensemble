"""CSVs in data/csv/ -> data/broadway.db, rebuilt from scratch every time.

    uv run build_db.py

Only grosses exist so far, so this first version creates *provisional* productions straight from
grosses_runs.csv. When link_shows.py exists it will write a canonical productions.csv (IBDB-backed)
and the PRODUCTIONS step below will read that instead. The rest of the script stays the same.

The database is disposable: delete it and rerun. Fix data in the CSVs or the ingest scripts,
never in the .db file.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

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


def main() -> None:
    weekly = pd.read_csv(CSV / "grosses_weekly.csv")
    runs = pd.read_csv(CSV / "grosses_runs.csv")

    prods = provisional_productions(runs)
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
    con.close()


if __name__ == "__main__":
    main()