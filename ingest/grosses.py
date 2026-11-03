"""Tidy Tuesday Broadway grosses (1985-06 to 2020-03) -> two staging CSVs.

    uv run ingest/grosses.py

The file has no production IDs: a row is (show name, theater, week). So this script can't fill
the schema's production_id yet. It writes:

  data/csv/grosses_weekly.csv   one row per show per week, in the schema's column names and units,
                                with a provisional run_id instead of production_id
  data/csv/grosses_runs.csv     one row per run: show, first and last week, theaters, still open
                                at the data's cutoff. link_shows.py matches these to IBDB.

A "run" is one stretch of consecutive weeks for a show. A gap of more than RUN_GAP_WEEKS starts a new
one, which is how a revival of 'Chicago' ends up apart from the original. It is a candidate, not the
truth: link_shows.py reconciles runs with IBDB's opening and closing dates, and a hand-fix file
overrides both. build_db.py joins weekly rows to production_id through the run_id.

The later scraper for 2020-to-now rows should write the same columns to the same file.
"""

import io
import re

import pandas as pd

from http_cache import ROOT, fetch  # the CSV goes through the cache too, like every other source

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-28/grosses.csv"
# What a citation points at. The dataset credits Playbill as its origin.
SOURCE_URL = "https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28"

OUT = ROOT / "data" / "csv"

# Shows go dark for a week or two (holidays, strikes) and a few for months. 174 of the
# 47k week-to-week gaps are over 26 weeks; below that they are almost all 2-5 week hiatuses.
RUN_GAP_WEEKS = 26


def load() -> pd.DataFrame:
    text = fetch(URL)
    return pd.read_csv(io.StringIO(text), na_values=["NA"])


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def to_schema(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename to the schema's columns and fix the units."""
    df = pd.DataFrame(
        {
            "show": raw["show"].str.strip(),
            "theatre": raw["theatre"].str.strip(),
            "week_ending": raw["week_ending"],  # already ISO 'YYYY-MM-DD'
            "gross": raw["weekly_gross"],
            "potential_gross": raw["potential_gross"],  # NA for all of the 1980s and most of the 1990s
            "attendance": raw["seats_sold"],
            # seats_in_theatre is the theater's size. Seats on sale that week = size x all performances,
            # previews included. This reproduces the file's own pct_capacity for 99.9% of rows.
            "seats_available": raw["seats_in_theatre"] * (raw["performances"] + raw["previews"]),
            "capacity_pct": (raw["pct_capacity"] * 100).round(2),  # 0-1 in the file, 0-100 in the schema
            "avg_ticket": raw["avg_ticket_price"],
            "top_ticket": raw["top_ticket_price"],  # NA for most of the pre-2000s
            "performances": raw["performances"],  # regular performances; previews are counted apart
            "previews": raw["previews"],
            "source_url": SOURCE_URL,
        }
    )
    return df


def assign_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Add run_id: '<show-slug>@<first week of the run>'."""
    df = df.sort_values(["show", "week_ending"]).reset_index(drop=True)
    week = pd.to_datetime(df["week_ending"])
    gap_weeks = week.groupby(df["show"]).diff().dt.days / 7
    new_run = gap_weeks.isna() | (gap_weeks > RUN_GAP_WEEKS)  # NaN: the show's first row
    run_number = new_run.groupby(df["show"]).cumsum()
    first_week = df.groupby(["show", run_number])["week_ending"].transform("min")
    df["run_id"] = df["show"].map(slug) + "@" + first_week
    return df


def summarize_runs(df: pd.DataFrame) -> pd.DataFrame:
    cutoff = df["week_ending"].max()
    runs = (
        df.groupby("run_id")
        .agg(
            show=("show", "first"),
            first_week=("week_ending", "min"),
            last_week=("week_ending", "max"),
            weeks=("week_ending", "count"),
            theatres=("theatre", lambda s: "; ".join(dict.fromkeys(s))),  # in order, no repeats
        )
        .reset_index()
    )
    # At the cutoff we can't tell "closed" from "still open": Broadway shut down on 2020-03-12.
    runs["open_at_cutoff"] = (runs["last_week"] == cutoff).astype(int)
    return runs.sort_values(["show", "first_week"])


def validate(df: pd.DataFrame) -> None:
    assert not df.duplicated(["run_id", "week_ending"]).any(), "a run has two rows for one week"
    sundays = pd.to_datetime(df["week_ending"]).dt.dayofweek == 6
    assert sundays.all(), f"{(~sundays).sum()} week_ending dates are not Sundays"
    assert df["capacity_pct"].between(0, 200).all(), "capacity_pct outside 0-200: a unit problem?"


def main() -> None:
    weekly = assign_runs(to_schema(load()))
    validate(weekly)
    runs = summarize_runs(weekly)

    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id", "show", "theatre", "week_ending", "gross", "potential_gross", "attendance",
        "seats_available", "capacity_pct", "avg_ticket", "top_ticket", "performances", "previews", "source_url",
    ]
    weekly[columns].to_csv(OUT / "grosses_weekly.csv", index=False)
    runs.to_csv(OUT / "grosses_runs.csv", index=False)

    # Things to know before trusting a trend, printed so they aren't a surprise later
    no_shows = (weekly["performances"] + weekly["previews"]) == 0
    zero_with_shows = (weekly["gross"] == 0) & ~no_shows
    print(f"{len(weekly):,} weekly rows, {weekly['show'].nunique():,} titles, {len(runs):,} runs")
    print(f"weeks {weekly['week_ending'].min()} to {weekly['week_ending'].max()}; {int(runs['open_at_cutoff'].sum())} runs open at the cutoff")
    print(f"{int(no_shows.sum()):,} rows have no performances and no previews (dark weeks): filter them out of trends")
    print(f"{int(zero_with_shows.sum())} rows have zero gross but performances: check them in the CSV")
    print(f"potential_gross missing in {weekly['potential_gross'].isna().mean():.0%} of rows")
    print("wrote", OUT / "grosses_weekly.csv", "and", OUT / "grosses_runs.csv")


if __name__ == "__main__":
    main()