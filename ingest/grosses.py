"""Weekly Broadway grosses, 1985-06 to 2023-03, from two Playbill-derived datasets -> two staging CSVs.

    uv run ingest/grosses.py

  1. Tidy Tuesday (2020-04-28): weeks ending 1985-06-09 to 2020-03-01.
  2. IST719 BroadwayGrosses (github.com/XimengDeng/IST719_BroadwayGrosses, a 2023 Syracuse class project
     that scraped Playbill): used only for the weeks after Tidy Tuesday ends, 2020-03-08 and then the
     reopening, 2021-08 to 2023-03-12. Where the two overlap (2018-08 to 2020-03, 2,557 show-weeks) every
     gross, attendance, ticket price and performance count agrees exactly. It has no potential gross, so
     gross-to-potential stops in 2020. Its show_type (musical/play/special) is kept for the runs it covers.

The files have no production IDs: a row is (show name, theater, week). So this script can't fill
the schema's production_id. It writes:

  data/csv/grosses_weekly.csv   one row per show per week, in the schema's column names and units,
                                with a provisional run_id instead of production_id
  data/csv/grosses_runs.csv     one row per run: show, first and last week, theaters, still open
                                at the data's cutoff, and category where IST719 gives one.

A "run" is one stretch of consecutive weeks for a show. A gap of more than RUN_GAP_WEEKS starts a new
one, which is how a revival of 'Chicago' ends up apart from the original. The COVID shutdown is the
exception: a show playing when Broadway closed in March 2020 that came back after it (Hamilton, Wicked,
Company in previews) continues the same run across the 18-month gap. It is a candidate, not the truth:
build_db.py makes one production per run, and joins weekly rows to production_id through the run_id.

A later source for 2023-to-now rows should write the same columns to the same file, and only for weeks
after the last one here.
"""

import io
import re

import pandas as pd

from http_cache import ROOT, fetch  # the CSV goes through the cache too, like every other source
from titles import fold

IST_FIRST_WEEK = "2018-08-05"  # the first week in the IST719 file

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-04-28/grosses.csv"
# What a citation points at. The dataset credits Playbill as its origin.
SOURCE_URL = "https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28"

IST_URL = "https://raw.githubusercontent.com/XimengDeng/IST719_BroadwayGrosses/main/data_201808-202303.csv"
IST_SOURCE_URL = "https://github.com/XimengDeng/IST719_BroadwayGrosses"

# IST719 spells some titles differently from Tidy Tuesday. Titles that fold() to the same string are
# matched automatically ('Moulin Rouge! The Musical' / '...Musical!'); these are the ones that don't.
# Found by listing titles in one source but not the other over the weeks both cover.
IST_TITLE_FIXES = {
    "Harry Potter and the Cursed Child": "Harry Potter and the Cursed Child, Parts One and Two",
    "Swedney Todd": "Sweeney Todd",  # a typo in IST719 (the 2023 revival)
}

# The COVID shutdown: Broadway closed on 2020-03-12 and reopened from 2021-08. A run whose last week
# before the gap is in SHUTDOWN_LAST_WEEKS and that has grosses again by REOPENED_BY is one run, not two.
SHUTDOWN_LAST_WEEKS = ("2020-02-23", "2020-03-15")
REOPENED_BY = "2022-06-30"

OUT = ROOT / "data" / "csv"

# Shows go dark for a week or two (holidays, strikes) and a few for months. 174 of the
# 47k week-to-week gaps are over 26 weeks; below that they are almost all 2-5 week hiatuses.
RUN_GAP_WEEKS = 26


def load() -> pd.DataFrame:
    text = fetch(URL)
    return pd.read_csv(io.StringIO(text), na_values=["NA"])


def load_ist() -> pd.DataFrame:
    text = fetch(IST_URL)
    return pd.read_csv(io.StringIO(text.lstrip("\ufeff")))  # the file starts with a byte-order mark


def _number(col: pd.Series) -> pd.Series:
    """'$1,095,904.55' / '94.59%' / '7,764' -> float."""
    return pd.to_numeric(col.astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")


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


def ist_to_schema(raw: pd.DataFrame, tt_shows: pd.Series) -> pd.DataFrame:
    """IST719 rows in the same columns as to_schema(), with each title spelled as Tidy Tuesday spells it."""
    names = raw["Show_name"].str.strip().str.replace("\u2014", "-").str.replace("\u2013", "-")  # em/en dashes
    names = names.replace(IST_TITLE_FIXES)
    tt_by_fold = {fold(t): t for t in tt_shows.unique()}
    names = names.map(lambda n: tt_by_fold.get(fold(n), n))
    perfs, previews = _number(raw["Perfs"]), _number(raw["Previews"])
    return pd.DataFrame(
        {
            "show": names,
            "theatre": raw["Show_theatre"].str.strip(),
            "week_ending": raw["date"].str.strip(),  # ISO 'YYYY-MM-DD'
            "gross": _number(raw["This_Week_Gross"]),
            "potential_gross": float("nan"),  # not in this dataset
            "attendance": _number(raw["Seats_Sold"]),
            "seats_available": _number(raw["Seats_in_theater"]) * (perfs + previews),
            "capacity_pct": _number(raw["%cap"]).round(2),  # already 0-100 here
            "avg_ticket": _number(raw["Average_ticket"]),
            "top_ticket": _number(raw["Top_ticket"]),
            "performances": perfs,
            "previews": previews,
            "source_url": IST_SOURCE_URL,
            "category": raw["show_type"].str.strip().str.lower(),  # musical / play / special
        }
    )


def combine(tt: pd.DataFrame, ist: pd.DataFrame) -> pd.DataFrame:
    """Tidy Tuesday, then IST719 for the weeks after it ends. IST719's category (musical/play/special)
    is attached to every row of a title it covers, so build_db.py can give it to the runs it overlaps."""
    cutoff = tt["week_ending"].max()
    later = ist[ist["week_ending"] > cutoff]
    category = ist.groupby("show")["category"].agg(lambda s: s.mode().iloc[0])
    out = pd.concat([tt, later.drop(columns="category")], ignore_index=True)
    out["category"] = out["show"].map(category)
    return out


def assign_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Add run_id: '<show-slug>@<first week of the run>'."""
    df = df.sort_values(["show", "week_ending"]).reset_index(drop=True)
    week = pd.to_datetime(df["week_ending"])
    gap_weeks = week.groupby(df["show"]).diff().dt.days / 7
    prev_week = df.groupby("show")["week_ending"].shift(1)
    # Playing when Broadway shut down and back after it: paused, not closed
    paused = prev_week.between(*SHUTDOWN_LAST_WEEKS) & (df["week_ending"] <= REOPENED_BY)
    new_run = gap_weeks.isna() | ((gap_weeks > RUN_GAP_WEEKS) & ~paused)  # NaN: the show's first row
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
    # IST719's category, only for runs it has weeks of (2018-08 on): an older production of the same
    # title isn't necessarily the same kind of show
    covered = df.groupby("run_id")["week_ending"].max() >= IST_FIRST_WEEK
    runs["category"] = runs["run_id"].map(df.groupby("run_id")["category"].first()).where(
        runs["run_id"].map(covered), None
    )
    # At the cutoff we can't tell "closed" from "still open".
    runs["open_at_cutoff"] = (runs["last_week"] == cutoff).astype(int)
    return runs.sort_values(["show", "first_week"])


def validate(df: pd.DataFrame) -> None:
    assert not df.duplicated(["run_id", "week_ending"]).any(), "a run has two rows for one week"
    sundays = pd.to_datetime(df["week_ending"]).dt.dayofweek == 6
    assert sundays.all(), f"{(~sundays).sum()} week_ending dates are not Sundays"
    assert df["capacity_pct"].between(0, 200).all(), "capacity_pct outside 0-200: a unit problem?"


def main() -> None:
    tt = to_schema(load())
    weekly = assign_runs(combine(tt, ist_to_schema(load_ist(), tt["show"])))
    validate(weekly)
    runs = summarize_runs(weekly)

    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id", "show", "theatre", "week_ending", "gross", "potential_gross", "attendance",
        "seats_available", "capacity_pct", "avg_ticket", "top_ticket", "performances", "previews", "source_url",
    ]
    by_source = weekly.groupby("source_url")["week_ending"].agg(["min", "max", "count"])
    for url, r in by_source.iterrows():
        print(f"  {url}: {r['count']:,} rows, {r['min']} to {r['max']}")
    paused = runs[(runs["first_week"] < "2020-03-12") & (runs["last_week"] > "2021-06-01")]
    print(f"{len(paused)} runs continue across the COVID shutdown, e.g. {', '.join(paused['show'].head(6))}")
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