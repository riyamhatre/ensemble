"""Query functions over data/broadway.db: the only way agents see the data.

    from broadway_data import find_production, get_grosses, grosses_around_date, get_awards, data_coverage
    from broadway_data import trouble_signals, tony_effect, get_recordings, compare_recordings, get_cast_changes

Every function returns a JSON string, and catches its own exceptions into {"error": ...}, because the
model can't see a Python exception (app.py wraps these as tools). Every fact carries a source_url so the
Writer can cite it. All arithmetic (averages, percent changes, ratios) happens here, never in the model.

The database is opened read-only. Rebuild it with `uv run build_db.py`.

Smoke test: uv run broadway_data.py
"""

import difflib
import functools
import json
import re
import sqlite3
import statistics
from datetime import date, timedelta
from pathlib import Path

from build_db import fold  # same title folding the Tony match uses, so the two agree

DB = Path(__file__).resolve().parent / "data" / "broadway.db"

# More weekly rows than this and get_grosses returns monthly totals instead: a 30-year run is
# 1,600+ weeks, which would fill the model's context with numbers it can't use.
MAX_WEEKS = 156
MAX_CANDIDATES = 10
MAX_AWARDS = 300

# A "dark" week has no performances and no previews (59 rows). It would read as a gross of zero,
# so every trend here leaves it out.
NOT_DARK = "(g.performances + g.previews) > 0"

# Said with every production: these values are guesses from the grosses data (build_db.py), since no
# source with full production records allows automated access (IBDB doesn't)
PROVISIONAL_NOTE = (
    "type (original/revival) is a guess from the grosses data, which starts in 1985: a revival of a show "
    "first staged before 1985 shows as 'original'. opening_date is unknown; first_week may be a preview week."
)

# trouble_signals: a show's change counts as a warning sign when it trails the market by this many points
TROUBLE_GROSS_VS_MARKET = -10.0  # average weekly gross
TROUBLE_TICKET_VS_MARKET = -5.0  # average ticket price: a show cutting prices faster than the market
# Closed productions in this many years before the week make the "how shows look before closing" benchmark
TROUBLE_BENCHMARK_YEARS = 5


def _connect() -> sqlite3.Connection:
    if not DB.exists():
        raise FileNotFoundError(f"{DB} not found. Build it with: uv run build_db.py")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with _connect() as con:
        return [dict(r) for r in con.execute(sql, params)]


def _json_tool(fn):
    """Return JSON, and turn any exception into {"error": ...} that the model can read and act on."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> str:
        try:
            return json.dumps(fn(*args, **kwargs), default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    return wrapper


def _r(x, nd=2):
    return None if x is None else round(x, nd)


def _pct_change(before, after):
    if before in (None, 0) or after is None:
        return None
    return round(100 * (after - before) / before, 1)


def _production(production_id: str) -> dict:
    rows = _rows(
        """SELECT p.production_id, p.title, p.type, p.category, p.theater, p.opening_date, p.closing_date,
                  MIN(g.week_ending) AS first_week, MAX(g.week_ending) AS last_week, COUNT(g.week_ending) AS weeks
           FROM productions p LEFT JOIN grosses g USING (production_id)
           WHERE p.production_id = ? GROUP BY p.production_id""",
        (production_id,),
    )
    if not rows:
        raise KeyError(f"no production '{production_id}'. Call find_production to get a valid production_id.")
    return rows[0]


def _source(name: str) -> dict:
    rows = _rows("SELECT * FROM data_sources WHERE source = ?", (name,))
    return rows[0] if rows else {}


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


@_json_tool
def find_production(name: str, year: int | None = None) -> dict:
    """Productions whose title (or any source's alias for it) matches name, best match first.

    year keeps only productions with grosses in that calendar year. Several candidates with one title
    are different runs (an original and its revivals): ask the user which one if the question doesn't say.
    """
    want = fold(name)
    titles = _rows(
        """SELECT production_id, title AS alias FROM productions
           UNION SELECT production_id, alias FROM aliases"""
    )
    by_fold: dict[str, set[str]] = {}
    for t in titles:
        by_fold.setdefault(fold(t["alias"]), set()).add(t["production_id"])

    # Rank: exact folded title, then title containing the query, then close spellings (typos)
    ranked: list[tuple[int, str]] = []
    for f, ids in by_fold.items():
        if f == want:
            rank = 0
        elif want and f" {want} " in f" {f} ":  # whole words: 'rent' matches 'rent', not 'parents'
            rank = 1
        else:
            continue
        ranked += [(rank, pid) for pid in ids]
    if not ranked:
        close = difflib.get_close_matches(want, by_fold.keys(), n=5, cutoff=0.75)
        ranked = [(2, pid) for f in close for pid in by_fold[f]]

    best: dict[str, int] = {}
    for rank, pid in ranked:
        best[pid] = min(rank, best.get(pid, 9))
    # Within a rank, the closer spelling first ('Hamiltn' -> Hamilton before Hamlet), then the older run
    similarity = {
        pid: max(difflib.SequenceMatcher(None, want, f).ratio() for f, ids in by_fold.items() if pid in ids)
        for pid in best
    }

    coverage_end = _source("grosses").get("covers_to")
    candidates = []
    for pid, rank in best.items():
        p = _production(pid)
        if year is not None and not (p["first_week"] and p["first_week"][:4] <= str(year) <= p["last_week"][:4]):
            continue
        wins, noms = _rows(
            "SELECT COALESCE(SUM(won), 0) AS w, COUNT(*) AS n FROM awards WHERE production_id = ?", (pid,)
        )[0].values()
        candidates.append(
            {
                **p,
                "running_at_data_end": p["last_week"] == coverage_end,
                "tony_wins": wins,
                "tony_nominations": noms,
                "match": ["exact", "partial", "close spelling"][rank],
            }
        )
    candidates.sort(
        key=lambda c: (["exact", "partial", "close spelling"].index(c["match"]),
                       -round(similarity[c["production_id"]], 2), c["first_week"] or "")
    )

    return {
        "query": name,
        "year": year,
        "candidates": candidates[:MAX_CANDIDATES],
        "more": max(0, len(candidates) - MAX_CANDIDATES),
        "notes": [
            "Broadway productions only: touring and off-Broadway productions are not in this database.",
            f"running_at_data_end means it had grosses in the last week of data ({coverage_end}), "
            "not that it is running today.",
            PROVISIONAL_NOTE,
        ],
        "source_url": _rows("SELECT source_url FROM grosses LIMIT 1")[0]["source_url"],
    }


# ---------------------------------------------------------------------------
# Grosses
# ---------------------------------------------------------------------------


def grosses_rows(production_id: str, start: str | None = None, end: str | None = None) -> list[dict]:
    """Weekly rows, dark weeks excluded, oldest first. Plain Python (no JSON) so app.py can chart them."""
    rows = _rows(
        f"""SELECT g.week_ending, g.gross, g.potential_gross, g.attendance, g.seats_available, g.capacity_pct,
                   g.avg_ticket, g.top_ticket, g.performances, g.previews, g.source_url
            FROM grosses g
            WHERE g.production_id = ? AND {NOT_DARK}
              AND g.week_ending >= COALESCE(?, '0000') AND g.week_ending <= COALESCE(?, '9999')
            ORDER BY g.week_ending""",
        (production_id, start, end),
    )
    for r in rows:
        r["preview_week"] = r["performances"] == 0  # previews only: the show hadn't opened yet
        r["gross_to_potential"] = (
            _r(r["gross"] / r["potential_gross"], 3) if r["gross"] is not None and r["potential_gross"] else None
        )
    return rows


def _window_stats(rows: list[dict]) -> dict:
    """Totals-based averages, so a big week counts for more than a small one."""
    gross = sum(r["gross"] or 0 for r in rows)
    att = sum(r["attendance"] or 0 for r in rows)
    seats = sum(r["seats_available"] or 0 for r in rows)
    with_pot = [r for r in rows if r["potential_gross"] and r["gross"] is not None]
    return {
        "weeks": len(rows),
        "preview_weeks": sum(r["preview_week"] for r in rows),
        "avg_weekly_gross": _r(gross / len(rows)) if rows else None,
        "capacity_pct": _r(100 * att / seats, 1) if seats else None,
        "avg_ticket": _r(gross / att) if att else None,
        "gross_to_potential": (
            _r(sum(r["gross"] for r in with_pot) / sum(r["potential_gross"] for r in with_pot), 3) if with_pot else None
        ),
        "weeks_with_potential": len(with_pot),
        "avg_potential_gross": _r(sum(r["potential_gross"] for r in with_pot) / len(with_pot)) if with_pot else None,
    }


@_json_tool
def get_grosses(production_id: str, start: str | None = None, end: str | None = None) -> dict:
    """Weekly box office for one production between start and end (ISO dates, both optional).

    Over MAX_WEEKS weeks it returns monthly totals instead: narrow start/end to get weekly rows.
    """
    p = _production(production_id)
    rows = grosses_rows(production_id, start, end)
    out = {
        "production": p,
        "start": start,
        "end": end,
        "summary": _window_stats(rows),
        "notes": [
            "Dark weeks (no performances or previews) are left out.",
            "capacity_pct is 0-100 and can exceed 100 with standing room.",
            "preview_week rows are before opening night and usually sell below a running show.",
            f"Grosses data ends {_source('grosses').get('covers_to')}.",
        ],
    }
    if len(rows) <= MAX_WEEKS:
        out["granularity"] = "week"
        out["rows"] = [{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    else:
        months: dict[str, list[dict]] = {}
        for r in rows:
            months.setdefault(r["week_ending"][:7], []).append(r)
        out["granularity"] = "month"
        out["notes"].append(
            f"{len(rows)} weeks is more than {MAX_WEEKS}, so these are monthly totals by week-ending month. "
            "Narrow start and end for weekly rows."
        )
        out["rows"] = [
            {"month": m, **_window_stats(rs), "total_gross": _r(sum(r["gross"] or 0 for r in rs)),
             "source_url": rs[0]["source_url"]}
            for m, rs in months.items()
        ]
    return out


def _market(before: list[str], after: list[str], exclude: str) -> tuple[dict, dict, int]:
    """Before and after stats for the other shows that played every one of these weeks.

    Same-store on purpose: many shows close right after the Tonys, so the street's total gross drops
    even when demand doesn't. Comparing a fixed set of shows takes that out.
    """
    weeks = before + after
    if not before or not after:
        return _window_stats([]), _window_stats([]), 0
    marks = ",".join("?" * len(weeks))
    rows = _rows(
        f"""WITH stayers AS (
              SELECT g.production_id FROM grosses g
              WHERE g.week_ending IN ({marks}) AND g.production_id != ? AND {NOT_DARK}
              GROUP BY g.production_id HAVING COUNT(*) = ?)
            SELECT g.week_ending, COUNT(*) AS shows, SUM(g.gross) / COUNT(*) AS gross,
                   SUM(g.attendance) AS attendance, SUM(g.seats_available) AS seats_available,
                   NULL AS potential_gross, 0 AS preview_week
            FROM grosses g JOIN stayers USING (production_id)
            WHERE g.week_ending IN ({marks}) GROUP BY g.week_ending""",
        (*weeks, exclude, len(weeks), *weeks),
    )
    n = rows[0]["shows"] if rows else 0
    return (_window_stats([r for r in rows if r["week_ending"] in before]),
            _window_stats([r for r in rows if r["week_ending"] in after]), n)


@_json_tool
def grosses_around_date(production_id: str, date_: str, weeks_before: int = 8, weeks_after: int = 8) -> dict:
    """Box office in the weeks before and after an event (a Tony win, a star joining or leaving).

    Weeks run Monday to Sunday (week_ending is the Sunday). 'before' is the weeks_before weeks ending before
    the date; 'after' is the weeks_after weeks starting after it. The week containing the date is mixed,
    so it is reported on its own as event_week and counted in neither window.

    Also returns the same change for the rest of Broadway in the same weeks. Broadway is seasonal
    (summer and the holidays are strong, January and February weak), so a show's change means little
    until it is compared with the market's.
    """
    return _around(production_id, date_, weeks_before, weeks_after)


def _around(production_id: str, date_: str, weeks_before: int = 8, weeks_after: int = 8) -> dict:
    """grosses_around_date as a plain dict, so tony_effect can run it for many productions."""
    event = date.fromisoformat(date_)
    p = _production(production_id)
    rows = grosses_rows(production_id)
    before = [r for r in rows if date.fromisoformat(r["week_ending"]) < event][-weeks_before:]
    after = [r for r in rows if date.fromisoformat(r["week_ending"]) - timedelta(days=6) > event][:weeks_after]
    event_week = [r for r in rows if r not in before and r not in after
                  and 0 <= (date.fromisoformat(r["week_ending"]) - event).days <= 6]

    show_b, show_a = _window_stats(before), _window_stats(after)
    mkt_b, mkt_a, n_market = _market([r["week_ending"] for r in before], [r["week_ending"] for r in after],
                                      production_id)

    metrics = ("avg_weekly_gross", "capacity_pct", "avg_ticket", "gross_to_potential", "avg_potential_gross")
    change = {m: _pct_change(show_b[m], show_a[m]) for m in metrics}
    market_change = {m: _pct_change(mkt_b[m], mkt_a[m]) for m in ("avg_weekly_gross", "capacity_pct", "avg_ticket")}
    vs_market = {m: (None if change[m] is None or market_change.get(m) is None else round(change[m] - market_change[m], 1))
                 for m in market_change}

    warnings = []
    if show_b["weeks"] < weeks_before:
        warnings.append(f"only {show_b['weeks']} of {weeks_before} weeks before the date have data")
    if show_a["weeks"] < weeks_after:
        warnings.append(f"only {show_a['weeks']} of {weeks_after} weeks after the date have data")
    if show_b["preview_weeks"]:
        warnings.append(f"{show_b['preview_weeks']} 'before' weeks are previews, which sell below a running show")
    if not before or not after:
        warnings.append("one window is empty, so there is no change to report")
    if change["avg_potential_gross"] and abs(change["avg_potential_gross"]) >= 5:
        warnings.append(
            f"potential_gross itself changed {change['avg_potential_gross']:+.1f}% (producers reset it, e.g. after "
            "a price change), so gross_to_potential moves for that reason too: compare avg_weekly_gross instead"
        )

    return {
        "production": p,
        "event_date": date_,
        "before": {"from": before[0]["week_ending"] if before else None,
                   "to": before[-1]["week_ending"] if before else None, **show_b},
        "after": {"from": after[0]["week_ending"] if after else None,
                  "to": after[-1]["week_ending"] if after else None, **show_a},
        "event_week": event_week[0]["week_ending"] if event_week else None,
        "pct_change": change,
        "market_pct_change": market_change,
        "pct_change_vs_market": vs_market,
        "warnings": warnings,
        "market_shows": n_market,
        "notes": [
            f"The market is the {n_market} other shows that played every week of both windows (same-store), "
            "so shows opening or closing in between don't move it.",
            "pct_change_vs_market is the show's percent change minus the market's, in percentage points.",
            "Averages are from totals (total gross / total attendance), not averages of weekly averages.",
        ],
        "source_url": rows[0]["source_url"] if rows else None,
    }


# ---------------------------------------------------------------------------
# Awards
# ---------------------------------------------------------------------------


@_json_tool
def get_awards(
    production_id: str | None = None,
    year: int | None = None,
    category: str | None = None,
    nominee_contains: str | None = None,
    won_only: bool = False,
) -> dict:
    """Tony Award nominations and wins. Give at least one filter.

    category matches part of the name, case-insensitively ('Best Musical' also matches nothing else,
    'Revival' matches every revival category). nominee_contains searches the nominee line, which finds
    nominees that aren't linked to a production (see notes).
    """
    if not any(v is not None for v in (production_id, year, category, nominee_contains)):
        raise ValueError("give production_id, year, category or nominee_contains")
    where, params = ["1=1"], []
    if production_id is not None:
        where.append("a.production_id = ?"); params.append(production_id)
    if year is not None:
        where.append("a.year = ?"); params.append(year)
    if category is not None:
        where.append("a.category LIKE ?"); params.append(f"%{category}%")
    if nominee_contains is not None:
        where.append("a.nominee LIKE ?"); params.append(f"%{nominee_contains}%")
    if won_only:
        where.append("a.won = 1")
    rows = _rows(
        f"""SELECT a.year, c.ceremony_number, c.ceremony_date, a.category, a.nominee, a.won, a.production_id,
                   a.source_url
            FROM awards a JOIN tony_ceremonies c USING (year)
            WHERE {' AND '.join(where)} ORDER BY a.year, a.category, a.won DESC LIMIT ?""",
        (*params, MAX_AWARDS + 1),
    )
    return {
        "awards": rows[:MAX_AWARDS],
        "truncated": len(rows) > MAX_AWARDS,
        "notes": [
            "Competitive categories only; special Tonys are not included. No ceremony was held in 2020.",
            "Nominees are linked to productions by a provisional title-and-season match: about 6% of "
            "1985-2021 nominees and every 2022+ nominee have production_id null. Search by nominee_contains "
            "to find those.",
        ],
    }


# ---------------------------------------------------------------------------
# Market-wide: questions about many shows at once, which no agent can answer one show at a time
# ---------------------------------------------------------------------------


def _weeks_up_to(week_ending: str | None, n: int) -> list[str]:
    """The last n week_endings with any grosses, on or before week_ending (default: the data's last week)."""
    rows = _rows(
        "SELECT DISTINCT week_ending FROM grosses WHERE week_ending <= COALESCE(?, '9999') "
        "ORDER BY week_ending DESC LIMIT ?",
        (week_ending, n),
    )
    return sorted(r["week_ending"] for r in rows)


def _median(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return _r(statistics.median(values), 3) if values else None


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return _r(statistics.mean(values), 1) if values else None


def _market_levels(weeks: list[str], exclude: str = "") -> dict:
    """The whole market's capacity and gross-to-potential over these weeks (every other show playing them),
    from totals. What a show's own levels are measured against, so January doesn't look like trouble."""
    if not weeks:
        return {"capacity_pct": None, "gross_to_potential": None}
    marks = ",".join("?" * len(weeks))
    r = _rows(
        f"""SELECT SUM(g.attendance) AS att, SUM(g.seats_available) AS seats,
                   SUM(CASE WHEN g.potential_gross > 0 THEN g.gross END) AS gross_p,
                   SUM(CASE WHEN g.potential_gross > 0 THEN g.potential_gross END) AS pot
            FROM grosses g WHERE g.week_ending IN ({marks}) AND g.production_id != ? AND {NOT_DARK}""",
        (*weeks, exclude),
    )[0]
    return {
        "capacity_pct": _r(100 * r["att"] / r["seats"], 1) if r["seats"] else None,
        "gross_to_potential": _r(r["gross_p"] / r["pot"], 3) if r["pot"] else None,
    }


def _minus(a, b, nd=1):
    return None if a is None or b is None else round(a - b, nd)


@_json_tool
def trouble_signals(week_ending: str | None = None, weeks: int = 8) -> dict:
    """Which shows running in a given week showed warning signs at the box office: level_signals (selling
    below the market now) and trend_signals (falling faster than the market on a year earlier).

    Broadway is seasonal (January and February are weak, summer and the holidays strong), so every
    measure here is against the rest of the market, never raw. For each show playing that week, over its
    last `weeks` weeks:
      - level signals: its capacity, and its gross-to-potential (~1996 on), minus the market's over the
        same weeks. A show is flagged when it trails the market by more than the median production did in
        its final `weeks` weeks before closing (the benchmark: productions that closed in the
        TROUBLE_BENCHMARK_YEARS years before the week).
      - trend signals, for shows with a year of grosses: average weekly gross and average ticket against
        the same weeks a year earlier, minus the market's year-over-year change (same-store). Flagged at
        TROUBLE_GROSS_VS_MARKET and TROUBLE_TICKET_VS_MARKET points (a falling ticket suggests discounting).
    """
    recent = _weeks_up_to(week_ending, weeks)
    if len(recent) < weeks:
        raise ValueError(f"not enough weeks of data on or before {week_ending}")
    week = recent[-1]
    # The same Sundays a year earlier (52 weeks back), for year-over-year trends
    year_ago = [(date.fromisoformat(w) - timedelta(weeks=52)).isoformat() for w in recent]

    # Benchmark: how far closing productions trailed the market in their final weeks
    start = date.fromisoformat(week) - timedelta(days=365 * TROUBLE_BENCHMARK_YEARS)
    closed = _rows(
        "SELECT production_id FROM productions WHERE closing_date IS NOT NULL AND closing_date > ? AND closing_date <= ?",
        (start.isoformat(), week),
    )
    rel_cap, rel_g2p = [], []
    for c in closed:
        rows = [r for r in grosses_rows(c["production_id"]) if not r["preview_week"]][-weeks:]
        if len(rows) < weeks:
            continue
        s = _window_stats(rows)
        m = _market_levels([r["week_ending"] for r in rows], c["production_id"])
        rel_cap.append(_minus(s["capacity_pct"], m["capacity_pct"]))
        rel_g2p.append(_minus(s["gross_to_potential"], m["gross_to_potential"], 3))
    bench = {
        "closed_productions": len(rel_cap),
        "closed_from": start.isoformat(),
        "closed_to": week,
        "median_capacity_vs_market": _median(rel_cap),  # percentage points
        "median_gross_to_potential_vs_market": _median(rel_g2p),
    }

    mkt_now = _market_levels(recent)
    yb, ya, n_market = _market(year_ago, recent, "")
    market_yoy = {m: _pct_change(yb[m], ya[m]) for m in ("avg_weekly_gross", "avg_ticket")}

    running = _rows(
        f"SELECT g.production_id, p.title FROM grosses g JOIN productions p USING (production_id) "
        f"WHERE g.week_ending = ? AND {NOT_DARK}",
        (week,),
    )
    shows = []
    for s in running:
        pid = s["production_id"]
        rows = grosses_rows(pid, end=week)
        now = _window_stats([r for r in rows if r["week_ending"] in recent])
        then = [r for r in rows if r["week_ending"] in year_ago and not r["preview_week"]]
        mkt = _market_levels(recent, pid)
        cap_vs = _minus(now["capacity_pct"], mkt["capacity_pct"])
        g2p_vs = _minus(now["gross_to_potential"], mkt["gross_to_potential"], 3)

        has_year = len(then) >= weeks / 2
        sb = _window_stats(then)
        gross_yoy = _pct_change(sb["avg_weekly_gross"], now["avg_weekly_gross"]) if has_year else None
        ticket_yoy = _pct_change(sb["avg_ticket"], now["avg_ticket"]) if has_year else None
        gross_vs = _minus(gross_yoy, market_yoy["avg_weekly_gross"])
        ticket_vs = _minus(ticket_yoy, market_yoy["avg_ticket"])

        level, trend = [], []  # selling below the market now / falling faster than the market
        if cap_vs is not None and bench["median_capacity_vs_market"] is not None \
                and cap_vs < bench["median_capacity_vs_market"]:
            level.append(f"capacity {now['capacity_pct']}% is {cap_vs:+.1f} points vs the market's "
                           f"{mkt['capacity_pct']}%, worse than closing shows' median {bench['median_capacity_vs_market']:+.1f}")
        if g2p_vs is not None and bench["median_gross_to_potential_vs_market"] is not None \
                and g2p_vs < bench["median_gross_to_potential_vs_market"]:
            level.append(f"gross-to-potential {now['gross_to_potential']} vs the market's {mkt['gross_to_potential']}, "
                           f"worse than closing shows' median gap {bench['median_gross_to_potential_vs_market']:+.3f}")
        if gross_vs is not None and gross_vs <= TROUBLE_GROSS_VS_MARKET:
            trend.append(f"average weekly gross {gross_yoy:+.1f}% on a year earlier vs the market's "
                           f"{market_yoy['avg_weekly_gross']:+.1f}%")
        if ticket_vs is not None and ticket_vs <= TROUBLE_TICKET_VS_MARKET:
            trend.append(f"average ticket {ticket_yoy:+.1f}% on a year earlier vs the market's "
                           f"{market_yoy['avg_ticket']:+.1f}% (possible discounting)")

        shows.append({
            "production_id": pid,
            "title": s["title"],
            "weeks_reported": len(rows),
            "preview_weeks_in_window": now["preview_weeks"],
            "capacity_pct": now["capacity_pct"],
            "capacity_vs_market": cap_vs,
            "gross_to_potential": now["gross_to_potential"],
            "gross_to_potential_vs_market": g2p_vs,
            "avg_weekly_gross": now["avg_weekly_gross"],
            "gross_pct_change_year_on_year": gross_yoy,
            "gross_yoy_vs_market": gross_vs,
            "ticket_yoy_vs_market": ticket_vs,
            "level_signals": level,
            "trend_signals": trend,
        })
    # Level signals first: a show selling below the market now is the stronger warning
    shows.sort(key=lambda s: (-len(s["level_signals"]), -len(s["trend_signals"]),
                              s["capacity_vs_market"] if s["capacity_vs_market"] is not None else 0))

    return {
        "week_ending": week,
        "weeks": {"from": recent[0], "to": recent[-1]},
        "year_earlier_weeks": {"from": year_ago[0], "to": year_ago[-1]},
        "market": {**mkt_now, "year_on_year_pct_change": market_yoy, "same_store_shows": n_market},
        "benchmark": bench,
        "shows": shows,
        "notes": [
            "Signals are warning signs, not predictions: a limited engagement closes on schedule whatever it grosses, "
            "and a strong show can still close when its theater is booked.",
            "Shows with under a year of grosses have no year-on-year trend; a show in previews sells below a running one.",
            "level_signals: selling below the market now, by more than closing shows typically did. "
            "trend_signals: falling faster than the market on a year earlier. A trend signal on a show still "
            "selling above the market's capacity usually means prices coming down from a peak, not trouble.",
            "Sorted by level signals, then trend signals, then capacity against the market (worst first).",
            f"Grosses data ends {_source('grosses').get('covers_to')}: 'right now' questions can only be answered "
            "for the last week of data.",
        ],
        "source_url": _rows("SELECT source_url FROM grosses WHERE week_ending = ? LIMIT 1", (week,))[0]["source_url"],
    }


@_json_tool
def tony_effect(category: str, from_year: int | None = None, to_year: int | None = None, weeks: int = 8) -> dict:
    """How box office changed after the Tonys for productions that won a category vs those only nominated.

    For every nominee in the matching categories that is linked to a production, runs the same
    before-and-after-the-ceremony comparison as grosses_around_date, and summarizes the change against
    the market (percentage points) for winners and for nominees that lost. A production nominated in a
    category several times (performers) counts once per ceremony, as a winner if any of its nominees won.

    Left out, with the reason: ceremonies whose after-window runs past the end of the grosses data, and
    productions with under half the weeks in either window (closed soon after, or opened just before), or
    whose before-window is mostly previews.
    """
    where, params = ["a.category LIKE ?", "a.production_id IS NOT NULL"], [f"%{category}%"]
    if from_year is not None:
        where.append("a.year >= ?"); params.append(from_year)
    if to_year is not None:
        where.append("a.year <= ?"); params.append(to_year)
    noms = _rows(
        f"""SELECT a.year, c.ceremony_date, a.production_id, p.title, MAX(a.won) AS won,
                   GROUP_CONCAT(DISTINCT a.category) AS categories, MIN(a.source_url) AS source_url
            FROM awards a JOIN tony_ceremonies c USING (year) JOIN productions p USING (production_id)
            WHERE {' AND '.join(where)}
            GROUP BY a.year, a.production_id ORDER BY a.year, won DESC""",
        tuple(params),
    )
    if not noms:
        raise ValueError(
            f"no nominees for category '{category}' in those years are linked to a production with grosses. "
            "Nominees after 2021 aren't linked (grosses end in March 2020). Check the category name with get_awards."
        )

    data_start = date.fromisoformat(_source("grosses").get("covers_from"))
    data_end = date.fromisoformat(_source("grosses").get("covers_to"))
    included, excluded = [], []
    for n in noms:
        ceremony = date.fromisoformat(n["ceremony_date"])
        base = {"year": n["year"], "production_id": n["production_id"], "title": n["title"], "won": bool(n["won"])}
        if ceremony + timedelta(weeks=weeks) > data_end:
            excluded.append({**base, "reason": f"the weeks after the ceremony are past the end of the data ({data_end})"})
            continue
        if ceremony < data_start:
            excluded.append({**base, "reason": f"the grosses data starts after this ceremony ({data_start})"})
            continue
        r = _around(n["production_id"], n["ceremony_date"], weeks, weeks)
        enough = r["before"]["weeks"] >= weeks / 2 and r["after"]["weeks"] >= weeks / 2
        mostly_previews = r["before"]["preview_weeks"] > r["before"]["weeks"] / 2
        vs = r["pct_change_vs_market"]["avg_weekly_gross"]
        if not enough or vs is None:
            excluded.append({**base, "reason": f"only {r['before']['weeks']} weeks before and {r['after']['weeks']} after"})
        elif mostly_previews:
            excluded.append({**base, "reason": "most of the weeks before the ceremony were previews"})
        else:
            included.append({**base, "gross_pct_change": r["pct_change"]["avg_weekly_gross"],
                             "gross_pct_change_vs_market": vs,
                             "capacity_pct_change_vs_market": r["pct_change_vs_market"]["capacity_pct"],
                             "source_url": n["source_url"]})

    def summary(group: list[dict]) -> dict:
        vs = [g["gross_pct_change_vs_market"] for g in group]
        return {
            "productions": len(group),
            "median_gross_pct_change_vs_market": _median(vs),
            "mean_gross_pct_change_vs_market": _mean(vs),
            "share_that_beat_the_market": _r(sum(v > 0 for v in vs) / len(vs), 2) if vs else None,
            "median_capacity_pct_change_vs_market": _median([g["capacity_pct_change_vs_market"] for g in group]),
        }

    winners = summary([g for g in included if g["won"]])
    losers = summary([g for g in included if not g["won"]])
    gap = (None if winners["median_gross_pct_change_vs_market"] is None or losers["median_gross_pct_change_vs_market"]
           is None else round(winners["median_gross_pct_change_vs_market"] - losers["median_gross_pct_change_vs_market"], 1))
    warnings = []
    if min(winners["productions"], losers["productions"]) < 10:
        warnings.append(f"small groups ({winners['productions']} winners, {losers['productions']} nominees that lost): "
                        "one show can move a median a lot")

    return {
        "category": category,
        "categories_matched": sorted({c for n in noms for c in n["categories"].split(",")}),
        "years": {"from": noms[0]["year"], "to": noms[-1]["year"]},
        "weeks_each_side": weeks,
        "winners": winners,
        "nominated_only": losers,
        "median_gap_winners_minus_nominated": gap,
        "productions": included,
        "excluded": excluded,
        "warnings": warnings,
        "notes": [
            "Changes are against the market (same-store, as in grosses_around_date), in percentage points.",
            "This is an association, not proof the award caused it: winners were often already the stronger "
            "sellers, and nominations alone can lift a show before the ceremony.",
            f"Grosses end {data_end}, so ceremonies after 2019 can't be measured.",
        ],
        "source_url": _rows("SELECT source_url FROM grosses LIMIT 1")[0]["source_url"],
    }


# ---------------------------------------------------------------------------
# Music: Broadway cast recordings (MusicBrainz)
# ---------------------------------------------------------------------------


def _song_key(title: str) -> str:
    """'Defying Gravity (Reprise)' and 'Defying Gravity' -> 'defying gravity': how compare_recordings matches songs."""
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", title)
    t = re.sub(r"\breprise\b", " ", fold(t))
    return re.sub(r"\s+", " ", t).strip()


def _recordings(production_id: str) -> list[dict]:
    recs = _rows(
        "SELECT recording_id, title, year, source_url FROM recordings WHERE production_id = ? ORDER BY year, title",
        (production_id,),
    )
    for r in recs:
        r["tracks"] = _rows(
            "SELECT disc, position, title FROM tracks WHERE recording_id = ? ORDER BY disc, position",
            (r["recording_id"],),
        )
    return recs


@_json_tool
def get_recordings(production_id: str) -> dict:
    """A production's Broadway cast recordings, each with its track list."""
    p = _production(production_id)
    recs = _recordings(production_id)
    return {
        "production": {k: p[k] for k in ("production_id", "title", "first_week", "last_week")},
        "recordings": recs,
        "notes": [
            "Only recordings MusicBrainz labels as a Broadway cast are here: no London, tour, film or "
            "off-Broadway albums.",
            "A cast album is not a complete song list: shows sometimes leave songs off the album.",
        ] + ([] if recs else ["This production has no linked Broadway cast recording."]),
    }


@_json_tool
def compare_recordings(production_a: str, production_b: str) -> dict:
    """Which songs are on both productions' cast recordings, and which only on one: e.g. an original vs
    its revival. Matches titles ignoring case, punctuation, accents, '(Reprise)' and other parentheticals.
    When a production has more than one recording, the earliest is used and the others are listed.
    """
    sides = {}
    for label, pid in (("a", production_a), ("b", production_b)):
        p = _production(pid)
        recs = _recordings(pid)
        if not recs:
            raise LookupError(f"{pid} has no linked Broadway cast recording, so there is nothing to compare. "
                              "Call get_recordings to check.")
        sides[label] = {"production": {k: p[k] for k in ("production_id", "title", "first_week", "last_week")},
                        "recording": {k: recs[0][k] for k in ("recording_id", "title", "year", "source_url")},
                        "other_recordings": [r["title"] for r in recs[1:]], "tracks": recs[0]["tracks"]}

    def by_key(tracks: list[dict]) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in tracks:
            # A medley track ('A Very Nice Prince / First Midnight') holds several songs
            for part in t["title"].split(" / "):
                out.setdefault(_song_key(part), part.strip())  # first spelling, in album order
        out.pop("", None)
        return out

    a, b = by_key(sides["a"]["tracks"]), by_key(sides["b"]["tracks"])
    warnings = []
    if fold(sides["a"]["production"]["title"]) != fold(sides["b"]["production"]["title"]):
        warnings.append("these are productions of two different shows, not an original and a revival")
    return {
        "warnings": warnings,
        "a": {k: v for k, v in sides["a"].items() if k != "tracks"} | {"songs": len(a)},
        "b": {k: v for k, v in sides["b"].items() if k != "tracks"} | {"songs": len(b)},
        "in_both": [a[k] for k in a if k in b],
        "only_in_a": [a[k] for k in a if k not in b],
        "only_in_b": [b[k] for k in b if k not in a],
        "notes": [
            "Compares the cast albums, not the stage shows: a song missing from an album may still have been performed.",
            "A renamed song shows up as one cut plus one added; check only_in_a against only_in_b for near matches.",
            "Reprises count as the same song, and a medley track counts as each song in it.",
        ],
    }


# ---------------------------------------------------------------------------
# Cast changes (hand-curated: no full source allows automated access)
# ---------------------------------------------------------------------------


@_json_tool
def get_cast_changes(production_id: str | None = None, person_contains: str | None = None) -> dict:
    """Dated cast changes (a performer joining or leaving), from a small hand-curated list. Give at least one filter."""
    if production_id is None and person_contains is None:
        raise ValueError("give production_id or person_contains")
    where, params = ["1=1"], []
    if production_id is not None:
        where.append("c.production_id = ?"); params.append(production_id)
    if person_contains is not None:
        where.append("c.person LIKE ?"); params.append(f"%{person_contains}%")
    rows = _rows(
        f"""SELECT c.production_id, p.title, c.person, c.role, c.start_date, c.end_date, c.is_star, c.source_url
            FROM cast_changes c JOIN productions p USING (production_id)
            WHERE {' AND '.join(where)} ORDER BY c.start_date""",
        tuple(params),
    )
    total = _rows("SELECT COUNT(*) AS n FROM cast_changes")[0]["n"]
    return {
        "cast_changes": rows,
        "notes": [
            f"Cast changes are hand-curated: {total} in the whole database, for a small set of shows. "
            "No match means it isn't recorded here, not that it didn't happen.",
        ],
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@_json_tool
def data_coverage() -> dict:
    """What each source covers, so an answer can say where the data stops."""
    counts = _rows(
        """SELECT (SELECT COUNT(*) FROM productions) AS productions, (SELECT COUNT(*) FROM grosses) AS gross_weeks,
                  (SELECT COUNT(*) FROM awards) AS award_rows, (SELECT COUNT(*) FROM recordings) AS recordings,
                  (SELECT COUNT(*) FROM cast_changes) AS cast_changes"""
    )[0]
    not_available = [
        "profitability and recoupment", "audience demographics", "per-performance data (grosses are weekly)",
        "touring and off-Broadway productions, and their recordings",
        "song lists beyond Broadway cast albums",
    ]
    if not counts["cast_changes"]:
        not_available.append("cast changes and replacement dates")
    return {
        "sources": _rows("SELECT source, description, covers_from, covers_to, fetched_at, notes FROM data_sources"),
        "counts": counts,
        "not_available": not_available,
    }


if __name__ == "__main__":
    # Smoke test against the real database. Each call prints a short view of its JSON.
    def show(label: str, out: str, keys=None) -> dict:
        d = json.loads(out)
        view = {k: d[k] for k in keys} if keys and "error" not in d else d
        print(f"\n== {label}\n{json.dumps(view, indent=1, default=str)[:1500]}")
        return d

    d = show("find_production('chicago')", find_production("chicago"), ["candidates"])
    assert any(c["production_id"] == "chicago-1996" for c in d["candidates"])
    d = show("find_production('les miserables') (accent-free query)", find_production("les miserables"), ["candidates"])
    assert len(d["candidates"]) >= 2, "original and revival"
    d = show("find_production('Hamiltn') (typo)", find_production("Hamiltn"), ["candidates"])
    assert d["candidates"] and d["candidates"][0]["production_id"] == "hamilton-2015"

    d = show("get_grosses hamilton, 6 weeks", get_grosses("hamilton-2015", "2016-05-01", "2016-06-12"),
             ["granularity", "summary", "rows"])
    assert d["granularity"] == "week" and len(d["rows"]) == 7
    d = show("get_grosses phantom, whole run", get_grosses("the-phantom-of-the-opera-1988"), ["granularity", "summary"])
    assert d["granularity"] == "month"
    assert all(r["performances"] + r["previews"] > 0 for r in grosses_rows("the-phantom-of-the-opera-1988"))

    d = show("grosses_around_date: Hamilton's 2016 Tony", grosses_around_date("hamilton-2015", "2016-06-12"),
             ["before", "after", "event_week", "pct_change", "market_pct_change", "pct_change_vs_market", "warnings"])
    assert d["before"]["weeks"] == 8 and d["after"]["weeks"] == 8 and d["event_week"] == "2016-06-12"

    d = show("get_awards hamilton, wins", get_awards("hamilton-2015", won_only=True), ["awards"])
    assert sum(a["won"] for a in d["awards"]) == 11, "Hamilton won 11 Tonys in 2016"
    d = show("get_awards Best Musical 2026", get_awards(year=2026, category="Best Musical"), ["awards"])
    assert d["awards"]

    show("data_coverage", data_coverage())
    d = show("bad id -> error, not an exception", get_grosses("no-such-show"))
    assert "error" in d
    d = show("no filter -> error", get_awards())
    assert "error" in d
    # Market-wide
    d = show("trouble_signals, week of 2019-08-04", trouble_signals("2019-08-04"), ["week_ending", "market", "benchmark"])
    flagged = [s["production_id"] for s in d["shows"] if s["level_signals"]]
    print("level signals:", flagged)
    # All three closed in August 2019, weeks after this date
    assert {"pretty-woman-the-musical-2018", "the-cher-show-2018", "king-kong-2018"} <= set(flagged), flagged
    for hit in ("wicked-2003", "the-lion-king-1997", "hamilton-2015", "dear-evan-hansen-2016"):
        assert hit not in flagged, f"{hit} sells above the market and shouldn't have a level signal"
    d = show("trouble_signals, too early -> error", trouble_signals("1985-07-01"))
    assert "error" in d

    d = show("tony_effect Best Musical", tony_effect("Best Musical"),
             ["winners", "nominated_only", "median_gap_winners_minus_nominated", "warnings"])
    assert d["winners"]["productions"] > 20 and d["nominated_only"]["productions"] > 40
    hamilton = [p for p in d["productions"] if p["production_id"] == "hamilton-2015"][0]
    same = json.loads(grosses_around_date("hamilton-2015", "2016-06-12"))["pct_change_vs_market"]["avg_weekly_gross"]
    assert hamilton["gross_pct_change_vs_market"] == same, "tony_effect must agree with grosses_around_date"
    d = show("tony_effect 2022+ -> error (not linked)", tony_effect("Best Musical", 2022, 2026))
    assert "error" in d

    # Music
    d = show("compare_recordings Into the Woods 1987 vs 2002", compare_recordings("into-the-woods-1987", "into-the-woods-2002"),
             ["only_in_a", "only_in_b", "warnings"])
    assert "Last Midnight" in d["only_in_b"], "written for the 1990s London production, so on the 2002 album only"
    assert not d["warnings"]
    d = show("compare_recordings two different shows -> warning", compare_recordings("hamilton-2015", "wicked-2003"), ["warnings"])
    assert d["warnings"]
    d = show("get_recordings with two albums", get_recordings("hello-dolly-1995"), ["recordings"])
    assert len(d["recordings"]) == 2

    # Cast (hand-curated, may be empty)
    d = show("get_cast_changes hamilton", get_cast_changes("hamilton-2015"))
    assert "cast_changes" in d
    d = show("get_cast_changes no filter -> error", get_cast_changes())
    assert "error" in d

    print("\nall checks passed")