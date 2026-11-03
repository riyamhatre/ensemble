"""Query functions over data/broadway.db: the only way agents see the data.

    from broadway_data import find_production, get_grosses, grosses_around_date, get_awards, data_coverage

Every function returns a JSON string, and catches its own exceptions into {"error": ...}, because the
model can't see a Python exception (app.py wraps these as tools). Every fact carries a source_url so the
Writer can cite it. All arithmetic (averages, percent changes, ratios) happens here, never in the model.

The database is opened read-only. Rebuild it with `uv run build_db.py`.

Smoke test: uv run broadway_data.py
"""

import difflib
import functools
import json
import sqlite3
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

# Said with every production: the provisional values from build_db.py, until IBDB data replaces them
PROVISIONAL_NOTE = (
    "type (original/revival) is a guess from the grosses data, which starts in 1985: a revival of a show "
    "first staged before 1985 shows as 'original'. opening_date is unknown; first_week may be a preview week."
)


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
# Coverage
# ---------------------------------------------------------------------------


@_json_tool
def data_coverage() -> dict:
    """What each source covers, so an answer can say where the data stops."""
    counts = _rows(
        """SELECT (SELECT COUNT(*) FROM productions) AS productions, (SELECT COUNT(*) FROM grosses) AS gross_weeks,
                  (SELECT COUNT(*) FROM awards) AS award_rows"""
    )[0]
    return {
        "sources": _rows("SELECT source, description, covers_from, covers_to, fetched_at, notes FROM data_sources"),
        "counts": counts,
        "not_available": [
            "profitability and recoupment", "audience demographics", "per-performance data (grosses are weekly)",
            "touring and off-Broadway productions",
        ],
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
    print("\nall checks passed")
