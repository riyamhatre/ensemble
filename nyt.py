"""New York Times press coverage of a show, looked up live through the Article Search API.

    from nyt import get_press, search_articles

    get_press("hamilton-2015")                      # articles from the run's dates, reviews flagged
    search_articles("Hamilton", start="2015-07-01", end="2015-09-01", reviews_only=True)

Unlike every other source, NYT content is never stored: not in data/csv/, not in broadway.db, not in
data/raw/ (so this does not use http_cache.fetch, whose disk cache keeps everything). The NYT API terms
(as read by the user, 2026-11-03) say:
  - API content may not be cached for user access for more than 24 hours after it is pulled,
  - data sets may not be stored to provide NYT content to anyone else,
  - cached content must be deleted at once if API access ends,
  - API data may not be used to train ML/AI systems or develop software products.
So responses live in a process-memory cache for at most CACHE_TTL, and vanish when the server stops.
The last point is a judgment call for this project, since the agents read these results: the user
chose this live design knowing that.

Set NYT_API_KEY in .env. Without it every call returns {"error": ...} explaining that.
Smoke test: uv run nyt.py [title]   (offline checks always; a live call too if the key is set)
"""

import json
import os
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Under the 24-hour limit, with a margin so a result fetched at 9:00 is gone well before 9:00 tomorrow
CACHE_TTL = 23 * 3600

# NOT VERIFIED against the current docs (developer.nytimes.com refuses automated fetches): the commonly
# documented limits are 5 requests per minute and 500 per day. Check your key's dashboard.
MIN_INTERVAL = 12.5  # seconds between requests
DAILY_LIMIT = 500
MAX_ARTICLES = 10  # one page of results; each further page is another 12.5 s request

ATTRIBUTION = "Data provided by The New York Times"

_cache: dict[tuple, tuple[float, list[dict]]] = {}  # query -> (fetched at, articles). Memory only.
_lock = threading.Lock()  # the agents call tools from several threads; one request at a time
_last_request = 0.0
_day = {"date": "", "count": 0}


def _api_key() -> str:
    key = os.environ.get("NYT_API_KEY", "").strip()
    if not key:
        raise RuntimeError("NYT_API_KEY is not set in .env. Get a key at developer.nytimes.com (Article Search API).")
    return key


def _yyyymmdd(iso: str | None) -> str | None:
    return iso.replace("-", "")[:8] if iso else None


# ---------------------------------------------------------------------------
# RESPONSE SHAPE. NOT VERIFIED against current docs: the fields Article Search v2 has long returned.
# Every field is read with a fallback, so a renamed field gives None instead of a crash, and
# parse_docs is the one place to fix if a live call shows different names.
# ---------------------------------------------------------------------------
def parse_docs(body: dict) -> list[dict]:
    docs = (body.get("response") or {}).get("docs") or []
    out = []
    for d in docs:
        headline = d.get("headline")
        headline = headline.get("main") if isinstance(headline, dict) else headline
        material = d.get("type_of_material") or ""
        section = d.get("section_name") or ""
        out.append(
            {
                "headline": headline,
                "abstract": d.get("abstract") or d.get("snippet"),
                "pub_date": (d.get("pub_date") or "")[:10] or None,
                "web_url": d.get("web_url"),
                "section": section or None,
                # Theater reviews have type_of_material 'Review'; their headlines also start 'Review:'
                "is_review": int(material.lower() == "review" or (headline or "").startswith("Review:")),
            }
        )
    return out


def _request(params: dict) -> dict:
    """One API call, at most one per MIN_INTERVAL and DAILY_LIMIT per day. The key never appears in an error."""
    global _last_request
    today = time.strftime("%Y-%m-%d")
    if _day["date"] != today:
        _day.update(date=today, count=0)
    if _day["count"] >= DAILY_LIMIT:
        raise RuntimeError(f"NYT daily limit ({DAILY_LIMIT}) reached for {today}; try again tomorrow")

    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()
    _day["count"] += 1
    try:
        r = httpx.get(URL, params={**params, "api-key": _api_key()}, timeout=30,
                      headers={"User-Agent": "broadway-research-project/0.1"})
    except httpx.TransportError as e:
        raise RuntimeError(f"NYT request failed: {type(e).__name__}") from None  # httpx's text has the URL
    if r.status_code == 429:
        raise RuntimeError("NYT rate limit hit (429): wait a minute, or the daily cap is used up")
    if r.status_code in (401, 403):
        raise RuntimeError(f"NYT refused the key ({r.status_code}): check NYT_API_KEY and that Article Search is enabled")
    if r.is_error:
        raise RuntimeError(f"NYT returned {r.status_code}")
    return r.json()


def _search(title: str, start: str | None, end: str | None) -> list[dict]:
    key = (title.lower(), start, end)
    now = time.time()
    with _lock:
        # Drop everything past its TTL, not just this key: nothing may outlive 24 hours
        for k in [k for k, (t, _) in _cache.items() if now - t > CACHE_TTL]:
            del _cache[k]
        if key in _cache:
            return _cache[key][1]
        params = {"q": f'"{title}" Broadway', "sort": "relevance"}
        if start:
            params["begin_date"] = _yyyymmdd(start)
        if end:
            params["end_date"] = _yyyymmdd(end)
        articles = parse_docs(_request(params))
        _cache[key] = (now, articles)
        return articles


def search_articles(title: str, start: str | None = None, end: str | None = None, reviews_only: bool = False) -> str:
    """NYT articles mentioning the title and Broadway between start and end (ISO dates), as JSON."""
    try:
        articles = _search(title, start, end)
        if reviews_only:
            articles = [a for a in articles if a["is_review"]]
        return json.dumps(
            {
                "query": {"title": title, "start": start, "end": end, "reviews_only": reviews_only},
                "articles": articles[:MAX_ARTICLES],
                "notes": [
                    ATTRIBUTION + ". Looked up live; not stored. Cite each article by its web_url.",
                    "Only the top results by relevance are returned, so this is a sample of the coverage, "
                    "not all of it. Headlines and abstracts only, never the article text.",
                ],
            }
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def get_press(production_id: str, reviews_only: bool = False) -> str:
    """NYT coverage of one production: from 4 months before its first week of grosses (previews and
    advance press) to a month after its last week."""
    try:
        import broadway_data as db  # here, not at the top: nyt.py works on its own without the database

        p = db._production(production_id)
        start = (date.fromisoformat(p["first_week"]) - timedelta(days=120)).isoformat()
        end = (date.fromisoformat(p["last_week"]) + timedelta(days=30)).isoformat()
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    out = json.loads(search_articles(p["title"], start, end, reviews_only))
    if "error" not in out:
        out["production_id"] = production_id
    return json.dumps(out)


def clear_cache() -> None:
    """Delete every cached NYT result. Call this if API access ends: the terms require it at once."""
    with _lock:
        _cache.clear()


if __name__ == "__main__":
    import sys

    # Offline checks: no key or network needed.
    sample = {"response": {"docs": [
        {"headline": {"main": "Review: ‘Hamilton,’ Young Rebels Changing History"}, "abstract": "a", "type_of_material": "Review",
         "section_name": "Theater", "pub_date": "2015-08-06T00:00:00+0000", "web_url": "https://www.nytimes.com/x"},
        {"headline": {"main": "Grosses Rise"}, "snippet": "s", "type_of_material": "News", "pub_date": "2016-06-20T00:00:00Z",
         "web_url": "https://www.nytimes.com/y"},
        {"headline": "a renamed shape"},
    ]}}
    docs = parse_docs(sample)
    assert [d["is_review"] for d in docs] == [1, 0, 0]
    assert docs[0]["pub_date"] == "2015-08-06" and docs[1]["abstract"] == "s" and docs[2]["web_url"] is None
    assert _yyyymmdd("2015-07-19") == "20150719"
    _cache[("x", None, None)] = (time.time() - CACHE_TTL - 1, [])  # an expired entry...
    saved = os.environ.pop("NYT_API_KEY", None)
    out = json.loads(search_articles("x"))  # ...must not be served (and with no key, this is an error)
    assert "error" in out and "NYT_API_KEY" in out["error"] and ("x", None, None) not in _cache
    if saved is not None:
        os.environ["NYT_API_KEY"] = saved
    assert "error" in json.loads(get_press("no-such-show"))
    print("offline checks passed")

    if not os.environ.get("NYT_API_KEY", "").strip():
        print("NYT_API_KEY not set: skipping the live call. Add it to .env and rerun.")
        sys.exit(0)
    title = sys.argv[1] if len(sys.argv) > 1 else "Hamilton"
    t = time.monotonic()
    first = json.loads(get_press("hamilton-2015") if title == "Hamilton" else search_articles(title))
    print(json.dumps(first, indent=1)[:2500])
    t2 = time.monotonic()
    again = get_press("hamilton-2015") if title == "Hamilton" else search_articles(title)
    print(f"first call {t2 - t:.1f}s, repeat {time.monotonic() - t2:.2f}s (memory cache)")
