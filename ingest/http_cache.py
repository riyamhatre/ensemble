"""Polite, cached HTTP for every scraper and API client in ingest/.

    from http_cache import fetch, fetch_json

    html = fetch("https://www.ibdb.com/broadway-production/wicked-12345")
    data = fetch_json("https://musicbrainz.org/ws/2/release-group", params={"query": "wicked", "fmt": "json"})

What it does:
  - Caches every successful response on disk under data/raw/<host>/, so re-parsing is free
    and no URL is fetched twice. Pass force=True to refresh one.
  - Rate-limits per host (HOST_INTERVALS), not globally.
  - Retries 429 / 5xx / network errors with backoff, and honors Retry-After.
  - Sends a descriptive User-Agent with a real contact. Set BROADWAY_CONTACT in .env
    (an email or a project URL). It refuses to fetch without one.
  - secret_params (API keys) are sent but never go into the cache key, the manifest,
    or error messages.

The cache is anchored to the project folder, so scripts behave the same from any directory.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent  # ingest/ sits directly under the project root
RAW = ROOT / "data" / "raw"

# Minimum seconds between requests to the same host. Check each site's current terms and
# API docs before a big run; these are conservative guesses, not official numbers.
DEFAULT_INTERVAL = 1.5
HOST_INTERVALS = {
    "musicbrainz.org": 1.1,  # MusicBrainz asks for at most ~1 request/second
    "www.ibdb.com": 2.0,
    "en.wikipedia.org": 1.0,
    # NYT is not here on purpose: its terms forbid keeping content over 24 hours, so nyt.py never uses this cache
}

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
MAX_WAIT = 120.0  # never sleep longer than this on one retry

_last_request: dict[str, float] = {}  # host -> time.monotonic() of the last request
_client: httpx.Client | None = None


def _user_agent() -> str:
    contact = os.environ.get("BROADWAY_CONTACT", "").strip()
    if not contact:
        raise RuntimeError(
            "Set BROADWAY_CONTACT in .env to an email or project URL. Sites like MusicBrainz "
            "require a User-Agent that says who to contact."
        )
    return f"broadway-research-project/0.1 ({contact})"


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=30, follow_redirects=True)
    return _client


def _wait_turn(host: str) -> None:
    interval = HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
    wait = interval - (time.monotonic() - _last_request.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.monotonic()


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Retry-After if the server sent seconds, else exponential backoff: 2s, 4s, 8s."""
    if response is not None:
        try:
            return min(float(response.headers["Retry-After"]), MAX_WAIT)
        except (KeyError, ValueError):
            pass
    return min(2.0**attempt, MAX_WAIT)


def _cache_path(url: httpx.URL, kind: str) -> Path:
    digest = hashlib.sha256(str(url).encode()).hexdigest()[:32]
    return RAW / url.host / f"{digest}.{'json' if kind == 'json' else 'html'}"


def fetch(
    url: str,
    *,
    params: dict | None = None,
    secret_params: dict | None = None,
    headers: dict | None = None,
    force: bool = False,
    kind: str = "html",
) -> str:
    """GET url and return the body as text, from the disk cache if we have it.

    params         part of the cache key (sorted, so their order doesn't matter)
    secret_params  sent with the request but kept out of the key, manifest and errors
    force          skip the cache read and overwrite the cached copy
    kind           "html" or "json": only sets the cache file's extension
    Raises httpx.HTTPStatusError on a final 4xx/5xx (a 404 is not cached) and
    httpx.TransportError if the network keeps failing.
    """
    key_url = httpx.URL(url)
    if params:
        key_url = key_url.copy_merge_params(dict(sorted(params.items())))
    path = _cache_path(key_url, kind)

    if path.exists() and not force:
        return path.read_text(encoding="utf-8")

    send_url = key_url.copy_merge_params(secret_params) if secret_params else key_url
    request_headers = {"User-Agent": _user_agent(), **(headers or {})}
    if kind == "json":
        request_headers.setdefault("Accept", "application/json")

    host = key_url.host
    response = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        _wait_turn(host)
        try:
            response = _get_client().get(send_url, headers=request_headers)
        except httpx.TransportError:
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(_retry_delay(None, attempt))
            continue
        if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            time.sleep(_retry_delay(response, attempt))
            continue
        break

    if response.is_error:
        # Build the message from key_url: raise_for_status() would print the secret params
        raise httpx.HTTPStatusError(
            f"{response.status_code} for {key_url}", request=response.request, response=response
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # write-then-rename: a crash never leaves a half file
    tmp.write_text(response.text, encoding="utf-8")
    tmp.replace(path)
    _log(key_url, path, response.status_code)
    return response.text


def fetch_json(url: str, **kwargs) -> dict | list:
    """fetch(), parsed as JSON."""
    return json.loads(fetch(url, kind="json", **kwargs))


def _log(url: httpx.URL, path: Path, status: int) -> None:
    """One line per fetch in data/raw/_manifest.jsonl: which file holds which URL."""
    entry = {
        "url": str(url),
        "file": str(path.relative_to(RAW)),
        "status": status,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(RAW / "_manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    # Smoke test: uv run ingest/http_cache.py [url]
    # The first fetch hits the network; the second must come from disk and take ~0 s.
    import sys

    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Hamilton_(musical)"
    for label in ("first", "second"):
        t = time.monotonic()
        body = fetch(test_url)
        print(f"{label}: {len(body):,} chars in {time.monotonic() - t:.2f}s")
    print("cache dir:", RAW)