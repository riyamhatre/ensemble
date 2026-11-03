"""Broadway cast recordings and their track lists from MusicBrainz -> staging CSVs.

    uv run ingest/musicbrainz.py                  every title in the grosses data (~25 min the first time)
    uv run ingest/musicbrainz.py Hamilton Chicago  just these titles, for testing

MusicBrainz web service v2 (JSON). Its rules, checked 2026-11-03: at most one request per second, and a
User-Agent naming the app and a contact (http_cache does both; going faster gets a 503, which fetch()
retries). Core data is CC0.

There is no "cast recording" type: cast albums are release groups with the secondary type Soundtrack,
same as film soundtracks. So for each show title this searches soundtrack release groups and keeps one
only when
  - its title, before any subtitle, is the show's title ('Hamilton: An American Musical' yes,
    'Chicago Hope' no), and
  - its disambiguation or a release title says Broadway cast ('2015 original Broadway cast',
    'Kinky Boots (Original Broadway Cast Recording)'), and nothing names another production
    (London, film, tour, studio, off-Broadway...). Compilations (highlights albums) are dropped.
Then one release from each kept group is looked up for its track list.

  data/csv/recordings.csv           one row per Broadway cast recording, with show_title and cast_year
                                    instead of production_id (build_db.py / link_shows.py resolve it)
  data/csv/tracks.csv               mbid, disc, position, title
  data/csv/recordings_rejected.csv  title matched but rejected, with the reason: check here for misses
"""

import re
import sys
import time

import httpx
import pandas as pd

from http_cache import ROOT, fetch_json
from titles import fold

API = "https://musicbrainz.org/ws/2"
OUT = ROOT / "data" / "csv"

# The signals, in one place. Built from real search results for Hamilton, Chicago, Kinky Boots and
# Les Misérables: the disambiguation is usually '<year> original Broadway cast' or '<year> Broadway
# revival cast'; some groups only say it in a release title.
BROADWAY = re.compile(r"\bbroadway\b", re.I)
OFF_BROADWAY = re.compile(r"off[-‐‑– ]broadway", re.I)  # MusicBrainz often writes the Unicode hyphen U+2010
OTHER_PRODUCTION = re.compile(
    r"london|west end|film|motion picture|movie|tour|studio|concept|concert|national|"
    r"japanese|german|vienna|stockholm|australian|korean|dutch|mexican|spanish|high school|workshop|"
    r"karaoke|instrumental|highlights|sing[- ]along|demo",
    re.I,
)
REVIVAL = re.compile(r"revival|new broadway cast", re.I)
REPLACEMENT = re.compile(r"replacement", re.I)
YEAR = re.compile(r"\b(19[2-9]\d|20[0-4]\d)\b")


def search(title: str) -> list[dict]:
    # Grosses titles carry suffixes MusicBrainz doesn't ('School of Rock - The Musical'): search without them
    query_title = re.sub(r"[\s:–-]*the musical!?$", "", title, flags=re.I) or title
    query_title = query_title.replace("\\", "\\\\").replace('"', '\\"')
    data = fetch_json(
        f"{API}/release-group",
        params={"query": f'releasegroup:"{query_title}" AND secondarytype:soundtrack', "fmt": "json", "limit": "100"},
    )
    return data.get("release-groups", [])


def base_title(t: str) -> str:
    """The title before a subtitle: 'Hamilton: An American Musical' -> 'Hamilton'."""
    return re.split(r" \(|: | - ", t, maxsplit=1)[0]


def classify(show_title: str, rg: dict) -> tuple[dict | None, str]:
    """(recording row, '') if rg is a Broadway cast recording of show_title, else (None, reason)."""
    want = fold(show_title)
    titles = [rg["title"]] + [r["title"] for r in rg.get("releases", [])]
    if not any(fold(base_title(t)) == want or fold(t) == want for t in titles):
        return None, "title"
    # Recordings of one performance ('Wicked (2004-06-04, Original Broadway Cast)') have only Bootleg releases
    if not any(r.get("status") == "Official" for r in rg.get("releases", [])):
        return None, "no official release"
    dis = rg.get("disambiguation", "")
    # The group's own name says which production it is. Its releases' titles can only add the Broadway
    # signal: one release in the Hamilton group is 'The Hamilton Instrumentals', which says nothing
    # about the group.
    own = f"{dis} | {rg['title']}"
    # Highlights albums are Compilations. But so, by mistake, are the full 'Mean Girls (Original Broadway
    # Cast Recording)' and Tootsie's: keep a compilation whose own name says Broadway. Highlights albums
    # that say so ('Highlights from the Original Broadway Cast') still fall to OTHER_PRODUCTION below.
    if "Compilation" in rg.get("secondary-types", []) and not BROADWAY.search(own):
        return None, "compilation"
    text = " | ".join([own] + titles)
    if OFF_BROADWAY.search(text):
        return None, "off-broadway"  # a different production; the schema has nowhere to put it yet
    if not BROADWAY.search(text):
        return None, "no broadway signal"
    if OTHER_PRODUCTION.search(own):
        return None, "names another production: " + OTHER_PRODUCTION.search(own).group(0)

    m = YEAR.search(dis) or YEAR.search(rg["title"])
    first = rg.get("first-release-date", "")
    return {
        "show_title": show_title,
        "mbid": rg["id"],
        "title": rg["title"],
        "disambiguation": dis,
        "first_release_date": first or None,
        # The cast's year in the disambiguation ('1996 Broadway revival cast') picks the production
        # better than the release date, which can be a reissue.
        "cast_year": int(m.group(1)) if m else (int(first[:4]) if first[:4].isdigit() else None),
        "cast_kind": "replacement" if REPLACEMENT.search(text) else "revival" if REVIVAL.search(text) else "original",
        "source_url": f"https://musicbrainz.org/release-group/{rg['id']}",
    }, ""


def pick_release(rg: dict) -> str | None:
    """The release to take tracks from: the earliest official one, which is the album as first issued.
    Later editions add bonus discs (Wicked's anniversary edition adds German-language tracks).
    Search results carry no release dates, so this browses the group's releases: one call per recording."""
    data = fetch_json(f"{API}/release", params={"release-group": rg["id"], "fmt": "json", "limit": "100"})
    official = [r for r in data.get("releases", []) if r.get("status") == "Official"]
    official.sort(key=lambda r: (r.get("date") or "9999", r["id"]))  # undated last; id breaks ties stably
    return official[0]["id"] if official else None


# Digital editions append the album to every track: 'Popular - From "Wicked" Original Broadway Cast Recording/2003'
TRACK_SUFFIX = re.compile(r'\s+-\s+(From|from) ["“].*$')


def clean_track(title: str) -> str:
    return TRACK_SUFFIX.sub("", title).strip()


def tracks(release_id: str, mbid: str) -> list[dict]:
    data = fetch_json(f"{API}/release/{release_id}", params={"inc": "recordings", "fmt": "json"})
    rows = []
    for disc, medium in enumerate(data.get("media", []), start=1):
        for t in medium.get("tracks", []):
            rows.append({"mbid": mbid, "disc": disc, "position": t["position"], "title": clean_track(t["title"])})
    return rows


RETRY_PAUSE = 60  # seconds before retrying the titles that failed. 503 also means "overloaded", and that passes


def search_all(titles: list[str]) -> tuple[dict[str, list[dict]], list[str]]:
    """{title: release groups}, and the titles whose search still failed after fetch()'s own retries."""
    found, failed = {}, []
    for i, title in enumerate(titles, 1):
        try:
            found[title] = search(title)
        except httpx.HTTPError as e:
            failed.append(title)
            print(f"  search failed for {title!r}: {e}", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(titles)} titles searched", flush=True)
    return found, failed


def main(titles: list[str]) -> None:
    results, failed = search_all(titles)
    if failed:
        print(f"retrying {len(failed)} failed titles in {RETRY_PAUSE} s", flush=True)
        time.sleep(RETRY_PAUSE)
        more, failed = search_all(failed)
        results.update(more)

    recordings, track_rows, rejected = [], [], []
    for title, groups in results.items():
        for rg in groups:
            row, reason = classify(title, rg)
            if row is None:
                if reason != "title":  # 'title' is noise like 'Chicago Hope'; the rest is worth a look
                    rejected.append({"show_title": title, "mbid": rg["id"], "title": rg["title"],
                                     "disambiguation": rg.get("disambiguation", ""), "reason": reason})
                continue
            release_id = None
            try:
                release_id = pick_release(rg)
                got = tracks(release_id, rg["id"]) if release_id else []
            except httpx.HTTPError as e:
                print(f"  track lookup failed for {rg['title']!r}: {e}; rerun to fill it in", flush=True)
                got = []
            recordings.append({**row, "release_mbid": release_id, "n_tracks": len(got)})
            track_rows += got

    rec = pd.DataFrame(recordings).drop_duplicates("mbid")  # two grosses titles can fold to one search
    tr = pd.DataFrame(track_rows).drop_duplicates(["mbid", "disc", "position"])
    rej = pd.DataFrame(rejected)

    # Checks
    assert rec["mbid"].is_unique
    assert set(tr["mbid"]) <= set(rec["mbid"]), "tracks for a recording we didn't keep"
    no_tracks = rec[rec["n_tracks"] == 0]

    OUT.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT / "recordings.csv", index=False)
    tr.to_csv(OUT / "tracks.csv", index=False)
    rej.to_csv(OUT / "recordings_rejected.csv", index=False)

    print(f"{len(titles)} titles searched -> {len(rec)} Broadway cast recordings for {rec['show_title'].nunique()} "
          f"titles, {len(tr):,} tracks -> {OUT}")
    print("by kind:", rec["cast_kind"].value_counts().to_dict())
    if failed:
        print(f"STILL FAILED ({len(failed)} titles, not in the CSVs; rerun later, the rest is cached): {failed}")
    print(f"recordings with no tracks: {len(no_tracks)}; rejected (see recordings_rejected.csv): "
          f"{rej['reason'].str.split(':').str[0].value_counts().to_dict() if len(rej) else {}}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        runs = pd.read_csv(OUT / "grosses_runs.csv")
        main(sorted(runs["show"].unique()))
