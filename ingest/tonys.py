"""Tony Awards winners and nominees from Wikipedia's per-ceremony pages -> two staging CSVs.

    uv run ingest/tonys.py

One page per ceremony ("76th Tony Awards"), fetched as rendered HTML through the MediaWiki parse API
(public, no key; Wikipedia asks for a descriptive User-Agent and serial requests, which fetch() does).

  data/csv/tony_ceremonies.csv   year, ceremony_number, ceremony_date, source_url
  data/csv/tony_awards.csv       one row per nominee per competitive category, with won 0/1

Every page from 1985 to 2026 lays the competitive awards out the same way: a grid where each header
cell is a category and the cell under it is a list. The top-level items are the winners (more than one
on a tie) and the nested list under them holds the other nominees.

The awards file has show_title (the italic title in the nominee line) instead of production_id.
link_shows.py resolves it, using the ceremony date to choose between productions with the same title.
Non-competitive awards (special Tonys, regional theatre award) are not collected.
"""

import copy
import re

import pandas as pd
from bs4 import BeautifulSoup

from http_cache import ROOT, fetch_json

API = "https://en.wikipedia.org/w/api.php"
OUT = ROOT / "data" / "csv"

# The 39th ceremony (June 1985) is the first inside the grosses data, which starts 1985-06-09.
# Bump LAST_CEREMONY after each June ceremony once its page lists the winners.
FIRST_CEREMONY = 39
LAST_CEREMONY = 79  # June 7, 2026

# Categories where the nominee is the show itself
SHOW_CATEGORIES = re.compile(r"^Best (Play|Musical|Revival|Special Theatrical Event)", re.I)

# Pages name the same category differently, so `category` holds one name per award and
# `category_printed` keeps the page's wording. Found by listing every name and the years it appears:
# 2012, 2016, 2017, 2022 and 2024 use "Best Actor in a Play"; 2021 uses "Scenic Design in a Play".
# "Best Revival" (to 1993) is left alone: it covered plays and musicals together.
CATEGORY_FIXES = [
    (re.compile(r"^Best (Actor|Actress) in a (Play|Musical)$"), r"Best Performance by a Leading \1 in a \2"),
    (re.compile(r"^Best Featured (Actor|Actress) in a (Play|Musical)$"), r"Best Performance by a Featured \1 in a \2"),
    (re.compile(r"^Best (Scenic|Costume|Lighting|Sound) Design in a (Play|Musical)$"), r"Best \1 Design of a \2"),
    (re.compile(r"^Best Original Score.*"), "Best Original Score (Music and/or Lyrics) Written for the Theatre"),
]
PLACEHOLDER = re.compile(r"^no (other|additional) nominees?$", re.I)
DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}")


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def fetch_page(n: int) -> tuple[str, int, BeautifulSoup]:
    data = fetch_json(
        API,
        params={
            "action": "parse",
            "page": f"{ordinal(n)}_Tony_Awards",
            "prop": "text|revid",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        },
    )
    if "error" in data:  # a 200 with an error body, e.g. the page doesn't exist yet
        raise ValueError(f"{ordinal(n)} Tony Awards: {data['error'].get('info')}")
    p = data["parse"]
    return p["title"], p["revid"], BeautifulSoup(p["text"], "lxml")


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("‡", "")
    text = re.sub(r"\[\w+\]", "", text)  # footnote markers like [a] or [12]
    return re.sub(r"\s+", " ", text).strip(" –—-")


def normalize_category(name: str) -> str:
    for pattern, repl in CATEGORY_FIXES:
        if pattern.match(name):
            return pattern.sub(repl, name)
    return name


def ceremony_date(soup: BeautifulSoup) -> str | None:
    for tr in soup.select("table.infobox tr"):
        th = tr.find("th")
        if th and th.get_text(strip=True) == "Date":
            m = DATE.search(clean(tr.get_text(" ")))  # the 74th writes the date with non-breaking spaces
            return pd.to_datetime(m.group(0)).date().isoformat() if m else None
    return None


def parse_awards(soup: BeautifulSoup) -> list[dict]:
    """The competitive-awards grid is the first wikitable on every page checked (39th, 50th, 76th, 79th)."""
    grid = soup.select_one("table.wikitable")
    rows = grid.select(":scope > tbody > tr")
    out = []
    # Rows alternate: a row of category headers, then a row of nominee cells under them
    for head, body in zip(rows[0::2], rows[1::2]):
        for th, td in zip(head.find_all("th"), body.find_all("td")):
            printed = clean(th.get_text(" "))
            category = normalize_category(printed)
            show_award = bool(SHOW_CATEGORIES.match(category))
            for top in td.find_all("li", recursive=False) or td.select(":scope > ul > li"):
                out.append(_nominee(top, category, printed, show_award, won=1))
                for li in top.select(":scope > ul > li"):
                    out.append(_nominee(li, category, printed, show_award, won=0))
    # 1995's Book and Score had a single nominee; the page lists "No additional nominees" in italics
    return [r for r in out if not PLACEHOLDER.match(r["nominee"])]


def _nominee(li, category: str, printed: str, show_award: bool, won: int) -> dict:
    own = copy.copy(li)
    for ul in own.find_all("ul"):  # drop the nested nominee list, keep only this line
        ul.decompose()
    titles = [clean(i.get_text()) for i in own.find_all("i")]
    titles = [t for t in titles if t]
    return {
        "category": category,
        "category_printed": printed,
        "nominee": clean(own.get_text("")),
        # The first italic is the show. A line rarely has two (a performer nominated for two shows);
        # n_titles lets the check below count those instead of losing them silently.
        "show_title": titles[0] if titles else None,
        "n_titles": len(titles),
        "is_show_award": int(show_award),
        "won": won,
        "bold": int(own.find("b") is not None),
    }


def main() -> None:
    ceremonies, awards = [], []
    for n in range(FIRST_CEREMONY, LAST_CEREMONY + 1):
        title, revid, soup = fetch_page(n)
        # A permalink to the exact revision parsed, so a citation still says the same thing next year
        url = f"https://en.wikipedia.org/w/index.php?title={title.replace(' ', '_')}&oldid={revid}"
        date = ceremony_date(soup)
        year = int(date[:4]) if date else None
        ceremonies.append({"year": year, "ceremony_number": n, "ceremony_date": date, "source_url": url})
        for row in parse_awards(soup):
            awards.append({"year": year, **row, "source_url": url})

    cer = pd.DataFrame(ceremonies)
    aw = pd.DataFrame(awards)

    # Checks. Each one is something that went wrong, or could, on a real page.
    assert cer["ceremony_date"].notna().all(), f"no date: {cer[cer['ceremony_date'].isna()]}"
    assert cer["year"].is_unique, "two ceremonies in one year"
    per = aw.groupby(["year", "category"])["won"].sum()
    assert (per >= 1).all(), f"categories with no winner:\n{per[per < 1]}"
    no_title = aw[aw["show_title"].isna()]
    assert len(no_title) <= 0.01 * len(aw), f"{len(no_title)} nominees with no show title:\n{no_title.head(20)}"

    OUT.mkdir(parents=True, exist_ok=True)
    cer.to_csv(OUT / "tony_ceremonies.csv", index=False)
    cols = ["year", "category", "category_printed", "nominee", "show_title", "is_show_award", "won", "source_url"]
    aw[cols].to_csv(OUT / "tony_awards.csv", index=False)

    print(f"{len(cer)} ceremonies ({cer['year'].min()}-{cer['year'].max()}), {len(aw):,} nominee rows, "
          f"{aw['won'].sum():,} wins -> {OUT}")
    print(f"ties (categories with 2+ winners): {(per > 1).sum()}")
    print(f"winners not bold: {((aw['won'] == 1) & (aw['bold'] == 0)).sum()}, "
          f"nominees bold: {((aw['won'] == 0) & (aw['bold'] == 1)).sum()}")
    print(f"lines with no show title: {len(no_title)}, with 2+ titles: {(aw['n_titles'] > 1).sum()}")


if __name__ == "__main__":
    main()
