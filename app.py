import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import uvicorn
from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import broadway_data as db
import nyt
import wikipedia
from context_panel import AgentContexts, PanelHooks, add_context_panel

# WIKIMEDIA_USERNAME and WIKIMEDIA_PASSWORD come from the .env file next to this one.
load_dotenv(Path(__file__).parent / ".env")

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# LiteLLM routes the call to Gemini. api_key is unused: Vertex AI uses your gcloud credentials.
MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash", api_key="unused")
# Researchers call tools and copy numbers out of them: the cheaper model is enough.
RESEARCHER_MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash-lite", api_key="unused")
# A stalled model call would hang the page: give up after 90 s, and retry twice.
SETTINGS = ModelSettings(extra_args={"vertex_location": "global", "timeout": 90, "num_retries": 2})

# The Lead gets this many researchers in all. Prompts ask for restraint; code enforces it.
MAX_RESEARCHERS = 6
# Each Researcher's turn cap: enough for its tool calls plus its answer, low enough that one
# Researcher can't eat the 5-minute grading window.
RESEARCHER_MAX_TURNS = 8

# What the Lead or Barfee replies when the question needs the user first
CLARIFY = "CLARIFY:"


def coverage_text() -> str:
    """What the data covers and doesn't, read from the database's data_sources table, so it can't
    drift from the data. Barfee, Glinda and Hamilton all quote it."""
    cov = json.loads(db.data_coverage())
    if "error" in cov:
        return f"(Couldn't read the data coverage: {cov['error']})"
    lines = []
    for s in cov["sources"]:
        dates = f" ({s['covers_from']} to {s['covers_to']})" if s.get("covers_from") else ""
        lines.append(f"- {s['source']}: {s['description']}{dates}. {s.get('notes') or ''}".rstrip())
    # The NYT isn't in the database (its terms forbid storing it), so data_sources doesn't list it
    if os.environ.get("NYT_API_KEY", "").strip():
        lines.append(
            "- press: New York Times articles (headline, date, abstract, link; whether it's a review), looked "
            "up live through the NYT Article Search API and never stored. Only the top results by relevance, "
            "so a sample of the coverage, not all of it. No full article text."
        )
    else:
        lines.append("- press: New York Times articles are not available on this server (no API key set).")
    lines.append("- Not in the data at all: " + ", ".join(cov["not_available"]) + ".")
    return "\n".join(lines)


# The database is a read-only snapshot, so reading its coverage once at startup is enough
COVERAGE = coverage_text()
# The last week of grosses, for prompts that need to name it
GROSSES_END = next(s["covers_to"] for s in json.loads(db.data_coverage())["sources"] if s["source"] == "grosses")


@dataclass
class Research:
    """One run's state, shared by every agent and tool in it. The model never sees it."""

    hooks: PanelHooks
    researchers: int = 0  # how many researchers have started, to number them
    findings: list[str] = field(default_factory=list)  # what each researcher reported, for the Writer


class ToolCallHooks(PanelHooks):
    """PanelHooks that also keep a record of every tool call, by every agent, for /chat's
    tool_calls: {"agent", "name", "args", "result"}, the same shape as Project 1 plus "agent"."""

    def __init__(self, contexts: AgentContexts, emit):
        super().__init__(contexts, emit)
        self.tool_calls: list[dict] = []

    async def on_tool_end(self, context, agent, tool, result):
        # For function tools, context is a ToolContext carrying the call's JSON arguments
        raw = getattr(context, "tool_arguments", None) or "{}"
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            args = {"raw": raw}
        self.tool_calls.append({"agent": agent.name, "name": tool.name, "args": args, "result": str(result)})
        # The page draws a box office chart for each production a Fate pulls grosses for
        if tool.name in CHART_TOOLS and isinstance(args.get("production_id"), str):
            self.emit({"type": "chart", "agent": agent.name, "production_id": args["production_id"],
                       "date": args.get("date")})


# Tools whose production the page charts (see /api/grosses)
CHART_TOOLS = {"get_grosses", "grosses_around_date"}


# --- Tools: thin wrappers over broadway_data, which does the queries and all the arithmetic ---
# Each returns a JSON string with a source_url for citing, and an {"error": ...} the model can read
# instead of raising.


@function_tool
def find_production(name: str, year: int | None = None) -> str:
    """Find Broadway productions by title. Returns candidates, each with its production_id, run dates,
    weeks of grosses, and Tony wins and nominations. Several candidates with one title are different
    runs (an original and its revivals).

    Args:
        name: The show's title, e.g. 'Hamilton'. Spelling and accents don't need to be exact.
        year: Optional. Keep only productions that played that calendar year.
    """
    return db.find_production(name, year)


@function_tool
def get_grosses(production_id: str, start: str | None = None, end: str | None = None) -> str:
    """Weekly box office for one production: gross, potential gross, capacity, average ticket.
    Long ranges come back as monthly totals; narrow start and end for weekly rows.

    Args:
        production_id: From find_production, e.g. 'hamilton-2015'.
        start: Optional first week, ISO date 'YYYY-MM-DD'.
        end: Optional last week, ISO date 'YYYY-MM-DD'.
    """
    return db.get_grosses(production_id, start, end)


@function_tool
def grosses_around_date(production_id: str, date: str, weeks_before: int = 8, weeks_after: int = 8) -> str:
    """Compare a production's box office before and after an event (a Tony win, a star joining or
    leaving), with percent changes already computed, and the same change for the rest of Broadway.

    Args:
        production_id: From find_production.
        date: The event's date, ISO 'YYYY-MM-DD'. For a Tony, the ceremony_date from get_awards.
        weeks_before: Weeks before the event to average. Default 8.
        weeks_after: Weeks after the event to average. Default 8.
    """
    return db.grosses_around_date(production_id, date, weeks_before, weeks_after)


@function_tool
def get_awards(
    production_id: str | None = None,
    year: int | None = None,
    category: str | None = None,
    nominee_contains: str | None = None,
    won_only: bool = False,
) -> str:
    """Tony Award nominations and wins, each with its ceremony_date. Give at least one filter.

    Args:
        production_id: From find_production.
        year: Ceremony year, e.g. 2016. There was no ceremony in 2020.
        category: Part of a category name, e.g. 'Best Musical' or 'Revival'.
        nominee_contains: Text in the nominee line: a show or a person's name.
        won_only: True for winners only.
    """
    return db.get_awards(production_id, year, category, nominee_contains, won_only)


@function_tool
def data_coverage() -> str:
    """What the database covers: each source's date range and known gaps, and what it can't answer."""
    return db.data_coverage()


@function_tool
def trouble_signals(week_ending: str | None = None, weeks: int = 8) -> str:
    """Which shows running in a given week showed box office warning signs, every one measured against
    the rest of Broadway: level_signals (selling below the market now, by more than closing shows
    typically did) and trend_signals (falling faster than the market on a year earlier). Sorted worst first.

    Args:
        week_ending: Optional ISO date 'YYYY-MM-DD'; the nearest week on or before it. Default: the last
            week of data.
        weeks: How many weeks to average. Default 8.
    """
    return db.trouble_signals(week_ending, weeks)


@function_tool
def tony_effect(category: str, from_year: int | None = None, to_year: int | None = None, weeks: int = 8) -> str:
    """How box office changed after the Tony ceremony for productions that won a category vs those only
    nominated, against the market: medians, counts, every production used, and those left out and why.

    Args:
        category: Part of a category name, e.g. 'Best Musical' or 'Leading Actress in a Musical'.
        from_year: Optional first ceremony year.
        to_year: Optional last ceremony year.
        weeks: Weeks before and after the ceremony. Default 8.
    """
    return db.tony_effect(category, from_year, to_year, weeks)


@function_tool
def get_recordings(production_id: str) -> str:
    """A production's Broadway cast recordings (MusicBrainz), each with its track list.

    Args:
        production_id: From find_production.
    """
    return db.get_recordings(production_id)


@function_tool
def compare_recordings(production_a: str, production_b: str) -> str:
    """Which songs are on both productions' cast albums and which only on one, e.g. an original vs its
    revival. Reprises and medleys are already matched: report the lists, don't compare tracks yourself.

    Args:
        production_a: The earlier production_id, from find_production.
        production_b: The later production_id.
    """
    return db.compare_recordings(production_a, production_b)


@function_tool
def get_cast_changes(production_id: str | None = None, person_contains: str | None = None) -> str:
    """Dated cast changes (a performer joining or leaving) from a small hand-curated list. Give at least
    one filter. No match means it isn't recorded here, not that it didn't happen.

    Args:
        production_id: From find_production.
        person_contains: Part of a performer's name.
    """
    return db.get_cast_changes(production_id, person_contains)


# NYT tools: live calls, at most one every 12.5 seconds (nyt.py enforces it), so keep them few.


@function_tool
def get_press(production_id: str, reviews_only: bool = False) -> str:
    """New York Times articles about one production, from about 4 months before its first week to a
    month after its last: headline, date, abstract, link, and whether it's a review. The top results
    by relevance only. Takes about 12 seconds.

    Args:
        production_id: From find_production.
        reviews_only: True for reviews only.
    """
    return nyt.get_press(production_id, reviews_only)


@function_tool
def search_nyt(title: str, start: str | None = None, end: str | None = None, reviews_only: bool = False) -> str:
    """New York Times articles mentioning a show's title and Broadway between two dates. Use it for a
    narrower window than get_press gives (the weeks around an event). Takes about 12 seconds.

    Args:
        title: The show's title, e.g. 'Hamilton'.
        start: Optional earliest publish date, ISO 'YYYY-MM-DD'.
        end: Optional latest publish date, ISO 'YYYY-MM-DD'.
        reviews_only: True for reviews only.
    """
    return nyt.search_articles(title, start, end, reviews_only)


@function_tool
def get_article(title_or_url: str) -> str:
    """Get the text of an English Wikipedia article: for background on a show (its story, its history),
    never for box office numbers or award results, which come from the database tools.

    Args:
        title_or_url: The article's title or URL, e.g. 'Hamilton (musical)' or
            'https://en.wikipedia.org/wiki/Hamilton_(musical)'.
    """
        # Wikipedia adds "(musical)" or "(play)" only when the bare name is taken
    # ("Hamilton (musical)", but plain "Hadestown"), so try the other forms before giving up.
    base = title_or_url.split("/wiki/")[-1].replace("_", " ").strip()
    bare = base.split(" (")[0]
    candidates = [base] + [t for t in (bare, f"{bare} (musical)", f"{bare} (play)") if t != base]
    article, errors = None, []
    for title in candidates:
        try:
            article = wikipedia.fetch_article(title)
            break
        except Exception as e:
            errors.append(f"{title}: {type(e).__name__}")
    if article is None:
        # The model cannot see an exception. Return something it can reason about.
        return json.dumps({"error": f"No Wikipedia article found. Tried: {'; '.join(errors)}"})
    return f"# {article['title']}\n{article['description']}\nsource_url: {article['url']}\n\n{article['text']}"


# --- Sources: each one a Fate can be sent to, with its tools and how to use them ---
# One place to add or rename a source: Glinda's list and the research tool both read from here.

SOURCES = {
    "grosses": {
        "about": "weekly box office (gross, capacity, ticket prices); changes around a date; which shows "
                 "were in trouble in a given week; how Tony wins vs nominations moved box office across shows",
        "tools": [find_production, get_grosses, grosses_around_date, trouble_signals, tony_effect, data_coverage],
        "how": """\
1. You're given a production_id. If not, find it with find_production.
2. For a change around an event (a Tony win, a cast change), call grosses_around_date with the
   event's date, and only that: it already has both windows' averages and the changes. Report the
   show's pct_change and pct_change_vs_market together: Broadway is seasonal, so a change means
   little until it is compared with the market's.
3. For a trend or a period, call get_grosses with start and end, and report its summary plus the
   highest and lowest weeks. Don't list week-by-week rows: the page draws them as a chart.
4. For "which shows are in trouble" in a week, call trouble_signals once. Report the shows with
   level_signals first (at most 5), each with its signals, then any hit with only trend_signals in
   one bullet: a trend signal on a show still selling above the market usually means prices coming
   down from a peak. Give the week and the benchmark it used.
5. For whether Tony wins help more than nominations, call tony_effect once for the category. Report
   both groups' median change against the market, how many productions each had, the gap, its
   warnings, and how many were left out. Say it's an association, not proof the award caused it.
Make at most 4 tool calls.""",
    },
    "awards": {
        "about": "Tony Award nominations, wins and ceremony dates; Wikipedia for a show's background",
        "tools": [find_production, get_awards, get_article],
        "how": """\
1. Look up nominations and wins with get_awards. Report each with its ceremony_date.
2. Use get_article only for background a question asks for (what a show is about), never for
   award results.
Make at most 6 tool calls.""",
    },
    "music": {
        "about": "Broadway cast recordings and track lists; which songs differ between two productions",
        "tools": [find_production, get_recordings, compare_recordings],
        "how": """\
1. You're given the production_ids. If not, find them with find_production.
2. To compare two productions (an original and its revival), call compare_recordings, the earlier one
   first, and report in_both as a count and only_in_a and only_in_b in full: it has already matched
   reprises and medleys, so don't compare track lists yourself. Point out pairs in only_in_a and
   only_in_b that look like the same song renamed.
3. For one production's songs, call get_recordings.
4. If a production has no cast recording, say so: there's nothing to compare.
Make at most 4 tool calls.""",
    },
    "cast": {
        "about": "dated cast changes (a performer joining or leaving), hand-curated for a few shows only",
        "tools": [find_production, get_cast_changes],
        "how": """\
1. You're given a production_id or a performer's name. Find the production with find_production if needed.
2. Call get_cast_changes, and report each change with its start_date and end_date.
3. If nothing is recorded, say the date isn't in the data: never guess one.
Make at most 3 tool calls.""",
    },
    "press": {
        "about": "New York Times coverage (live): reviews and news, with dates; the date a show was "
                 "reviewed; how critics received it",
        "tools": [find_production, get_press, search_nyt],
        "how": """\
1. You're given a production_id. If not, find it with find_production.
2. Call get_press with the production_id. For how critics received a show, or the date of its
   review, set reviews_only to true. Use search_nyt only for a narrower window (the weeks around an
   event) than get_press covers.
3. Report each relevant article as its headline, its pub_date and its abstract in one line, linked
   to its web_url, e.g. [NYT, 2015-08-06: Review: 'Hamilton,' Young Rebels Changing History](web_url).
   For a review's date, give the pub_date of the review itself.
4. Say what the abstracts say; don't characterize a review beyond them. They are a sample of the
   coverage (the top results by relevance), not all of it.
5. Each call takes about 12 seconds: make at most 2 tool calls. If a tool says the NYT isn't
   available (no API key, rate limit), don't call it again: report the gap.""",
    },
}
Source = Literal["grosses", "awards", "music", "cast", "press"]  # keep in step with SOURCES' keys
SOURCE_LIST = "\n".join(f"- {name}: {s['about']}" for name, s in SOURCES.items())


# --- The triage: asks for clarification when the question is too vague to research ---

triage = Agent(
    name="Barfee",  # Triage
    instructions=f"""\
You decide whether a question about Broadway can be researched as asked.

The message starts with today's date and the current Broadway season. Use them, not your own
sense of the date, to work out what "current season", "last season", "this year" or "last
month" mean. When you rewrite the question, replace those phrases with the actual season or dates.

A question is ready when it names a specific show (or a clear group of shows) and a time
period, or when the time period obviously doesn't matter ("What was Hamilton's best week?").

- If it's vague ("How's Wicked doing?"), reply CLARIFY: followed by one short question that offers
  2 or 3 concrete options, e.g. "CLARIFY: Over which years, and do you mean box office or awards?"
- A question about which shows are in trouble or struggling needs a week. If it says "now" or gives
  none, ask, offering the last week of the grosses data as one option, e.g. "CLARIFY: The grosses data
  ends {GROSSES_END}. Do you mean the last weeks of the data, or a specific earlier date?"
- If it's ready, reply RESEARCH: followed by the question, rewritten to be specific.
  Don't add facts or assumptions the user didn't give you.
- If it isn't about Broadway, reply CLARIFY: and say in one sentence what you can help with.
- If the message gives an earlier question and the user's answer to a clarifying question,
  combine them. Don't ask again unless the answer still leaves it unclear which show is meant:
  reply RESEARCH: with your best specific reading.
- Don't ask about dates the data doesn't cover: reply RESEARCH: and the team will say so.

Reply with one line, starting with CLARIFY: or RESEARCH:, and nothing else.

What the data covers:
{COVERAGE}""",
    model=MODEL,
    model_settings=SETTINGS,
)


# --- The researchers: the Fates, one task each, from one source ---

FATE_INSTRUCTIONS = """\
You answer one research task about Broadway, using only your tools.

{how}

Then reply with the facts that answer your task and nothing else, at most 8 bullets.
- Never state a number, date or name you didn't get from a tool result. Copy numbers exactly as
  the tools return them, and never compute one yourself: if you need a change or a comparison,
  call the tool that computes it.
- End every bullet with a markdown link to the source_url in the tool result, labeled with the
  source and the rows it came from, never with a tool's name. For grosses, the weeks: from
  grosses_around_date use before.from to after.to, e.g.
  [grosses, weeks ending 2016-04-17 to 2016-08-07](source_url). For awards, the ceremony, e.g.
  [70th Tony Awards](source_url).
- Only link a source_url that appears in a tool result: never write one yourself. A bullet
  saying data is missing gets no link.
- Pass on the tools' warnings that bear on your task, one bullet each. Don't repeat their notes
  on method (how averages or the market are computed): those are for you, not findings.
- If your tools can't answer part of the task, say so in a last bullet, and why: outside the
  dates covered, not tracked, or show not found.
- If a tool says a source isn't available or isn't configured, don't call it again: report the gap."""

researcher = Agent(
    name="Fates",  # Researcher: research() gives each copy its source's tools and instructions
    instructions=FATE_INSTRUCTIONS.format(how=""),
    model=RESEARCHER_MODEL,
    model_settings=SETTINGS,
)


@function_tool
async def research(ctx: RunContextWrapper[Research], source: Source, task: str) -> str:
    """Send a researcher to answer one task from one source. Returns its findings, each with its
    source link.

    Args:
        source: 'grosses' for weekly box office and changes around a date; 'awards' for Tony
            nominations, wins and ceremony dates, and a show's background.
        task: One specific thing to find out. Include the production_id and any dates you already have.
    """
    if ctx.context.researchers >= MAX_RESEARCHERS:
        return "No researchers left. Reply DONE."
    if source not in SOURCES:
        return f"Unknown source {source!r}. Use one of: {', '.join(SOURCES)}."
    # What researcher.as_tool() does: run the researcher in a fresh context and return its answer.
    # Written out so each run gets a numbered copy, its own column in the context panel, and
    # only the tools and instructions for its source.
    ctx.context.researchers += 1
    agent = researcher.clone(
        name=f"Fate {ctx.context.researchers} ({source})",
        instructions=FATE_INSTRUCTIONS.format(how=SOURCES[source]["how"]),
        tools=SOURCES[source]["tools"],
    )
    result = await Runner.run(
        agent, f"Task: {task}", context=ctx.context, hooks=ctx.context.hooks, max_turns=RESEARCHER_MAX_TURNS
    )
    ctx.context.findings.append(f"[{source}] {task}\n\n{result.final_output}")
    return str(result.final_output)


# --- The lead: identifies the show, splits the question across the Fates, checks for gaps ---

lead = Agent(
    name="Glinda",  # Lead
    instructions=f"""\
You lead a research team that answers questions about Broadway shows from a database. You find
the show yourself; researchers look up everything else, one source each.

The message starts with today's date and the current Broadway season. Use them, not your own
sense of the date, for anything relative ("current season", "recent", "this year"). If the dates
asked about fall outside what the data covers (below), don't send anyone for that part: the
writer will say so.

Sources (pass one as `source` to research):
{SOURCE_LIST}

1. Call find_production for each show the question names.
   - If a title has several candidates and the question doesn't say which (no year, no "revival"),
     reply {CLARIFY} followed by one short question for the user, naming the candidates and their
     years. Then stop. Don't ask when the question is about the productions themselves ("songs cut
     between the original and the revival"): use the earliest and the latest.
   - The database has Broadway productions only. A question about a tour is one to clarify.
2. Questions about many shows at once need one task, not one per show:
   - "Which shows are in trouble / struggling" in a week: one grosses task, "Use trouble_signals
     for week <YYYY-MM-DD>".
   - "Do Tony wins help more than nominations": one grosses task per category asked about, "Use
     tony_effect for category <name>" (Best Musical if none is named), with any years given.
3. For a question about box office around an event, get the date first, then write the grosses task
   as "Use grosses_around_date for <production_id> with date <YYYY-MM-DD>": that is the tool that
   compares the show with the rest of Broadway.
   - A Tony: the ceremony_date from get_awards.
   - A performer joining or leaving: send a cast task for the date first, and the grosses task once
     you have it. If the cast task finds no date, don't guess one: the writer will say it isn't in
     the data, unless the user gave the date themselves.
   - A New York Times review: send a press task for the review's date first (reviews only), and
     the grosses task once you have it.
4. For songs cut or added between productions, send one music task with both production_ids, the
   earlier first. Off-Broadway and London productions aren't in the data.
   For how critics or the press received a show, or news about it, send a press task. Press
   tasks take longer (each NYT lookup is about 12 seconds): send at most two in a run.
5. Split the question into 1 to 4 independent tasks and call research once per task, all in the
   same turn, so the researchers work in parallel. Pick each task's source, and put the
   production_id and any dates in the task. The writer sees only the researchers' findings, never
   your own tool results: if the answer states an award fact (a win, a nomination), send an awards
   task for it too, so it reaches the writer with its source.
6. Read the findings. If part of the question is still unanswered and a source could plausibly
   answer it, call research again for just that part, at most twice.
7. If no source can answer a part (see below), don't send anyone for it: the writer will say so.
8. When the question is covered, reply DONE and nothing else: a writer writes the report.

What the data covers:
{COVERAGE}""",
    model=MODEL,
    model_settings=SETTINGS,
    tools=[find_production, get_awards, research],
)

# --- The writer: one call, after the research ends ---

writer = Agent(
    name="Hamilton",  # Writer
    instructions=f"""\
Write a short report in markdown that answers the question, using only the findings you're given.

- Start with a one-paragraph answer, then a short section per part of the question. Don't repeat
  a number the answer paragraph already gave.
- Keep the links: end every sentence that states a fact with the source link from its finding,
  e.g. ([grosses, weeks ending 2016-04-17 to 2016-08-07](url)). Never link something that isn't
  in the findings, and never add facts of your own, even ones you're sure of.
- Use only numbers that appear in the findings. You may round them for reading ($1,795,970.38 ->
  $1.80 million), but never compute new ones.
- When a finding compares a show with the rest of Broadway, give both numbers: the market's change
  is what tells a real effect from the season.
- For shows in trouble: name the week, lead with the shows selling below the market, and say these
  are warning signs, not predictions of closing.
- For Tony wins vs nominations: give both groups' medians and how many productions each had, and
  say it's an association, not proof the award caused the change.
- For songs between productions: list what was cut and what was added, and say the comparison is
  of the cast albums, which can leave songs out.
- For New York Times coverage: link each article as its finding does, describe reviews only as far
  as their abstracts go, and never quote more than a headline. If the report uses any NYT article,
  end it with the line "Data provided by The New York Times" (the NYT API's terms ask for it).
- Pass on the findings' warnings where they bear on the answer. If findings disagree, say so
  rather than picking one.
- Summarize: no week-by-week lists, and no notes on method beyond one sentence saying what the
  comparison is.
- End with a short section "What the data can't tell us" when the question reaches something the
  findings don't answer (dates past the coverage, profitability or recoupment, audience
  demographics, per-performance numbers, touring productions). Mention each gap once, in plain
  terms a theater fan would understand: say the data isn't available, not why the system failed.
  Sentences about missing data get no link. Don't list limits the question doesn't reach.
- Write plain, specific sentences. Don't claim more significance than the findings do: no
  "profoundly", "pivotal", "paving the way", or "it is worth noting".

What the data covers:
{COVERAGE}""",
    model=MODEL,
    model_settings=SETTINGS,
)


def broadway_season(day: date) -> str:
    """The Broadway season a date falls in, e.g. '2026-27'. Computed here rather than left to the
    model. Seasons turn over in late May; this treats June 1 as the start, so the last week or so
    of May still counts as the old season."""
    start = day.year if day.month >= 6 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _after_clarify(text: str) -> str:
    return text.split(":", 1)[1].strip()


async def deep_research(question: str, hooks: PanelHooks) -> tuple[str, bool]:
    """Returns (text, clarifying): the report, or a clarifying question (from Barfee, or from
    Glinda when a title matches several productions) when clarifying is True."""
    research_run = Research(hooks=hooks)
    # The real date, read on every run: the models have no clock, and the server can run for days
    now = date.today()
    today = f"Today's date: {now.isoformat()} (current Broadway season: {broadway_season(now)})"

    # 0. Triage: a vague question gets a clarifying question back instead of a report
    verdict = str((await Runner.run(triage, f"{today}\n\n{question}", context=research_run, hooks=hooks, max_turns=10)).final_output).strip()
    if verdict.upper().startswith(CLARIFY):
        return _after_clarify(verdict), True
    if verdict.upper().startswith("RESEARCH:"):
        question = _after_clarify(verdict) or question

    # 1. Research: identify the show, fan out, check for gaps
    lead_result = await Runner.run(lead, f"{today}\nQuestion: {question}", context=research_run, hooks=hooks, max_turns=100)
    answer = str(lead_result.final_output).strip()
    if answer.upper().startswith(CLARIFY):
        return _after_clarify(answer), True

    # 2. Write: one call, outside the loop. The Writer sees only the question and the findings.
    findings = "\n\n---\n\n".join(research_run.findings) or "(No findings.)"
    report = await Runner.run(writer, f"{today}\nQuestion: {question}\n\nFindings:\n\n{findings}", hooks=hooks)
    return str(report.final_output), False


# --- FastAPI App ---

HERE = Path(__file__).parent

# run_id -> every agent's context, in order. In-memory, single process. The context panel reads it.
sessions: dict[str, AgentContexts] = {}

app = FastAPI()
add_context_panel(app, sessions)


@app.get("/")
def index():
    # no-cache: browsers otherwise keep serving an old page after a redeploy
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/grosses")
def grosses_series(production_id: str):
    """One production's weekly grosses, for the page's box office chart. Dark weeks are left out."""
    try:
        p = db._production(production_id)
    except KeyError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    rows = db.grosses_rows(production_id)
    return {
        "production_id": production_id,
        "title": p["title"],
        "weeks": [{"week_ending": r["week_ending"], "gross": r["gross"], "capacity_pct": r["capacity_pct"],
                   "preview": r["preview_week"]} for r in rows],
        "source_url": rows[0]["source_url"] if rows else None,
    }


@app.get("/run")
async def run(question: str):
    """Research one question and stream what happens as Server-Sent Events."""
    run_id = str(uuid.uuid4())
    sessions[run_id] = AgentContexts()
    queue: asyncio.Queue = asyncio.Queue()
    hooks = ToolCallHooks(sessions[run_id], queue.put_nowait)

    async def work():
        try:
            text, clarifying = await deep_research(question, hooks)
            # clarifying: text is Barfee's (or Glinda's) question, not a report. The page asks the user, then
            # sends "Original question: ... / User's clarification: ..." as a new run, as /chat does.
            queue.put_nowait({"type": "done", "markdown": text, "clarifying": clarifying})
        except Exception as e:
            # Auth, quota, a model that won't stop: show it on the page, not as a dropped stream.
            queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {str(e)[:500]}"})

    task = asyncio.create_task(work())

    async def events():
        try:
            yield f"data: {json.dumps({'type': 'start', 'run_id': run_id})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event["type"] in ("done", "error"):
                    break
        finally:
            task.cancel()  # the page closed the stream: stop spending tokens

    return StreamingResponse(events(), media_type="text/event-stream")


# --- /chat: one JSON answer, in Project 1's shape ---

# session_id -> the question waiting on a clarification, or None.
# In-memory, single process, like Project 1's sessions.
chat_sessions: dict[str, str | None] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    pending = chat_sessions.get(session_id)

    # A clarifying question went out last turn: this message is the answer, not a new question
    if pending:
        question = f"Original question: {pending}\nUser's clarification: {request.message}"
    else:
        question = request.message

    # The context panel's run log, so a /chat run can be inspected like a /run one
    run_id = str(uuid.uuid4())
    sessions[run_id] = AgentContexts()
    hooks = ToolCallHooks(sessions[run_id], lambda event: None)  # nothing to stream to

    try:
        response, clarifying = await deep_research(question, hooks)
        # Remember the question only while waiting on its clarification
        chat_sessions[session_id] = question if clarifying else None
    except Exception as e:
        # Auth, quota, a model that won't stop: show it in the chat, not as a 500.
        # Leave the session as it was, so the user can resend.
        response = f"Agent run failed: {type(e).__name__}: {str(e)[:300]}"

    return ChatResponse(response=response, session_id=session_id, tool_calls=hooks.tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    chat_sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    # Cloud Run sends traffic to the port in $PORT, on every interface
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))