import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import uvicorn
from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import wikipedia
from context_panel import AgentContexts, PanelHooks, add_context_panel

# WIKIMEDIA_USERNAME and WIKIMEDIA_PASSWORD come from the .env file next to this one.
load_dotenv(Path(__file__).parent / ".env")

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# LiteLLM routes the call to Gemini. api_key is unused: Vertex AI uses your gcloud credentials.
MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash", api_key="unused")
# Researchers only read and take notes: the cheaper model is enough.
RESEARCHER_MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash-lite", api_key="unused")
# A stalled model call would hang the page: give up after 90 s, and retry twice.
SETTINGS = ModelSettings(extra_args={"vertex_location": "global", "timeout": 90, "num_retries": 2})

# The Lead gets this many researchers in all. Prompts ask for restraint; code enforces it.
MAX_RESEARCHERS = 6
# Each Researcher's turn cap. Low, so one Researcher can't eat the 5-minute grading window.
RESEARCHER_MAX_TURNS = 6

# What the data covers and doesn't. The Triage, Lead and Writer prompts all quote it, so the
# agents know their limits. TODO: fill in from your partner's data once it's final.
COVERAGE = """\
- Weekly Broadway grosses (gross, attendance, % capacity, average and top ticket price,
  performances) for weeks ending YYYY-MM-DD to YYYY-MM-DD.
- No data for the COVID shutdown (March 2020 to September 2021).
- Grosses are weekly totals: nothing per performance.
- Not public, so not in the data: profit, running costs, recoupment, audience demographics."""


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


# --- Tools ---
# Every tool should return its data with a citation tag the Writer can copy, e.g.
# [grosses: Hamilton, weeks ending 2019-01-06 to 2019-03-31]. Agree on the format with whoever
# writes the data tools: the Researcher and Writer prompts below rely on it.


@function_tool
def get_article(title_or_url: str) -> str:
    """Get the text of an English Wikipedia article: production history, cast replacements,
    Tony Awards, song lists.

    Args:
        title_or_url: The article's title or URL, e.g. 'Hamilton (musical)' or
            'https://en.wikipedia.org/wiki/Hamilton_(musical)'.
    """
    try:
        article = wikipedia.fetch_article(title_or_url)
    except Exception as e:
        # The model cannot see an exception. Return something it can reason about.
        return json.dumps({"error": f"Could not fetch {title_or_url}: {type(e).__name__}: {e}"})
    links = "\n".join(f"- {link['text']}: {link['url']}" for link in article["links"])
    cite = f"[Wikipedia: {article['title']}]({article['url']})"
    return f"Cite as: {cite}\n\n# {article['title']}\n{article['description']}\n\n{article['text']}\n\n## Links\n{links}"


# TODO: replace these stubs with your partner's real data tools. They say "not available" so
# the agents report a gap instead of inventing numbers while the data isn't ready.


@function_tool
def weekly_grosses(show: str, start: str, end: str) -> str:
    """Weekly box office for one show: gross, attendance, % capacity, average ticket price.

    Args:
        show: The show's name, e.g. 'Hamilton'.
        start: First week ending to include, YYYY-MM-DD.
        end: Last week ending to include, YYYY-MM-DD.
    """
    return json.dumps({"error": "The grosses data isn't connected yet."})


@function_tool
def search_cast_recordings(show: str) -> str:
    """Cast recordings for a show, with their track lists.

    Args:
        show: The show's name, e.g. 'Hadestown'.
    """
    return json.dumps({"error": "The cast recordings source isn't connected yet."})


@function_tool
def search_press(query: str, start: str = "", end: str = "") -> str:
    """News articles and reviews: headline, date, link.

    Args:
        query: What to search for, e.g. 'Hadestown review'.
        start: Earliest publish date, YYYY-MM-DD, or empty for no limit.
        end: Latest publish date, YYYY-MM-DD, or empty for no limit.
    """
    return json.dumps({"error": "The press source isn't connected yet."})


# Each source a Researcher can be sent to, and the only tools it gets there.
SOURCES = {
    "grosses": [weekly_grosses],
    "wikipedia": [get_article],
    "music": [search_cast_recordings],
    "press": [search_press],
}


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

- If it's vague ("How's Wicked doing?", "Which shows are struggling?"), reply
  CLARIFY: followed by one short question that offers 2 or 3 concrete options, e.g.
  "CLARIFY: Do you mean the Broadway production or the tour, and over which years?"
- If it's ready, reply RESEARCH: followed by the question, rewritten to be specific.
  Don't add facts or assumptions the user didn't give you.
- If it isn't about Broadway, reply CLARIFY: and say in one sentence what you can help with.
- If the message gives an earlier question and the user's answer to your clarifying question,
  combine them. Don't ask again unless the answer still leaves it unclear which show is meant:
  reply RESEARCH: with your best specific reading.

Reply with one line, starting with CLARIFY: or RESEARCH:, and nothing else.

What the data covers:
{COVERAGE}""",
    model=MODEL,
    model_settings=SETTINGS,
)


# --- The researcher: one task, from one source ---

researcher = Agent(
    name="Fates",  # Researcher
    instructions="""\
You answer one research task about Broadway, using only your tools.

1. Call your tools to get the data. Never state a number, date or name you didn't get from a
   tool result. If a show isn't found, try another spelling once before giving up.
2. Reply with the facts that answer your task and nothing else, at most 8 bullets. Each bullet
   states one fact and ends with the citation tag from the tool result, copied exactly, e.g.
   [grosses: Hamilton, weeks ending 2019-01-06 to 2019-03-31] or
   [Wikipedia: Hamilton (musical)](https://en.wikipedia.org/wiki/Hamilton_(musical)).
3. Prefer comparisons to raw numbers when the task asks about change: "gross rose 18% in the 4
   weeks after the opening vs the 4 weeks before."
4. If your tools can't answer part of the task, say so in a last bullet, and say why: outside
   the dates covered, not tracked, show not found, or the source isn't available.
5. Only use a citation tag that appears in a tool result. Never build one yourself from the
   examples above. A bullet saying data is missing gets no citation.
6. If a tool says the source isn't available or isn't configured, don't call it again: report
   the gap.""",
    model=RESEARCHER_MODEL,
    model_settings=SETTINGS,
)


@function_tool
async def research(ctx: RunContextWrapper[Research], task: str, source: str) -> str:
    """Send a researcher to answer one task from one source. Returns its findings, each with
    a citation.

    Args:
        task: One specific thing to find out, e.g. "Hamilton's weekly gross and capacity in the
            8 weeks before and after the June 2016 Tony Awards".
        source: Where to look: 'grosses', 'wikipedia', 'music' or 'press'.
    """
    if ctx.context.researchers >= MAX_RESEARCHERS:
        return "No researchers left. Reply DONE."
    if source not in SOURCES:
        return f"Unknown source {source!r}. Use one of: {', '.join(SOURCES)}."
    # What researcher.as_tool() does: run the researcher in a fresh context and return its answer.
    # Written out so each run gets a numbered copy, its own column in the context panel, and
    # only the tools for its source.
    ctx.context.researchers += 1
    agent = researcher.clone(name=f"Fate {ctx.context.researchers} ({source})", tools=SOURCES[source])
    result = await Runner.run(
        agent, f"Task: {task}", context=ctx.context, hooks=ctx.context.hooks, max_turns=RESEARCHER_MAX_TURNS
    )
    ctx.context.findings.append(f"Task ({source}): {task}\n\n{result.final_output}")
    return str(result.final_output)


# --- The lead: plans by source, fans out to researchers, and checks for gaps ---

lead = Agent(
    name="Glinda",  # Lead
    instructions=f"""\
You lead a research team that answers questions about Broadway shows. You don't look anything
up yourself: researchers do, one source each.

The message starts with today's date and the current Broadway season. Use them, not your own
sense of the date, for anything relative ("current season", "recent", "this year"). If the dates asked about fall outside what
the data covers (below), don't send anyone for that part: the writer will say so.

Sources (pass one as `source` to research):
- grosses: weekly box office, attendance, % capacity, ticket prices
- wikipedia: production history, opening and closing dates, cast and cast replacements,
  Tony Awards, song lists
- music: cast recordings and their track lists
- press: news coverage and reviews, with dates

1. Split the question into 2 to 4 tasks, each answerable from ONE source. Call research once
   per task, all in the same turn, so the researchers work in parallel. Make each task
   specific: name the show, the dates and the numbers you need.
   Some questions need two steps, e.g. "did grosses rise after X joined the cast?": first find
   the date X joined (wikipedia), then compare grosses around that date (grosses). Send the
   first step now and the second once you have the date.
2. Read the findings. If part of the question is still unanswered and another source could
   plausibly answer it, call research again for just that part. Do this at most twice.
3. If no source can answer a part (see what the data covers below), don't send anyone for it:
   the writer will say it can't be answered.
4. When the question is covered, reply DONE and nothing else: a writer writes the report.

What the data covers:
{COVERAGE}""",
    model=MODEL,
    model_settings=SETTINGS,
    tools=[research],
)

# --- The writer: one call, after the research ends ---

writer = Agent(
    name="Hamilton",  # Writer
    instructions=f"""\
Write a short report in markdown that answers the question, using only the findings you're given.

- Start with a 2 to 3 sentence direct answer, then a section per part of the question.
- End every sentence that states a fact with its citation tag, copied exactly from the
  findings. Never cite something that isn't in the findings, and never add facts or numbers
  of your own, even ones you're sure of.
- If findings disagree, say so rather than picking one.
- End with a section "What the data can't tell us": the parts of the question the findings
  didn't answer, and why (outside the dates covered, not tracked, not public).
- Mention each gap once, in "What the data can't tell us", in plain terms a theater fan would
  understand: say the data isn't available, not why the system failed. Sentences about missing
  data get no citation.
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


async def deep_research(question: str, hooks: PanelHooks) -> tuple[str, bool]:
    """Returns (text, clarifying): the report, or Barfee's clarifying question when clarifying is True."""
    research_run = Research(hooks=hooks)
    # The real date, read on every run: the models have no clock, and the server can run for days
    now = date.today()
    today = f"Today's date: {now.isoformat()} (current Broadway season: {broadway_season(now)})"

    # 0. Triage: a vague question gets a clarifying question back instead of a report
    verdict = str((await Runner.run(triage, f"{today}\n\n{question}", context=research_run, hooks=hooks, max_turns=1)).final_output).strip()
    if verdict.upper().startswith("CLARIFY:"):
        return verdict.split(":", 1)[1].strip(), True
    if verdict.upper().startswith("RESEARCH:"):
        question = verdict.split(":", 1)[1].strip() or question

    # 1. Research: plan, fan out, check for gaps
    await Runner.run(lead, f"{today}\nQuestion: {question}", context=research_run, hooks=hooks, max_turns=12)

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


@app.get("/run")
async def run(question: str, url: str = ""):  # url is unused now: kept so the old page still works
    """Research one question and stream what happens as Server-Sent Events."""
    run_id = str(uuid.uuid4())
    sessions[run_id] = AgentContexts()
    queue: asyncio.Queue = asyncio.Queue()
    hooks = ToolCallHooks(sessions[run_id], queue.put_nowait)

    async def work():
        try:
            report, _ = await deep_research(question, hooks)
            queue.put_nowait({"type": "done", "markdown": report})
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

# session_id -> the question Barfee is waiting on a clarification for, or None.
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

    # Barfee asked a clarifying question last turn: this message is the answer, not a new question
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