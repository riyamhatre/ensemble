# Box Office Detective

A deep research agent that answers questions about Broadway shows from a local database of weekly box office grosses, Tony Awards and cast recordings. Every number in a report comes from a database query, and every sentence that states a fact links to the source row it came from.

## Target Users

- **Theater fans** who want the numbers behind the gossip: Did a show's box office really rise after its Tony win, or after a rave review?
- **Producers and analysts** to read the market: which shows are selling below the rest of Broadway? How does a show compare with every other show playing the same weeks?

The second point is the one a model can't do from memory. Broadway is seasonal (summers and holidays are strong, January is weak), so a show's change means little until it's compared with the market's. The agent always reports both.

## Structure

Four agents, each with its own job, plus researchers sent to one data source at a time:

1. **Barfee (triage)** decides whether a question can be researched as asked. A vague one ("How's Wicked doing?") gets a clarifying question back, with concrete options, instead of a guess.
2. **Glinda (lead)** identifies the production (one title can be an original and several revivals), splits the question into tasks, and sends them to researchers in parallel.
3. **The Fates (researchers)**, one per task. Each copy gets the tools for one source only: `grosses`, `awards`, `music`, `cast` or `press`. They copy numbers from tool results and never compute their own.
4. **Hamilton (writer)** sees only the question and the researchers' findings, and writes the cited report. It says plainly what the data can't answer.

All of the math (averages, percent changes, comparisons with the market, medians across shows) happens in Python and SQL in [broadway_data.py](broadway_data.py), never in the model.

**To see the full trace, click the Backstage button** at the top right of the page. It shows every agent's exact context as it works, as a timeline with one column per agent. The subagents' tool calls there carry more detail than the activity list on the main page: which tool each agent called, with what arguments, and what came back.

## Running Locally

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/getting-started/installation/), a free Wikimedia Enterprise account, and Google Cloud credentials for Vertex AI. An NYT API key is optional.

1. Clone the repo and enter it:
   ```bash
   git clone https://github.com/riyamhatre/ensemble.git
   cd ensemble
   ```

2. Set up a GCP project with billing and the Agent Platform API enabled (older docs and the endpoint itself still call it Vertex AI). Model calls go to `gemini-3.5-flash` for Barfee, Glinda and Hamilton, and `gemini-3.5-flash-lite` for the Fates.

3. Authenticate with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```
   The app uses your gcloud default project, so make sure it's set to the project from step 2:
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

4. Create a [NYT account](https://help.nytimes.com/115010905707-Log-in-to-Your-New-York-Times-Account) (used for news articles) and copy `.env.example` to `.env` in the project root:
   ```
   NYT_API_KEY=your_key_here 
   ```
   Without `NYT_API_KEY`, the press source reports that it isn't available.

5. Start the app:
   ```bash
   uv run app.py
   ```
   Then open http://localhost:8000.

Try: *"Did Hamilton's grosses rise after it won Best Musical?"*

> The database ([data/broadway.db](data/broadway.db)) is committed, so the app runs without fetching any data. Grosses, awards and music answers all come from that file; only show background (Wikipedia) and press (NYT) are looked up live.

Endpoints:

| Endpoint | What it does |
|---|---|
| `GET /` | The page. |
| `GET /run?question=...` | Runs one question and streams every step as Server-Sent Events. |
| `POST /chat` | `{"message": ..., "session_id": ...}` returns `{"response", "session_id", "tool_calls"}`. A clarifying question waits on the session for the user's answer. |
| `GET /api/grosses?production_id=...` | One production's weekly grosses, for the page's chart. |

## Where the data came from

Everything is in one SQLite file, [data/broadway.db](data/broadway.db), built from the staging CSVs in [data/csv/](data/csv/). Both are committed, so the app runs without fetching anything.

| Source | What it gives | Coverage | Tables | Ingest script |
|---|---|---|---|---|
| [Tidy Tuesday Broadway grosses](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28) (2020-04-28), credited to Playbill | Weekly gross, potential gross, attendance, capacity, average and top ticket, performances | Weeks ending 1985-06-09 to 2020-03-01 | `grosses`, `productions` | [ingest/grosses.py](ingest/grosses.py) |
| [IST719 BroadwayGrosses](https://github.com/XimengDeng/IST719_BroadwayGrosses), also scraped from Playbill | The same weekly figures, without potential gross | Weeks ending 2020-03-08 to 2023-03-12, with a gap for the COVID shutdown (no weeks between 2020-03-08 and the August 2021 reopening) | `grosses` | [ingest/grosses.py](ingest/grosses.py) |
| Wikipedia's per-ceremony pages ("70th Tony Awards"), through the MediaWiki parse API | Every nominee and winner in the competitive categories, plus each ceremony's date | Ceremonies 1985 to 2026 | `tony_ceremonies`, `awards` | [ingest/tonys.py](ingest/tonys.py) |
| [MusicBrainz](https://musicbrainz.org/) web service (core data is CC0) | Broadway cast recordings and their track lists | 237 recordings for 227 productions | `recordings`, `tracks` | [ingest/musicbrainz.py](ingest/musicbrainz.py) |
| New York Times Article Search API | Headlines, abstracts and dates of reviews and news | Looked up live, never stored | none | [nyt.py](nyt.py) |
| Hand-curated CSV (optional) | Dated cast changes, for the star-casting question | Empty unless `data/csv/cast_changes_manual.csv` exists | `cast_changes` | [build_db.py](build_db.py) |

The database holds 1,362 productions, 49,644 weekly gross rows, 4,028 Tony nominee rows and 5,316 tracks. Its `data_sources` table records what each source covers; the agents read it so they can say where the data ends.

A few notes on sources:
- **The two grosses datasets agree.** Where they overlap (2018-08 to 2020-03, 2,557 show-weeks), every gross, attendance, ticket price and performance count matches exactly.
- **Productions are matched across sources by title and date**, because the sources share no IDs. The provisional match rules are labeled blocks in [build_db.py](build_db.py). About 94% of 1985–2021 Tony nominees link to a production in the grosses data.
- **NYT content is never stored.** The NYT API's terms forbid keeping content longer than 24 hours, or storing it to give to others. So [nyt.py](nyt.py) looks articles up when a question needs them and keeps them only in memory, for at most 23 hours.
- **IBDB isn't used.** Its terms page refuses automated readers. That's why there is no source of production dates, categories or cast changes beyond what's listed above.

### Rebuilding the dataset from scratch

The scripts fetch politely, at most one request per second per site, through a disk cache in `data/raw/` (gitignored), so a rerun doesn't fetch anything twice.

```bash
# BROADWAY_CONTACT in .env: an email or project URL for the User-Agent, which MusicBrainz requires
uv run ingest/grosses.py       # both grosses datasets -> data/csv/grosses_weekly.csv, grosses_runs.csv
uv run ingest/tonys.py         # 41 Wikipedia pages -> tony_ceremonies.csv, tony_awards.csv
uv run ingest/musicbrainz.py   # about 25 minutes the first time -> recordings.csv, tracks.csv
uv run build_db.py             # data/csv/*.csv -> data/broadway.db, with checks
uv run broadway_data.py        # smoke test of every query function against the database
```

### What the data cannot answer

The report says so when a question reaches any of these:
- box office after 2023-03-12,
- profitability or recoupment,
- audience demographics,
- per-performance numbers (grosses are weekly),
- touring and off-Broadway productions.

## Three sample queries

Each query below was run through the agent (`POST /chat`) and, separately, through the same model with no tools and no system prompt, on 2026-11-09. The no-tools answers came from a single user message to `vertex_ai/gemini-3.5-flash`, the model the agent's lead and writer use:

```python
import litellm
r = litellm.completion(model="vertex_ai/gemini-3.5-flash", vertex_location="global",
                       messages=[{"role": "user", "content": question}])
print(r.choices[0].message.content)
```

Model output varies from run to run. The expected answers below come straight from the query functions, so those numbers stay the same every time.

### 1. "Did Hamilton's grosses rise after it won Best Musical?"

**Expected answer:** yes. In the 8 weeks after the 2016-06-12 ceremony, compared with the 8 weeks before:
- Hamilton's average weekly gross rose **15.1%**, from $1.80M to $2.07M.
- The same-store market (the 22 other shows playing every one of those weeks) rose **6.9%**, so Hamilton beat it by **8.2 points**.
- Capacity stayed flat at **101.7%**, because the show was already sold out. The rise came from the average ticket price (+13.4%, against +2.9% for the market).
- Warning: potential gross was reset +45.1% in the same period, so gross-to-potential isn't a fair measure here.

**Where it comes from:**
- `grosses` table (Tidy Tuesday rows), through `grosses_around_date("hamilton-2015", "2016-06-12")` in [broadway_data.py](broadway_data.py).
- The ceremony date comes from `tony_ceremonies` (Wikipedia, [70th Tony Awards](https://en.wikipedia.org/w/index.php?title=70th_Tony_Awards&oldid=1380685596)), through `get_awards`.

| Agent (tools) | Same model, no tools |
|---|---|
| Gives the expected numbers exactly, each linked to its source rows. Compares Hamilton with the 22-show market. Notes the potential-gross reset. Says the rise came from ticket prices, with capacity unchanged. | Also says yes, and that the rise came from prices because the show was sold out. Its weekly figures are close to the database: it says "$1.5–1.8M" in spring (the data says $1.68–1.92M) and a "$2.3M" July peak (the data says $2.28M in the week ending 2016-07-17, a nine-performance week). It has no market comparison, so it can't say how much of the rise was the season. It also says the $849 premium price "took effect shortly after the Tony win". In the weekly data, the top ticket stays at $475 through the eight weeks after the win, and first rises, to $549, in the week ending 2016-09-25. |

<details><summary>Agent's full answer</summary>

Yes, Hamilton's average weekly gross rose by 15.1% after it won the Tony Award for Best Musical on June 12, 2016, outperforming the rest of the Broadway market ([70th Tony Awards](https://en.wikipedia.org/w/index.php?title=70th_Tony_Awards&oldid=1380685596); [grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). This rise was primarily driven by a 13.4% increase in the average ticket price, while the show's capacity remained completely unchanged at 101.7% ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### Grosses and Ticket Prices

The comparison looks at grosses in the 8-week windows before and after the Tony ceremony ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). In the 8 weeks before the event, Hamilton averaged $1,795,970.38 per week, whereas it averaged $2,067,969.69 per week in the 8 weeks following the win ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). Meanwhile, the market of 22 other shows experienced a 6.9% change in average weekly gross ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). 

Hamilton's average ticket price was $167.08 before the ceremony and increased to $189.52 afterward ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). In comparison, the rest of the market only saw a 2.9% change in average ticket price ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### Capacity and Potential Gross

While Hamilton's capacity stayed constant, the rest of the market had a 1.9% change in capacity percentage ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). The show's gross-to-potential ratio was 1.345 before the event and 1.067 afterward ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). However, the show's average potential gross changed by 45.1% because producers reset it, meaning that gross-to-potential metrics are less reliable and weekly grosses should be compared instead ([grosses, weeks ending 2016-04-17 to 2016-08-07](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### What the data can't tell us

This data cannot tell us about Hamilton's profitability or recoupment. We also lack data on audience demographics and per-performance numbers. Finally, because the dataset only covers Broadway, details on touring productions or grosses past the covered dates are not available.

</details>

<details><summary>No-tools answer, in full</summary>

**Yes, *Hamilton*’s box office grosses did rise after it won the Tony Award for Best Musical on June 12, 2016.** However, because the show was already an unprecedented, sold-out blockbuster before the awards, the way its grosses increased was unique. 

Since the show was already operating at over 100% capacity (including standing-room tickets) every week, it could not sell more tickets. Instead, the rise in grosses was driven by **increased ticket prices, specifically premium pricing.**

Here is how the post-Tony win affected *Hamilton*’s box office:

#### 1. The Immediate Post-Tony Price Hike
Just days before the Tony Awards, anticipating a massive sweep (the show went on to win 11 Tonys), the producers announced a historic hike in ticket prices. 
* **Premium Tickets:** The price for premium seats was raised from $475 to a record-breaking **$849**. 
* **Regular Tickets:** Regular tickets were also raised to run from $179 to $199.
* This strategy was designed to combat secondary-market scalpers and keep more of the astronomical profits in the hands of the production and its creators.

#### 2. The Rise in Weekly Grosses
These new pricing structures took effect shortly after the Tony win, leading to a noticeable jump in weekly box office numbers:
* **Before the Tonys (Spring 2016):** *Hamilton* was grossing roughly **$1.5 million to $1.8 million** per week. 
* **Immediately After the Tonys (June/July 2016):** Weekly grosses surged past **$2 million** per week, peaking at $2.3 million in early July as fans rushed to see the original cast (including Lin-Manuel Miranda) before their contracts ended.
* **Fall/Winter 2016:** Even after the original cast departed, demand remained so high that the $849 premium seats regularly sold out. By Thanksgiving week in November 2016, *Hamilton* broke the Broadway record for the highest-grossing week in history, pulling in **$3.3 million** for an eight-performance week. 

Ultimately, while the Tony Award for Best Musical acts as a vital marketing boost to help struggling or moderately successful shows find an audience, for *Hamilton* it served as the catalyst to officially crown it as the most expensive and highest-grossing ticket in Broadway history.

</details>

### 2. "Do Tony wins help box office more than nominations for Best Musical?"

**Expected answer:** yes, on average.
- **Method:** this compares each nominee's average weekly gross in the 8 weeks after its ceremony with the 8 weeks before, minus the same-store market's change.
- **Winners:** the 34 Best Musical winners had a median change of **+14.05 points** against the market, and 82% beat it.
- **Nominated only:** the 75 shows that were nominated but didn't win had a median of **+7.3 points**, and 72% beat the market.
- **The gap** between the two medians is **6.8 points**.
- **Left out:** 38 productions, mostly shows that closed within 8 weeks of the ceremony and the 1985 nominees that predate the data. The September 2021 ceremony is left out too, because no grosses exist before it (the COVID shutdown).
- **Caveat:** this is an association, not proof that the award causes the rise.

**Where it comes from:** the `awards` and `tony_ceremonies` tables (Wikipedia) joined with the `grosses` table, through `tony_effect("Best Musical")` in [broadway_data.py](broadway_data.py).

| Agent (tools) | Same model, no tools |
|---|---|
| Gives the expected medians, counts, share that beat the market, the 6.8-point gap and the 38 exclusions. Says it's an association, not proof. | Says yes, citing "industry consensus". It explains why a nomination helps (the TV broadcast slot) and why a win matters long-term (marketing, tours, licensing), and gives examples like *Wicked* losing to *Avenue Q* in 2004. It gives no measured effect: no numbers on how much grosses move, or how many shows the claim rests on. It also makes claims that can't be checked here, such as "$1 million to $2 million in single-day ticket sales" the Monday after a win. |

<details><summary>Agent's full answer</summary>

Best Musical winners generally see a larger gross increase relative to the market than productions that are only nominated ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). This difference is also apparent in their average grosses and theater capacities relative to the market ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). However, these trends demonstrate an association rather than proof that winning a Tony Award causes an increase in sales, as winners are often already stronger sellers and nominations themselves can lift a show's grosses before the ceremony takes place ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### Tony Wins vs. Nominations

The comparison evaluates how the weekly grosses and capacities of Best Musical nominees and winners change relative to the overall Broadway market ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). Among Best Musical nominees, the 34 productions that won the award had a median gross change vs the market of 14.05%, with 82% of these shows beating the market ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). In comparison, the 75 nominated-only productions had a median gross change vs the market of 7.3%, with 72% of them beating the market ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). The median gap in gross change vs the market between winners and nominated-only productions was 6.8 percentage points ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). 

In terms of mean changes, winners achieved a mean gross change vs the market of 23.2% and a median capacity change vs the market of 3.45% ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). Meanwhile, nominated-only productions had a mean gross change vs the market of 15% and a median capacity change vs the market of 2% ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). These differences represent an association and not proof that the Tony win caused the change, because winning shows are often already stronger sellers and nominations alone can lift a show before the ceremony ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### Data Exclusions and Limitations

Because weekly grosses in this analysis end on March 12, 2023, a ceremony is only measured when its eight weeks after are all before this date, which excluded 38 productions from the dataset ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). Additionally, no ceremony is measured across the COVID shutdown, meaning the September 2021 ceremony is excluded because it has no weeks of gross data before it ([tidytuesday, 2020-04-28](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### What the data can't tell us

While these findings outline historical trends for Broadway productions, several details are not available in the data:

* Grosses data is not available for any dates past the coverage ending on March 12, 2023.
* Information regarding show profitability or recoupment is not available.
* The data does not include audience demographics.
* Because grosses are only reported weekly, per-performance numbers are not available.
* The dataset is limited to Broadway shows and does not include touring productions.

</details>

<details><summary>No-tools answer, in full</summary>

In the Broadway industry, the general consensus is that **a Tony win for Best Musical helps the box office significantly more than a nomination alone**, particularly in the long run. 

However, the dynamic between nominations and wins is nuanced. A nomination is crucial for short-term survival, while a win is the key to long-term profitability and national touring power.

Here is a breakdown of how nominations versus wins impact the box office.

---

#### 1. The Power of the Nomination: Short-Term Survival
For a struggling or moderately successful show, a nomination for Best Musical is a lifesaver. 

* **The TV Broadcast Performance:** The biggest benefit of a Best Musical nomination is not the nod itself, but the fact that it secures the show a performance slot on the live, nationally televised Tony Awards broadcast. This 3-to-4-minute performance acts as a multi-million-dollar commercial to millions of potential tourists. 
* **The "Nomination Bounce":** In the six weeks between the nomination announcement in May and the ceremony in June, nominated shows usually see a significant spike in ticket sales. 
* **Validation:** For tourists planning summer trips, a Best Musical nomination is a seal of approval that makes them comfortable buying expensive tickets.

**The catch:** If a show is nominated but does not win, the box office boost usually fades by the end of the summer, and many non-winning nominees close by the fall.

---

#### 2. The Power of the Win: Long-Term Longevity
Winning the Tony for Best Musical is the ultimate prize because it transforms a show from a "Broadway hit" into a global brand.

* **The "Morning-After" Surge:** The Monday after the Tony Awards, the winner of Best Musical almost always experiences a massive, immediate box office surge—often pulling in $1 million to $2 million in single-day ticket sales.
* **The Infinite Marketing Hook:** The phrase **"Winner of the Tony Award for Best Musical"** can be used in marketing materials forever. It is the most powerful marketing tagline in commercial theater.
* **The National Tour (Where the Real Money is):** This is where a win vastly outperforms a nomination. Touring presenters in cities across North America (like Chicago, Los Angeles, and Atlanta) heavily rely on the "Best Musical Winner" label to sell subscription packages. A winning show can tour successfully for a decade or more, whereas a nominee that lost has a much harder time booking lucrative touring contracts.
* **International and Licensing Rights:** A Best Musical win makes the show highly attractive to international producers (West End, Australia, Germany) and ensures high licensing fees from amateur and school theatrical groups for decades.

---

#### The Exceptions: When a Win Doesn't Matter
While the win is generally more lucrative, there are famous exceptions where nominees (or even shows that were snubbed) outperformed the winners:

* **The *Wicked* vs. *Avenue Q* (2004) anomaly:** *Avenue Q* won Best Musical in a major upset. While the win kept the puppet musical running for years, the runner-up, *Wicked*, became one of the highest-grossing cultural phenomena in Broadway history. *Wicked* didn't need the win because it already had massive word-of-mouth appeal.
* **The *Chicago* vs. *A Chorus Line* (1976) anomaly:** *A Chorus Line* swept the Tonys, leaving the brilliant *Chicago* empty-handed. However, *Chicago*’s eventual 1996 revival became the longest-running American show in Broadway history.
* **Pre-sold Blockbusters:** Shows like *The Lion King* (which did win) or *Frozen* (which was nominated but lost) have massive Disney brand recognition. Their box office is driven by the brand, not the awards.

#### Summary
* **Nominations** keep a show alive through the spring and summer by granting them national television exposure.
* **Wins** secure a show’s financial future for the next 10 to 20 years, driving massive ticket sales on Broadway, generating highly lucrative national tours, and cementing its legacy for international markets.

</details>

### 3. "Which shows were in trouble in early August 2019?"

**Expected answer:** for the 8 weeks ending 2019-08-04, these shows sold well below the market. The market averaged 90.6% capacity, and in the five years before, shows that closed sat a median of 5 points under it:

| Show | Capacity | Points below the market |
|---|---|---|
| *Pretty Woman: The Musical* | 73.2% | 17.9 |
| *Beautiful: The Carole King Musical* | 73.3% | 17.8 |
| *The Cher Show* | 74.0% | 17.2 |
| *Waitress* | 74.7% | 16.3 |
| *King Kong* | 75.2% | 16.1 |
| *Tootsie* | 75.3% | 15.9 |
| *Be More Chill* | 77.2% | 13.7 |

*Manilow Broadway*, a two-week limited engagement, and *Chicago* also show warning signs. The flags are warning signs, not predictions, but the database's own last weeks of grosses bear them out:
- *Be More Chill* closed the week ending 2019-08-11.
- *The Cher Show*, *King Kong* and *Pretty Woman* all closed 2019-08-18.
- *Beautiful* closed 2019-10-27.
- *Tootsie* and *Waitress* closed 2020-01-05.

**Where it comes from:** the `grosses` table (Tidy Tuesday rows), through `trouble_signals("2019-08-04")` in [broadway_data.py](broadway_data.py). Its benchmark is 158 productions that closed between 2014-08-05 and 2019-08-04.

| Agent (tools) | Same model, no tools |
|---|---|
| Names the expected shows with their capacity and gross-to-potential against the market and against the closing benchmark, and says they're warning signs, not predictions. In this run it read "early August" as two weeks, and reported the weeks ending August 4 and 11 separately. | Took "shows" to mean **television**: *The Rookie*, *BH90210*, *The OA*'s cancellation and others. It names no Broadway show. |

<details><summary>Agent's full answer</summary>

In early August 2019, several Broadway shows fell significantly below market benchmarks, indicating they were struggling, though these are warning signs and not predictions of closing ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28); [grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). During the weeks ending August 4 and August 11, productions such as *Waitress*, *Beautiful: The Carole King Musical*, *Pretty Woman: The Musical*, *The Cher Show*, *King Kong*, and *Tootsie* registered capacities and gross-to-potential ratios well below the broader Broadway market ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28); [grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). Additional shows including *Be More Chill*, *Manilow Broadway*, and *Chicago* also displayed warning signals ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### Week Ending August 4, 2019

For the week ending August 4, 2019, multiple shows performed below market levels, serving as warning signs rather than predictions of closing ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). The analysis compares each show's weekly metrics against the broader Broadway market benchmarks. During this week, Broadway's same-store market benchmark averaged 90.6% capacity and a 0.808 gross-to-potential, while closing shows typically had a median capacity of -5% and a median gross-to-potential of -0.184 relative to the market ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). 

Specifically, *Pretty Woman: The Musical* reported a capacity of 73.2% (representing a -17.9 point gap versus the market) and a gross-to-potential of 0.45 (a -0.369 gap versus the market) ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *Beautiful: The Carole King Musical* recorded a capacity of 73.3% (-17.8 points versus the market) and a gross-to-potential of 0.517 (-0.298 versus the market), while its average weekly gross declined 32.0% year-on-year compared to the market's decline of -16.6% ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *The Cher Show* saw a capacity of 74.0% (-17.2 points versus the market) and a gross-to-potential of 0.534 (-0.284 versus the market) ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *Waitress* posted a capacity of 74.7% (-16.3 points versus the market) and a gross-to-potential of 0.534 (-0.28 versus the market), while its average weekly gross fell 35.0% year-on-year (compared to the market's -16.6% decline) and its average ticket price fell 17.9% (compared to the market's -12.5% decline) ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *King Kong* recorded a capacity of 75.2% (-16.1 points versus the market) and a gross-to-potential of 0.51 (-0.308 versus the market) ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). Additionally, level warning signals were displayed by *Tootsie*, *Be More Chill*, *Manilow Broadway*, and *Chicago* ([tidytuesday, weeks ending 2019-06-16 to 2019-08-04](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### Week Ending August 11, 2019

For the week ending August 11, 2019, multiple shows continued to perform below the market, representing warning signs and not predictions of closing ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). The benchmark for this period utilized 159 closed productions from 2014-08-12 to 2019-08-11, where closing shows' median capacity versus the market was -5% and gross-to-potential versus the market was -0.182 ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). 

*Beautiful: The Carole King Musical* recorded a 72.5% capacity (which is -19.2 points versus a market average of 91.7%, performing worse than the closing median of -5.0%) and a gross-to-potential of 0.511 (versus the market's 0.825, worse than the closing median gap of -0.182), alongside average weekly grosses that fell 31.1% on a year earlier compared to the market's decline of -16.6% ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *Waitress* reported a capacity of 76.8% (which is -14.8 points versus the market's 91.6%, worse than the closing median of -5.0%) and a gross-to-potential of 0.552 (versus the market's 0.824, worse than the closing median gap of -0.182), while its average weekly gross declined 31.9% on a year earlier compared to the market's decline of -16.6% ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *Tootsie* saw a capacity of 74.2% (which is -17.7 points versus the market's 91.9%, worse than the closing median of -5.0%) and a gross-to-potential of 0.554 (versus the market's 0.829, worse than the closing median gap of -0.182) ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *King Kong* recorded a capacity of 74.9% (which is -17.1 points versus the market's 92.0%, worse than the closing median of -5.0%) and a gross-to-potential of 0.505 (versus the market's 0.829, worse than the closing median gap of -0.182) ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). *The Cher Show* posted a capacity of 75.0% (which is -16.8 points versus the market's 91.8%, worse than the closing median of -5.0%) and a gross-to-potential of 0.538 (versus the market's 0.828, worse than the closing median gap of -0.182) ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)). These metrics represent warning signs and not predictions of closing, since limited engagements close on schedule regardless of grosses, and even strong shows can close if their theaters are booked ([grosses, weeks ending 2019-06-23 to 2019-08-11](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-04-28)).

#### What the data can't tell us

While we can evaluate these warning signals, the data cannot tell us if these productions eventually achieved overall financial profitability or recoupment. We also lack information on audience demographics and specific per-performance numbers. Finally, the data does not cover touring productions.

</details>

<details><summary>No-tools answer, in full</summary>

In early August 2019, several television shows were "in trouble," whether due to low ratings on the summer "bubble," sudden cancellations, or severe behind-the-scenes controversies. 

Here is a breakdown of the shows that were making headlines for being in trouble during that specific time:

#### 1. Behind-the-Scenes & PR Trouble
*   **_The Rookie_ (ABC):** On August 4, 2019, series co-star Afton Williamson announced on Instagram that she would not be returning for Season 2. She alleged that she had experienced racial discrimination, sexual harassment, and assault on set, and that her repeated complaints to showrunners and HR had been ignored. This threw the high-profile ABC drama into a major PR crisis just as production for the second season was underway.
*   **_BH90210_ (Fox):** The highly anticipated, meta-revival of *Beverly Hills, 90210* premiered on August 7, 2019. While it debuted to decent ratings, it was heavily plagued by behind-the-scenes drama. Just before its August launch, the original showrunner and several head writers quit due to clashes with some of the lead actresses. The intense behind-the-scenes fighting signaled to many that the series was a one-and-done "event" rather than a sustainable long-term show (it was canceled that November).

#### 2. Summer Shows in Ratings Trouble ("On the Bubble")
During August, networks decide the fates of their summer replacement series. Several were struggling significantly:
*   **_Grand Hotel_ (ABC):** Produced by Eva Longoria, this soapy summer drama was airing its first season but pulling in very soft ratings. By August, TV line-up analysts had placed it on "cancellation watch" (it was officially canceled in October).
*   **_Reef Break_ (ABC):** This Poppy Montgomery-led crime drama was burning through its summer run in August to dismal ratings and poor reviews, making it a prime target for cancellation.
*   **_The InBetween_ (NBC):** NBC’s summer psychic-procedural was struggling to find an audience throughout August, consistently landing on "likely to be canceled" lists.

#### 3. Shock Cancellations (Where the Axe Fell in August)
Several shows that were on the bubble officially met their end in early-to-mid August:
*   **_The OA_ (Netflix):** In a shocking move that devastated its dedicated fanbase, Netflix officially canceled the sci-fi drama on **August 5, 2019**, after two seasons. The cancellation sparked massive online protests, a fan-funded billboard in Times Square, and even a fan hunger strike outside Netflix's headquarters.
*   **_Instinct_ (CBS):** The Alan Cumming-led police procedural was struggling with low ratings during its second season. On **August 16, 2019**, series creator Michael Rauch announced that CBS had officially canceled the show.
*   **_Krypton_ (Syfy):** On **August 16, 2019**, Syfy officially canceled this Superman prequel series after two seasons due to a steep drop-off in live viewership. At the same time, Syfy announced they were declining to move forward with a planned spin-off centering on the character *Lobo*.

</details>

## Code map

| File | What it is |
|---|---|
| [app.py](app.py) | The agents, their tools, and the FastAPI app. |
| [broadway_data.py](broadway_data.py) | The query functions the tools wrap: all the SQL and arithmetic. |
| [index.html](index.html), [context-panel.js](context-panel.js), [context_panel.py](context_panel.py) | The page, its chart, and the Backstage drawer. |
| [build_db.py](build_db.py), [schema.sql](schema.sql) | CSVs into the database, with the cross-source matching rules. |
| [ingest/](ingest/) | One script per source, plus the polite, cached HTTP client and the shared title folding. |
| [nyt.py](nyt.py), [wikipedia.py](wikipedia.py) | Live clients for the NYT and for Wikipedia show background. |
