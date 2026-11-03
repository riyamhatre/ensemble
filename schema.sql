-- Broadway research database. Built from data/csv/*.csv by build_db.py; never edited by hand.
--
-- Conventions
--   Dates are ISO text, 'YYYY-MM-DD'. SQLite has no date type; ISO text sorts and compares correctly.
--   Booleans are INTEGER 0/1. Money is REAL, in dollars.
--   NULL means "unknown", never zero. potential_gross is missing in some years, for example.
--   Every fact table carries a source_url so the Writer can cite it.
--   Foreign keys are only enforced if the connection runs PRAGMA foreign_keys = ON (build_db.py does).

-- ---------------------------------------------------------------------------
-- Identity: one row per production, plus every name each source calls it
-- ---------------------------------------------------------------------------

CREATE TABLE productions (
  production_id     TEXT PRIMARY KEY,   -- stable slug, e.g. 'wicked-2003', 'chicago-1996'
  title             TEXT NOT NULL,      -- canonical display title
  type              TEXT NOT NULL CHECK (type IN ('original', 'revival')),
  category          TEXT CHECK (category IN ('musical', 'play', 'special')),
  theater           TEXT,
  first_preview_date TEXT,
  opening_date      TEXT,
  closing_date      TEXT,               -- NULL = still running as of the snapshot
  total_performances INTEGER,           -- from IBDB; NULL while running
  closed_early      INTEGER CHECK (closed_early IN (0, 1)),
                                        -- derived in build_db.py from run length; NULL = running or unknown
  ibdb_id           TEXT,
  ibdb_url          TEXT
);

-- Several rows per production: each source's spelling of the title, ideally with its own ID.
-- One alias can map to several productions ('Chicago' = the 1975 original and the 1996 revival);
-- dates in the source row pick between them.
CREATE TABLE aliases (
  alias         TEXT NOT NULL COLLATE NOCASE,
  source        TEXT NOT NULL,          -- 'grosses' | 'ibdb' | 'tonys' | 'musicbrainz' | 'nyt' | 'manual'
  source_id     TEXT,                   -- that source's own ID, when it has one
  production_id TEXT NOT NULL REFERENCES productions(production_id),
  match_method  TEXT CHECK (match_method IN ('exact', 'fuzzy', 'manual')),
  match_score   REAL,                   -- fuzzy score, so you can review the weakest matches
  PRIMARY KEY (source, alias, production_id)
);
CREATE INDEX idx_aliases_alias ON aliases(alias);

-- ---------------------------------------------------------------------------
-- Weekly box office (Broadway League; Broadway only: tours are not in here)
-- ---------------------------------------------------------------------------

CREATE TABLE grosses (
  production_id    TEXT NOT NULL REFERENCES productions(production_id),
  week_ending      TEXT NOT NULL,       -- Sunday
  gross            REAL,
  potential_gross  REAL,                -- NULL in some years
  attendance       INTEGER,             -- seats sold
  seats_available  INTEGER,
  capacity_pct     REAL CHECK (capacity_pct >= 0),
                                        -- 0-100 scale (Tidy Tuesday's pct_capacity is 0-1: multiply by 100
                                        -- on ingest). Can exceed 100 with standing room.
  avg_ticket       REAL,
  top_ticket       REAL,
  performances     INTEGER,
  previews         INTEGER,
  source_url       TEXT,
  PRIMARY KEY (production_id, week_ending)
);
-- trouble_signals compares across shows in the same week, so the date alone needs an index too
CREATE INDEX idx_grosses_week ON grosses(week_ending);

-- ---------------------------------------------------------------------------
-- Cast (IBDB)
-- ---------------------------------------------------------------------------

CREATE TABLE cast_changes (
  cast_change_id   INTEGER PRIMARY KEY,
  production_id    TEXT NOT NULL REFERENCES productions(production_id),
  person           TEXT NOT NULL,
  person_ibdb_id   TEXT,
  role             TEXT,
  start_date       TEXT,                -- NULL for opening-night cast with no date given
  end_date         TEXT,                -- NULL = stayed to the end (or still there)
  is_replacement   INTEGER NOT NULL DEFAULT 0 CHECK (is_replacement IN (0, 1)),
  is_star          INTEGER CHECK (is_star IN (0, 1)),
                                        -- hand-curated "celebrity casting" flag; NULL = not reviewed
  source_url       TEXT
);
CREATE INDEX idx_cast_production ON cast_changes(production_id, start_date);

-- ---------------------------------------------------------------------------
-- Songs (IBDB) and cast recordings (MusicBrainz)
-- ---------------------------------------------------------------------------

CREATE TABLE songs (
  song_id        INTEGER PRIMARY KEY,
  production_id  TEXT NOT NULL REFERENCES productions(production_id),
  act            INTEGER,
  position       INTEGER,
  title          TEXT NOT NULL,         -- as printed
  title_norm     TEXT NOT NULL,         -- lowercased, punctuation and '(Reprise)' stripped: compare_songs joins on this
  source_url     TEXT
);
CREATE INDEX idx_songs_production ON songs(production_id);

CREATE TABLE recordings (
  recording_id   TEXT PRIMARY KEY,      -- your own ID; the MusicBrainz ID is in mbid
  production_id  TEXT NOT NULL REFERENCES productions(production_id),
  title          TEXT,
  year           INTEGER,
  mbid           TEXT,
  source_url     TEXT
);
CREATE INDEX idx_recordings_production ON recordings(production_id);

CREATE TABLE tracks (
  recording_id   TEXT NOT NULL REFERENCES recordings(recording_id),
  disc           INTEGER NOT NULL DEFAULT 1,
  position       INTEGER NOT NULL,
  title          TEXT NOT NULL,
  PRIMARY KEY (recording_id, disc, position)
);

-- ---------------------------------------------------------------------------
-- Tony Awards
-- ---------------------------------------------------------------------------

-- The ceremony date is what grosses_around_date needs for "the Tony bump".
CREATE TABLE tony_ceremonies (
  year             INTEGER PRIMARY KEY,
  ceremony_number  INTEGER,
  ceremony_date    TEXT,
  source_url       TEXT
);

CREATE TABLE awards (
  award_id       INTEGER PRIMARY KEY,
  year           INTEGER NOT NULL REFERENCES tony_ceremonies(year),
  category       TEXT NOT NULL,
  nominee        TEXT NOT NULL,         -- a person or a show, as listed
  production_id  TEXT REFERENCES productions(production_id),
                                        -- the production the nominee belongs to; NULL if it never matched
  won            INTEGER NOT NULL CHECK (won IN (0, 1)),
  source_url     TEXT
);
CREATE INDEX idx_awards_production ON awards(production_id);
CREATE INDEX idx_awards_year ON awards(year);

-- ---------------------------------------------------------------------------
-- Press (NYT Article Search: metadata only, no full text)
-- ---------------------------------------------------------------------------

CREATE TABLE articles (
  production_id  TEXT NOT NULL REFERENCES productions(production_id),
  url            TEXT NOT NULL,
  headline       TEXT,
  abstract       TEXT,
  pub_date       TEXT,
  is_review      INTEGER CHECK (is_review IN (0, 1)),
  PRIMARY KEY (production_id, url)
);
CREATE INDEX idx_articles_production ON articles(production_id, pub_date);

-- ---------------------------------------------------------------------------
-- Provenance: what each source covers, so agents can say "my data ends at week X"
-- ---------------------------------------------------------------------------

CREATE TABLE data_sources (
  source        TEXT PRIMARY KEY,       -- 'grosses' | 'ibdb' | 'tonys' | 'musicbrainz' | 'nyt'
  description   TEXT,
  covers_from   TEXT,
  covers_to     TEXT,
  fetched_at    TEXT,                   -- when the snapshot was taken
  notes         TEXT                    -- known gaps, e.g. 'no potential_gross before 2000'
);