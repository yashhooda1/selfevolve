"""Two-collection experience store, on a single SQLite file.

    insights      short reusable rules distilled from feedback
    trajectories  whole episodes: input, comments, engineer verdicts

Why SQLite and a brute-force scan instead of a vector database:

  * One file. `cp .selfevolve/experience.db backup.db` is the entire backup
    story, and `sqlite3` is the entire debugging story.
  * No server, no daemon, no telemetry, no ONNX runtime, no dependency that
    might phone home on import. In an airgapped build, every dependency is a
    surface you have to audit.
  * At the scale this actually operates — hundreds to low thousands of rules —
    a full scan of float32 vectors is faster than an index lookup, because there
    is no index to traverse and no network hop. 2,000 × 768-dim cosine scores in
    pure Python is single-digit milliseconds.

If this ever holds 100k+ rules, swap the ranking loop in `retrieve_insights` for
sqlite-vec. It is the only place that scans, and nothing above it changes.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from array import array
from typing import Iterable

from .config import Config, DEFAULT
from .embeddings import Embedder, get_embedder
from .models import Feedback, Insight, Item, Metrics, Scope, Trajectory

SCHEMA = """
CREATE TABLE IF NOT EXISTS insights (
    id             TEXT PRIMARY KEY,
    rule           TEXT NOT NULL,
    rationale      TEXT NOT NULL DEFAULT '',
    origin_action  TEXT NOT NULL DEFAULT 'accept',
    project        TEXT NOT NULL DEFAULT 'default',
    language       TEXT NOT NULL DEFAULT 'any',
    framework      TEXT NOT NULL DEFAULT 'any',
    team           TEXT NOT NULL DEFAULT 'any',
    confidence     REAL NOT NULL DEFAULT 0.5,
    support        INTEGER NOT NULL DEFAULT 1,
    contradictions INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    retired        INTEGER NOT NULL DEFAULT 0,
    fingerprint    TEXT NOT NULL,
    embedding      BLOB
);
CREATE INDEX IF NOT EXISTS idx_insights_fp      ON insights(fingerprint);
CREATE INDEX IF NOT EXISTS idx_insights_live    ON insights(retired, confidence);

CREATE TABLE IF NOT EXISTS trajectories (
    id            TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    project       TEXT NOT NULL DEFAULT 'default',
    language      TEXT NOT NULL DEFAULT 'any',
    framework     TEXT NOT NULL DEFAULT 'any',
    team          TEXT NOT NULL DEFAULT 'any',
    input_summary TEXT NOT NULL DEFAULT '',
    payload       TEXT NOT NULL,
    accepted      INTEGER NOT NULL DEFAULT 0,
    rejected      INTEGER NOT NULL DEFAULT 0,
    edited        INTEGER NOT NULL DEFAULT 0,
    n_items       INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    embedding     BLOB
);
CREATE INDEX IF NOT EXISTS idx_traj_created ON trajectories(created_at DESC);

-- Embedding cache. Embedding is the slowest step in the loop, and the same
-- rule text is re-embedded every time it's reinforced. Cache on the text hash
-- and that cost goes to zero.
CREATE TABLE IF NOT EXISTS embed_cache (
    key        TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""


def _pack(vec: Iterable[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    a = array("f")
    a.frombytes(blob)
    return list(a)


def _cosine(a: list[float], b: list[float]) -> float:
    """Vectors are stored normalized, so this is a dot product — but we don't
    assume it, because a swapped embedding backend might not normalize."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class ExperienceStore:
    def __init__(self, cfg: Config | None = None, embedder: Embedder | None = None):
        self.cfg = cfg or DEFAULT
        self.path = self.cfg.db_path()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        # Deliberately NOT WAL. WAL is faster under concurrency, but it leaves
        # recent commits in a sidecar -wal file, which quietly breaks the promise
        # this store is built on: that `cp experience.db` is a complete backup.
        # A test caught exactly that. At one writer and millisecond writes, the
        # default journal costs nothing and keeps the file self-contained.
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.commit()
        self.embedder = embedder or get_embedder(self.cfg, cache=self._cache_handle())

    def _cache_handle(self):
        """Hand the embedder a cache backed by this same file — one artifact on
        disk, not a model cache in some other directory."""
        conn = self.conn

        class _Cache:
            def get(self, key: str, model: str) -> list[float] | None:
                row = conn.execute(
                    "SELECT embedding FROM embed_cache WHERE key=? AND model=?", (key, model)
                ).fetchone()
                return _unpack(row["embedding"]) if row else None

            def put(self, key: str, model: str, vec: list[float]) -> None:
                conn.execute(
                    "INSERT OR REPLACE INTO embed_cache(key, model, dim, embedding, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (key, model, len(vec), _pack(vec), time.time()),
                )
                conn.commit()

        return _Cache()

    # ------------------------------------------------------------------ write

    def add_insight(self, insight: Insight) -> Insight:
        """Add a rule, merging it into an existing one if we've learned it before.

        Merge policy:
          * same fingerprint (normalized rule + scope), same direction
            -> support += 1, confidence rises toward 1.0
          * same fingerprint, opposite direction (a rule born from `accept` now
            contradicted by a `reject`) -> contradictions += 1, confidence halves,
            and the rule retires once it drops below the floor.
        """
        existing = self._find_by_fingerprint(insight.fingerprint())
        if existing is None:
            vec = self.embedder.embed_one(self._insight_text(insight))
            self.conn.execute(
                "INSERT INTO insights(id, rule, rationale, origin_action, project, language,"
                " framework, team, confidence, support, contradictions, created_at, updated_at,"
                " retired, fingerprint, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    insight.id, insight.rule, insight.rationale, insight.origin_action,
                    insight.scope.project, insight.scope.language, insight.scope.framework,
                    insight.scope.team, insight.confidence, insight.support,
                    insight.contradictions, insight.created_at, insight.updated_at,
                    int(insight.retired), insight.fingerprint(), _pack(vec),
                ),
            )
            self.conn.commit()
            return insight

        agrees = existing.origin_action == insight.origin_action or insight.origin_action == "accept"
        if agrees:
            existing.support += 1
            existing.confidence = min(1.0, existing.confidence + (1.0 - existing.confidence) * 0.35)
        else:
            existing.contradictions += 1
            existing.confidence = max(0.0, existing.confidence * 0.5)
        if insight.rationale and insight.rationale not in existing.rationale:
            existing.rationale = (existing.rationale + " | " + insight.rationale).strip(" |")[:1200]
        existing.updated_at = time.time()
        if existing.confidence < self.cfg.retire_below:
            existing.retired = True
        return self.update_insight(existing)

    def add_trajectory(self, traj: Trajectory) -> Trajectory:
        payload = json.dumps(
            {
                "items": [i.model_dump() for i in traj.items],
                "feedback": [f.model_dump() for f in traj.feedback],
                "insight_ids": traj.insight_ids,
            }
        )
        counts = traj.counts()
        vec = self.embedder.embed_one(traj.input_text or traj.input_summary)
        self.conn.execute(
            "INSERT OR REPLACE INTO trajectories(id, task, project, language, framework, team,"
            " input_summary, payload, accepted, rejected, edited, n_items, created_at, embedding)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                traj.id, traj.task, traj.scope.project, traj.scope.language,
                traj.scope.framework, traj.scope.team, traj.input_summary, payload,
                counts.get("accept", 0), counts.get("reject", 0), counts.get("edit", 0),
                len(traj.items), traj.created_at, _pack(vec),
            ),
        )
        self.conn.commit()
        return traj

    def update_insight(self, insight: Insight) -> Insight:
        insight.updated_at = time.time()
        vec = self.embedder.embed_one(self._insight_text(insight))
        self.conn.execute(
            "UPDATE insights SET rule=?, rationale=?, origin_action=?, project=?, language=?,"
            " framework=?, team=?, confidence=?, support=?, contradictions=?, updated_at=?,"
            " retired=?, fingerprint=?, embedding=? WHERE id=?",
            (
                insight.rule, insight.rationale, insight.origin_action, insight.scope.project,
                insight.scope.language, insight.scope.framework, insight.scope.team,
                insight.confidence, insight.support, insight.contradictions, insight.updated_at,
                int(insight.retired), insight.fingerprint(), _pack(vec), insight.id,
            ),
        )
        self.conn.commit()
        return insight

    def retire_insight(self, insight_id: str) -> None:
        ins = self.get_insight(insight_id)
        if ins:
            ins.retired = True
            self.update_insight(ins)

    def delete_insight(self, insight_id: str) -> None:
        self.conn.execute("DELETE FROM insights WHERE id=?", (insight_id,))
        self.conn.commit()

    # ------------------------------------------------------------------- read

    def retrieve_insights(
        self, query: str, scope: Scope | None = None, k: int | None = None
    ) -> list[Insight]:
        k = k or self.cfg.top_k_insights
        rows = self.conn.execute(
            "SELECT * FROM insights WHERE retired=0 AND confidence>=?", (self.cfg.min_confidence,)
        ).fetchall()
        if not rows:
            return []
        qvec = self.embedder.embed_one(query)
        scored: list[tuple[float, Insight]] = []
        for row in rows:
            ins = _insight_from_row(row)
            if scope is not None and not _scope_applies(ins.scope, scope):
                continue
            scored.append((_cosine(qvec, _unpack(row["embedding"])), ins))
        # Similarity picks the candidates; confidence and support break ties, so
        # a well-supported rule outranks a marginally closer one-off.
        scored.sort(key=lambda t: (-t[0], -t[1].confidence, -t[1].support))
        return [ins for _, ins in scored[:k]]

    def retrieve_trajectories(
        self, query: str, scope: Scope | None = None, k: int | None = None
    ) -> list[Trajectory]:
        k = k or self.cfg.top_k_trajectories
        rows = self.conn.execute("SELECT * FROM trajectories").fetchall()
        if not rows:
            return []
        qvec = self.embedder.embed_one(query)
        scored: list[tuple[float, Trajectory]] = []
        for row in rows:
            traj = _traj_from_row(row)
            if scope is not None and not _scope_applies(traj.scope, scope):
                continue
            scored.append((_cosine(qvec, _unpack(row["embedding"])), traj))
        scored.sort(key=lambda t: -t[0])
        return [t for _, t in scored[:k]]

    def get_insight(self, insight_id: str) -> Insight | None:
        row = self.conn.execute("SELECT * FROM insights WHERE id=?", (insight_id,)).fetchone()
        return _insight_from_row(row) if row else None

    def all_insights(self, include_retired: bool = False) -> list[Insight]:
        sql = "SELECT * FROM insights"
        if not include_retired:
            sql += " WHERE retired=0"
        sql += " ORDER BY confidence DESC, support DESC"
        return [_insight_from_row(r) for r in self.conn.execute(sql).fetchall()]

    def all_trajectories(self) -> list[Trajectory]:
        rows = self.conn.execute("SELECT * FROM trajectories ORDER BY created_at DESC").fetchall()
        return [_traj_from_row(r) for r in rows]

    def metrics(self, scope: Scope | None = None) -> Metrics:
        m = Metrics()
        for row in self.conn.execute("SELECT * FROM trajectories").fetchall():
            traj_scope = Scope(
                project=row["project"], language=row["language"],
                framework=row["framework"], team=row["team"],
            )
            if scope is not None and not _scope_applies(traj_scope, scope):
                continue
            m.reviews += 1
            m.items += row["n_items"]
            m.accepted += row["accepted"]
            m.rejected += row["rejected"]
            m.edited += row["edited"]
        m.insights = len(
            [i for i in self.all_insights() if scope is None or _scope_applies(i.scope, scope)]
        )
        return m

    def reset(self) -> None:
        self.conn.executescript(
            "DELETE FROM insights; DELETE FROM trajectories; DELETE FROM embed_cache;"
        )
        self.conn.commit()

    def backup(self, dest) -> str:
        """Consistent copy while the store is open, via SQLite's online backup.

        `cp` is fine when nothing is writing. This is correct even when Streamlit
        has the database open in another process.
        """
        dest = str(dest)
        with sqlite3.connect(dest) as out:
            self.conn.backup(out)
        return dest

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # --------------------------------------------------------------- internal

    def _find_by_fingerprint(self, fp: str) -> Insight | None:
        row = self.conn.execute("SELECT * FROM insights WHERE fingerprint=?", (fp,)).fetchone()
        return _insight_from_row(row) if row else None

    @staticmethod
    def _insight_text(insight: Insight) -> str:
        return f"{insight.rule}\n{insight.rationale}"


def _insight_from_row(row: sqlite3.Row) -> Insight:
    return Insight(
        id=row["id"],
        rule=row["rule"],
        rationale=row["rationale"],
        origin_action=row["origin_action"],
        scope=Scope(
            project=row["project"], language=row["language"],
            framework=row["framework"], team=row["team"],
        ),
        confidence=row["confidence"],
        support=row["support"],
        contradictions=row["contradictions"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        retired=bool(row["retired"]),
    )


def _traj_from_row(row: sqlite3.Row) -> Trajectory:
    payload = json.loads(row["payload"])
    return Trajectory(
        id=row["id"],
        task=row["task"],
        scope=Scope(
            project=row["project"], language=row["language"],
            framework=row["framework"], team=row["team"],
        ),
        input_summary=row["input_summary"],
        items=[Item(**i) for i in payload.get("items", [])],
        feedback=[Feedback(**f) for f in payload.get("feedback", [])],
        insight_ids=payload.get("insight_ids", []),
        created_at=row["created_at"],
    )


def _scope_applies(rule_scope: Scope, query_scope: Scope) -> bool:
    """A rule applies if each of its fields is either a wildcard or a match.

    `any` is the wildcard, so a general rule ("don't flag missing docstrings")
    is retrieved everywhere while a Spark-specific one stays with Spark.
    `default` — the project name you get when you never set one — is treated as
    a wildcard too, so the zero-config path still works before anyone thinks
    about scoping.
    """
    pairs = [
        (rule_scope.project, query_scope.project),
        (rule_scope.language, query_scope.language),
        (rule_scope.framework, query_scope.framework),
        (rule_scope.team, query_scope.team),
    ]
    return all(r in ("any", "default", q) or q == "any" for r, q in pairs)
