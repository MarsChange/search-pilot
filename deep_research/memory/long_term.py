from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    entry_id: str
    session_id: str
    claim: str
    source: str
    url: str
    confidence: float
    agent_id: str
    timestamp: float
    evidence_type: str
    topic: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "claim": self.claim,
            "source": self.source,
            "url": self.url,
            "confidence": float(self.confidence),
            "agent_id": self.agent_id,
            "timestamp": float(self.timestamp),
            "evidence_type": self.evidence_type,
            "topic": self.topic,
            "embedding_json": json.dumps(self.embedding),
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MemoryEntry":
        return cls(
            entry_id=row["entry_id"],
            session_id=row["session_id"],
            claim=row["claim"],
            source=row["source"],
            url=row["url"],
            confidence=float(row["confidence"]),
            agent_id=row["agent_id"],
            timestamp=float(row["timestamp"]),
            evidence_type=row["evidence_type"],
            topic=row["topic"],
            embedding=json.loads(row["embedding_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


@dataclass
class ConflictRecord:
    conflict_id: str
    entry_id_1: str
    entry_id_2: str
    claim_1: str
    claim_2: str
    similarity: float
    status: str = "open"
    resolution: str | None = None
    created_at: float = field(default_factory=time.time)


class LongTermMemory:
    def __init__(self, db_path: str = "data/deep_research_memory.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entries (
                        entry_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        claim TEXT NOT NULL,
                        source TEXT NOT NULL,
                        url TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        agent_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        evidence_type TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_topic ON entries(topic)")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conflicts (
                        conflict_id TEXT PRIMARY KEY,
                        entry_id_1 TEXT NOT NULL,
                        entry_id_2 TEXT NOT NULL,
                        claim_1 TEXT NOT NULL,
                        claim_2 TEXT NOT NULL,
                        similarity REAL NOT NULL,
                        status TEXT NOT NULL,
                        resolution TEXT,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status)")
                conn.commit()
            finally:
                conn.close()

    def insert_entry(self, entry: MemoryEntry) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entries (
                        entry_id, session_id, claim, source, url, confidence, agent_id,
                        timestamp, evidence_type, topic, embedding_json, metadata_json
                    ) VALUES (
                        :entry_id, :session_id, :claim, :source, :url, :confidence,
                        :agent_id, :timestamp, :evidence_type, :topic,
                        :embedding_json, :metadata_json
                    )
                    """,
                    entry.to_row(),
                )
                conn.commit()
            finally:
                conn.close()

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM entries WHERE entry_id = ?", (entry_id,)).fetchone()
                return MemoryEntry.from_row(row) if row else None
            finally:
                conn.close()

    def get_all_entries(self, session_id: str | None = None) -> list[MemoryEntry]:
        with self._lock:
            conn = self._connect()
            try:
                if session_id is None:
                    rows = conn.execute("SELECT * FROM entries ORDER BY timestamp DESC").fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM entries WHERE session_id = ? ORDER BY timestamp DESC",
                        (session_id,),
                    ).fetchall()
                return [MemoryEntry.from_row(row) for row in rows]
            finally:
                conn.close()

    def count_entries(self, session_id: str | None = None) -> int:
        with self._lock:
            conn = self._connect()
            try:
                if session_id is None:
                    return int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
                return int(
                    conn.execute("SELECT COUNT(*) FROM entries WHERE session_id = ?", (session_id,)).fetchone()[0]
                )
            finally:
                conn.close()

    def delete_entry(self, entry_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM conflicts WHERE entry_id_1 = ? OR entry_id_2 = ?", (entry_id, entry_id))
                cur = conn.execute("DELETE FROM entries WHERE entry_id = ?", (entry_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def insert_conflict(self, record: ConflictRecord) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO conflicts (
                        conflict_id, entry_id_1, entry_id_2, claim_1, claim_2,
                        similarity, status, resolution, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.conflict_id,
                        record.entry_id_1,
                        record.entry_id_2,
                        record.claim_1,
                        record.claim_2,
                        float(record.similarity),
                        record.status,
                        record.resolution,
                        float(record.created_at),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_conflicts(self, status: str | None = None) -> list[ConflictRecord]:
        with self._lock:
            conn = self._connect()
            try:
                if status is None:
                    rows = conn.execute("SELECT * FROM conflicts ORDER BY created_at DESC").fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM conflicts WHERE status = ? ORDER BY created_at DESC",
                        (status,),
                    ).fetchall()
                return [
                    ConflictRecord(
                        conflict_id=row["conflict_id"],
                        entry_id_1=row["entry_id_1"],
                        entry_id_2=row["entry_id_2"],
                        claim_1=row["claim_1"],
                        claim_2=row["claim_2"],
                        similarity=float(row["similarity"]),
                        status=row["status"],
                        resolution=row["resolution"],
                        created_at=float(row["created_at"]),
                    )
                    for row in rows
                ]
            finally:
                conn.close()

    def update_conflict_resolution(
        self,
        conflict_id: str,
        status: str,
        resolution: str | None = None,
    ) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "UPDATE conflicts SET status = ?, resolution = ? WHERE conflict_id = ?",
                    (status, resolution, conflict_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()
