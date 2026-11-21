"""Galeria local em SQLite, com exclusão completa por pessoa (SPEC-006)."""
from __future__ import annotations

import shutil
import sqlite3
import threading
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .errors import DuplicatePersonError, PersonNotFoundError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,
    consent_confirmed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id);
"""


def name_key(name: str) -> str:
    """Normaliza o nome para detectar duplicatas: sem acento, caixa ou espaços extras."""
    decomposed = unicodedata.normalize("NFKD", name)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(without_marks.casefold().split())


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    created_at: str
    num_embeddings: int


class FaceStore:
    def __init__(self, db_path: Path, photos_dir: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.photos_dir = photos_dir
        self._lock = threading.RLock()
        self._version = 0
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA secure_delete=ON")
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.executescript(_SCHEMA)

    @property
    def version(self) -> int:
        """Muda a cada alteração, permitindo invalidar caches da galeria."""
        with self._lock:
            return self._version

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_person(
        self, name: str, embeddings: np.ndarray, model_id: str, photos: list[bytes] | None = None
    ) -> Person:
        """Grava pessoa, embeddings e fotos atomicamente."""
        key = name_key(name)
        person_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        person_photos = self.photos_dir / person_id
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if self._conn.execute("SELECT 1 FROM persons WHERE name_key=?", (key,)).fetchone():
                    raise DuplicatePersonError()
                self._conn.execute(
                    "INSERT INTO persons(id,name,name_key,consent_confirmed_at,created_at) VALUES(?,?,?,?,?)",
                    (person_id, " ".join(name.split()), key, now, now),
                )
                self._conn.executemany(
                    "INSERT INTO embeddings(person_id,model_id,dim,vector) VALUES(?,?,?,?)",
                    [(person_id, model_id, len(v), np.asarray(v, dtype=np.float32).tobytes()) for v in embeddings],
                )
                if photos:
                    person_photos.mkdir(parents=True)
                    for index, data in enumerate(photos):
                        (person_photos / f"{index:03d}.img").write_bytes(data)
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                shutil.rmtree(person_photos, ignore_errors=True)
                raise
            self._version += 1
        return Person(person_id, " ".join(name.split()), now, len(embeddings))

    def list_persons(self) -> list[Person]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT p.id, p.name, p.created_at, COUNT(e.id)
                   FROM persons p LEFT JOIN embeddings e ON e.person_id = p.id
                   GROUP BY p.id ORDER BY p.name_key"""
            ).fetchall()
        return [Person(*row) for row in rows]

    def name_of(self, person_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT name FROM persons WHERE id=?", (person_id,)).fetchone()
        return row[0] if row else None

    def load_gallery(self, model_id: str) -> tuple[np.ndarray, list[str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, dim, vector FROM embeddings WHERE model_id=? ORDER BY id", (model_id,)
            ).fetchall()
        if not rows:
            return np.empty((0, 512), dtype=np.float32), []
        vectors = np.stack([np.frombuffer(vector, dtype=np.float32, count=dim) for _, dim, vector in rows])
        return vectors, [row[0] for row in rows]

    def count_other_models(self, model_id: str) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_id<>?", (model_id,)
            ).fetchone()[0]

    def delete_person(self, person_id: str) -> None:
        """Remove cadastro, embeddings e fotos, sem deixar resíduos no arquivo do banco."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
            if cursor.rowcount == 0:
                raise PersonNotFoundError()
            shutil.rmtree(self.photos_dir / person_id, ignore_errors=True)
            self._conn.execute("VACUUM")
            self._version += 1
