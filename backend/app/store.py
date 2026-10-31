"""SQLite 存储层：FTS5 全文检索 + sqlite-vec 向量检索。

为什么是 SQLite
---------------
- 单文件、零外部依赖，魔搭免费容器直接跑
- FTS5 内置，bm25 排序可用
- sqlite-vec 提供 vec0 虚拟表，HNSW 之外最省心的选择

表结构
------
docs      文档元信息（标题、分类、更新时间）
chunks    切分后的文本块（含结构化字段快照）
chunks_fts  FTS5 全文索引（三元组分词用 unicode61 足够）
vec_chunks sqlite-vec 向量索引
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DIM = 512

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS docs (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL UNIQUE,
    pageid      INTEGER,
    categories  TEXT,          -- JSON array
    touched     TEXT,
    text_len    INTEGER,
    fields      TEXT           -- JSON object 结构化字段
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    title       TEXT NOT NULL,
    section     TEXT,
    text        TEXT NOT NULL,
    token_len   INTEGER,
    kind        TEXT DEFAULT 'text'   -- text | table | infobox
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    title,
    content='chunks',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL UNIQUE,   -- 英文名 / 文件名主键
    name_zh     TEXT,
    icon_file   TEXT,
    fields      TEXT,       -- JSON: growth/sellprice/regrow/...
    doc_title   TEXT
);

CREATE TABLE IF NOT EXISTS icon_vec (
    id          INTEGER PRIMARY KEY,
    item_id     INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    dim         INTEGER NOT NULL          -- 特征长度（多特征拼接）
);
"""


class Store:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._init_schema()

    # ---------------------------------------------------------------- 连接
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            self._load_vec(c)
            self._local.conn = c
        return c

    @staticmethod
    def _load_vec(conn: sqlite3.Connection) -> None:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            # 无 sqlite-vec 时退化为纯 BM25，服务仍可用
            pass

    @contextmanager
    def tx(self):
        conn = self.conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def has_vec(self) -> bool:
        try:
            self.conn.execute("SELECT vec_version()")
            return True
        except Exception:
            return False

    # ---------------------------------------------------------------- schema
    def _init_schema(self) -> None:
        with self._init_lock:
            with self.tx() as conn:
                conn.executescript(SCHEMA)
                if self.has_vec():
                    conn.execute(
                        f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                            USING vec0(chunk_id INTEGER PRIMARY KEY,
                                       embedding FLOAT[{DIM}])"""
                    )

    # ---------------------------------------------------------------- 写入
    def clear(self) -> None:
        with self.tx() as conn:
            for t in ("vec_chunks", "chunks_fts", "chunks", "docs", "icon_vec", "items"):
                try:
                    conn.execute(f"DELETE FROM {t}")
                except sqlite3.OperationalError:
                    pass

    def insert_docs(self, docs: Iterable[dict[str, Any]], chunks: Iterable[dict[str, Any]],
                    embeddings: Any | None = None, batch: int = 500) -> int:
        """写入文档与切块；embeddings 与 chunks 顺序一一对应。"""
        docs = list(docs)
        chunks = list(chunks)
        title_to_id: dict[str, int] = {}
        with self.tx() as conn:
            for d in docs:
                cur = conn.execute(
                    """INSERT OR REPLACE INTO docs
                       (title, pageid, categories, touched, text_len, fields)
                       VALUES (?,?,?,?,?,?)""",
                    (d["title"], d.get("pageid"), json.dumps(d.get("categories", []),
                                                             ensure_ascii=False),
                     d.get("touched"), d.get("text_len"), json.dumps(d.get("fields", {}),
                                                                     ensure_ascii=False)),
                )
                title_to_id[d["title"]] = cur.lastrowid  # type: ignore[assignment]

            chunk_ids: list[int] = []
            for c in chunks:
                doc_id = title_to_id.get(c["title"])
                if doc_id is None:
                    continue
                cur = conn.execute(
                    """INSERT INTO chunks (doc_id, ord, title, section, text, token_len, kind)
                       VALUES (?,?,?,?,?,?,?)""",
                    (doc_id, c.get("ord", 0), c["title"], c.get("section"),
                     c["text"], c.get("token_len", len(c["text"])), c.get("kind", "text")),
                )
                chunk_ids.append(cur.lastrowid)  # type: ignore[arg-type]

            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

            if embeddings is not None and self.has_vec() and len(chunk_ids):
                import numpy as np

                emb = np.asarray(embeddings, dtype=np.float32)
                rows = [
                    (int(cid), emb[i].tobytes())
                    for i, cid in enumerate(chunk_ids)
                    if i < len(emb)
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding) VALUES (?,?)", rows
                )
        return len(chunk_ids)

    def insert_items(self, items: list[dict[str, Any]]) -> int:
        with self.tx() as conn:
            for it in items:
                conn.execute(
                    """INSERT OR REPLACE INTO items (key, name_zh, icon_file, fields, doc_title)
                       VALUES (?,?,?,?,?)""",
                    (it["key"], it.get("name_zh"), it.get("icon_file"),
                     json.dumps(it.get("fields", {}), ensure_ascii=False),
                     it.get("doc_title")),
                )
        return len(items)

    # ---------------------------------------------------------------- 查询
    def search_bm25(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        sql = """
        SELECT c.id AS chunk_id, c.title, c.section, c.text, c.doc_id,
               bm25(chunks_fts, 1.0, 2.0) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """
        safe = _fts_query(query)
        if not safe:
            return []
        try:
            rows = self.conn.execute(sql, (safe, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "section": r["section"],
                "text": r["text"],
                "doc_id": r["doc_id"],
                "score": -float(r["score"] or 0.0),  # bm25 越小越好，取负便于统一排序
            }
            for r in rows
        ]

    def search_vec(self, embedding: Any, limit: int = 20) -> list[dict[str, Any]]:
        if not self.has_vec():
            return []
        import numpy as np

        q = np.asarray(embedding, dtype=np.float32).tobytes()
        sql = """
        SELECT v.chunk_id, v.distance, c.title, c.section, c.text, c.doc_id
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.chunk_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """
        try:
            rows = self.conn.execute(sql, (q, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "section": r["section"],
                "text": r["text"],
                "doc_id": r["doc_id"],
                "score": 1.0 - float(r["distance"]),  # cosine 距离 → 相似度
            }
            for r in rows
        ]

    def get_doc(self, title: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM docs WHERE title = ?", (title,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["categories"] = json.loads(d.get("categories") or "[]")
        d["fields"] = json.loads(d.get("fields") or "{}")
        return d

    def get_item(self, key_or_name: str) -> dict[str, Any] | None:
        r = self.conn.execute(
            "SELECT * FROM items WHERE key = ? OR name_zh = ? OR doc_title = ? LIMIT 1",
            (key_or_name, key_or_name, key_or_name),
        ).fetchone()
        if not r:
            return None
        d = dict(r)
        d["fields"] = json.loads(d.get("fields") or "{}")
        return d

    def list_items(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM items").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["fields"] = json.loads(d.get("fields") or "{}")
            out.append(d)
        return out

    def stats(self) -> dict[str, int]:
        def one(sql: str) -> int:
            try:
                return int(self.conn.execute(sql).fetchone()[0])
            except Exception:
                return 0

        return {
            "docs": one("SELECT COUNT(*) FROM docs"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "items": one("SELECT COUNT(*) FROM items"),
            "vec": one("SELECT COUNT(*) FROM vec_chunks") if self.has_vec() else 0,
            "vec_available": int(self.has_vec()),
        }


_PUNCT = set('.,!?;:()[]{}<>"\'`~@#$%^&*+=|\\/-_，。！？；：（）【】《》“”‘’、·…—')


def _fts_query(query: str) -> str:
    """把自然语言查询转为 FTS5 安全查询串（分词 + 短语）。"""
    if not query:
        return ""
    tokens = []
    buf = []
    for ch in query:
        if ch.isspace() or ch in _PUNCT:
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))

    # 中文 FTS5 无分词器，用前缀匹配 + 短语组合提升召回
    parts: list[str] = []
    for t in tokens[:8]:
        if not t:
            continue
        esc = t.replace('"', '""')
        if len(t) >= 2:
            parts.append(f'"{esc}"')
            # 双字切片，弥补无分词
            for i in range(len(t) - 1):
                parts.append(f'"{esc[i:i+2]}"*')
        else:
            parts.append(f'"{esc}"*')
    return " OR ".join(parts)
