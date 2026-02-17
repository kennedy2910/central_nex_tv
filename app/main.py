# -*- coding: utf-8 -*-

import os
import re
import sqlite3
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from urllib.request import Request as UrlRequest, urlopen


def fetch_html(url: str) -> str:
    # FIX: use UrlRequest (urllib) and not FastAPI/Starlette Request
    req = UrlRequest(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def youtube_duration_seconds(url: str) -> int | None:
    html = fetch_html(url)

    match = re.search(r'"lengthSeconds":"(\d+)"', html)
    if match:
        return int(match.group(1))

    match = re.search(r'itemprop="duration"\s+content="(PT[^"]+)"', html)
    if match:
        return parse_iso8601(match.group(1))

    return None


def parse_iso8601(val: str) -> int:
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", val)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s


def parse_datetime_utc(value: str | None) -> datetime | None:
    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_linear_now(schedule_start: str | None, items: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    start = parse_datetime_utc(schedule_start)
    if not start or not items:
        return None

    normalized = []
    total = 0
    for row in items:
        duration = int(row.get("duration") or 0)
        if duration <= 0:
            continue
        normalized.append({
            "type": row.get("type"),
            "url": row.get("url"),
            "duration": duration,
        })
        total += duration

    if total <= 0 or not normalized:
        return None

    now = datetime.now(timezone.utc)
    elapsed = int((now - start).total_seconds())
    if elapsed < 0:
        cycle_pos = 0
    else:
        cycle_pos = elapsed % total

    cursor = 0
    current = normalized[0]
    for row in normalized:
        duration = row["duration"]
        if cycle_pos < cursor + duration:
            current = row
            break
        cursor += duration

    payload = {
        "type": current["type"],
        "offset": max(0, cycle_pos - cursor),
        "duration": current["duration"],
    }
    if current.get("url"):
        payload["url"] = current["url"]
    return payload


def normalize_channel_kind(raw_kind: str | None, source_url: str) -> str:
    kind_raw = (raw_kind or "auto").strip().lower()
    if kind_raw not in ("auto", "hls", "youtube", "youtube_linear"):
        kind_raw = "auto"
    if kind_raw == "auto":
        return "youtube" if is_youtube_url(source_url) else "hls"
    return kind_raw


# Optional dependency for YouTube metadata (duration/title)
try:
    import yt_dlp  # type: ignore
except Exception:  # pragma: no cover
    yt_dlp = None


DATA_DIR = os.getenv("DATA_DIR", "/data")
DB_PATH = os.path.join(DATA_DIR, "central_nex.db")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title=os.getenv("CENTRAL_TITLE", "Central-Nex"))

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


def youtube_to_embed(url: str | None):
    if not url:
        return None

    yt_patterns = [
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
        r"youtube\.com/live/([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"youtube\.com/embed/([a-zA-Z0-9_-]+)"
    ]

    for pat in yt_patterns:
        match = re.search(pat, url)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/embed/{video_id}"

    return None


# -------------------------
# Helpers (URL typing)
# -------------------------

YOUTUBE_HOST_SNIPPETS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
)


def is_youtube_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(h in u for h in YOUTUBE_HOST_SNIPPETS)


def to_youtube_embed(url: str) -> str:
    """Best-effort conversion to an embeddable URL.
    If conversion fails, return the original URL.
    """
    if not url:
        return url
    if "youtube.com/embed/" in url or "youtube-nocookie.com/embed/" in url:
        return url

    # youtu.be/<id>
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        vid = m.group(1)
        return f"https://www.youtube.com/embed/{vid}"

    # youtube.com/watch?v=<id>
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    if m:
        vid = m.group(1)
        return f"https://www.youtube.com/embed/{vid}"

    # youtube.com/live/<id>
    m = re.search(r"youtube\.com/live/([A-Za-z0-9_-]{6,})", url)
    if m:
        vid = m.group(1)
        return f"https://www.youtube.com/embed/{vid}"

    return url


def youtube_metadata(url: str) -> dict:
    """Fetch best-effort YouTube metadata (title/duration) using yt-dlp.

    This is used to auto-fill durations for youtube_linear items when the admin
    leaves duration empty.
    """
    if yt_dlp is None:
        raise RuntimeError("yt_dlp is not installed. Add 'yt-dlp' to requirements.txt")
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[attr-defined]
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "webpage_url": info.get("webpage_url") or url,
    }


def db() -> sqlite3.Connection:
    # timeout avoids immediate 'database is locked' on concurrent access
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # Ensure FK constraints + cascades work as expected in SQLite.
    conn.execute("PRAGMA foreign_keys = ON")
    # Improve read/write concurrency in SQLite
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    # -------------------------
    # Providers
    # -------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS providers (
        provider_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT DEFAULT ''
    )
    """)

    # -------------------------
    # Edges
    # -------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        edge_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        api_key TEXT NOT NULL UNIQUE,
        hls_base_url TEXT NOT NULL,
        grade_id INTEGER,
        country TEXT DEFAULT '',
        state TEXT DEFAULT '',
        city TEXT DEFAULT '',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS edge_providers (
        edge_id TEXT NOT NULL,
        provider_id TEXT NOT NULL,
        PRIMARY KEY(edge_id, provider_id),
        FOREIGN KEY(edge_id) REFERENCES edges(edge_id) ON DELETE CASCADE,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE
    )
    """)

    # -------------------------
    # Distribution Grades (lista de canais)
    # -------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS channel_grades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT DEFAULT '',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS channel_grade_channels (
        grade_id INTEGER NOT NULL,
        channel_id INTEGER NOT NULL,
        sort_order INTEGER NOT NULL DEFAULT 100,
        PRIMARY KEY(grade_id, channel_id),
        FOREIGN KEY(grade_id) REFERENCES channel_grades(id) ON DELETE CASCADE,
        FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE
    )
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_grade_channels_sort
    ON channel_grade_channels(grade_id, sort_order, channel_id)
    """)

    edge_cols = [r["name"] for r in cur.execute("PRAGMA table_info(edges)").fetchall()]
    if "grade_id" not in edge_cols:
        cur.execute("ALTER TABLE edges ADD COLUMN grade_id INTEGER")
    if "country" not in edge_cols:
        cur.execute("ALTER TABLE edges ADD COLUMN country TEXT DEFAULT ''")
    if "state" not in edge_cols:
        cur.execute("ALTER TABLE edges ADD COLUMN state TEXT DEFAULT ''")
    if "city" not in edge_cols:
        cur.execute("ALTER TABLE edges ADD COLUMN city TEXT DEFAULT ''")

    # -------------------------
    # Channels (v2 schema)
    # -------------------------
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='channels'"
    )
    has_channels = cur.fetchone() is not None

    if has_channels:
        cols = [
            r["name"]
            for r in cur.execute("PRAGMA table_info(channels)").fetchall()
        ]
        is_v2 = "channel_number" in cols and "id" in cols

        if not is_v2:
            # Rename old table
            cur.execute("ALTER TABLE channels RENAME TO channels_old")

            # Create new schema
            cur.execute("""
            CREATE TABLE channels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_number TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                source_url TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'auto',
                schedule_start TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                sort_order INTEGER NOT NULL DEFAULT 100,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE,
                UNIQUE(provider_id, channel_number)
            )
            """)

            # Migrate data
            cur.execute("""
            INSERT INTO channels (
                channel_number,
                name,
                category,
                provider_id,
                source_url,
                is_active,
                sort_order,
                created_at
            )
            SELECT
                c.channel_id,
                c.name,
                COALESCE(p.name, 'Geral') AS category,
                c.provider_id,
                c.source_url,
                c.is_active,
                c.sort_order,
                c.created_at
            FROM channels_old c
            LEFT JOIN providers p ON p.provider_id = c.provider_id
            """)

            cur.execute("DROP TABLE channels_old")

    else:
        # Fresh install
        cur.execute("""
        CREATE TABLE channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_number TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            source_url TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'auto',
            schedule_start TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            sort_order INTEGER NOT NULL DEFAULT 100,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE,
            UNIQUE(provider_id, channel_number)
        )
        """)

    # Ensure new columns for scheduling exist (safe for upgrades)
    cols = [
        r["name"]
        for r in cur.execute("PRAGMA table_info(channels)").fetchall()
    ]
    if "kind" not in cols:
        cur.execute("ALTER TABLE channels ADD COLUMN kind TEXT NOT NULL DEFAULT 'auto'")
    if "schedule_start" not in cols:
        cur.execute("ALTER TABLE channels ADD COLUMN schedule_start TEXT")

    # -------------------------
    # Legacy Channel Items (backward compatibility)
    # -------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS channel_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        type TEXT NOT NULL,
        url TEXT,
        duration INTEGER NOT NULL,
        FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE
    )
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_channel_items_channel_pos
    ON channel_items(channel_id, position)
    """)

    # -------------------------
    # Linear playlists per channel
    # -------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS playlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE,
        UNIQUE(channel_id, name)
    )
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_playlists_channel
    ON playlists(channel_id, is_active)
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS playlist_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        playlist_id INTEGER NOT NULL,
        position INTEGER NOT NULL,
        type TEXT NOT NULL,
        url TEXT,
        duration INTEGER NOT NULL,
        FOREIGN KEY(playlist_id) REFERENCES playlists(id) ON DELETE CASCADE
    )
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_playlist_items_playlist_pos
    ON playlist_items(playlist_id, position)
    """)

    # -------------------------
    # Edge linear programming (advanced)
    # -------------------------
    cur.execute("""
    CREATE TABLE IF NOT EXISTS edge_programming (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        edge_id TEXT NOT NULL,
        channel_id INTEGER NOT NULL,
        playlist_id INTEGER NOT NULL,
        schedule_start TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY(edge_id) REFERENCES edges(edge_id) ON DELETE CASCADE,
        FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE,
        FOREIGN KEY(playlist_id) REFERENCES playlists(id) ON DELETE CASCADE
    )
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_edge_programming_lookup
    ON edge_programming(edge_id, channel_id, is_active, created_at)
    """)

    # Guarantee one default playlist per channel.
    channel_rows = cur.execute("SELECT id FROM channels").fetchall()
    for row in channel_rows:
        cur.execute(
            "INSERT OR IGNORE INTO playlists(channel_id, name, description, is_active) VALUES (?,?,?,1)",
            (row["id"], "grade_default", "Grade padrao automatica"),
        )

    # One-time migration from legacy channel_items -> playlist_items.
    has_playlist_items = cur.execute("SELECT COUNT(*) AS c FROM playlist_items").fetchone()["c"] > 0
    has_legacy_items = cur.execute("SELECT COUNT(*) AS c FROM channel_items").fetchone()["c"] > 0
    if has_legacy_items and not has_playlist_items:
        channels_with_items = cur.execute(
            "SELECT DISTINCT channel_id FROM channel_items ORDER BY channel_id"
        ).fetchall()
        for row in channels_with_items:
            channel_id = row["channel_id"]
            pl = cur.execute(
                "SELECT id FROM playlists WHERE channel_id=? ORDER BY id LIMIT 1",
                (channel_id,),
            ).fetchone()
            if not pl:
                continue
            cur.execute(
                """
                INSERT INTO playlist_items(playlist_id, position, type, url, duration)
                SELECT ?, position, type, url, duration
                FROM channel_items
                WHERE channel_id=?
                ORDER BY position
                """,
                (pl["id"], channel_id),
            )

    # -------------------------
    # Seed default data
    # -------------------------
    cur.execute("SELECT COUNT(*) AS c FROM providers")
    if cur.fetchone()["c"] == 0:
        cur.execute(
            "INSERT INTO providers(provider_id,name,description) VALUES (?,?,?)",
            ("prov-nex", "Nex (Default)", "Pacote padr\u00e3o do MVP")
        )

    cur.execute("SELECT COUNT(*) AS c FROM edges")
    if cur.fetchone()["c"] == 0:
        key = "edge_" + secrets.token_hex(8)
        cur.execute(
            "INSERT INTO edges(edge_id,name,api_key,hls_base_url,is_active) VALUES (?,?,?,?,1)",
            ("edge-001", "Edge Default", key, "http://EDGE_IP:8080/hls")
        )
        cur.execute(
            "INSERT OR IGNORE INTO edge_providers(edge_id,provider_id) VALUES (?,?)",
            ("edge-001", "prov-nex")
        )

    conn.commit()
    conn.close()


@app.on_event("startup")
def _startup():
    init_db()


def fetch_all(sql: str, args=()) -> List[sqlite3.Row]:
    conn = db()
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return rows


def fetch_one(sql: str, args=()):
    conn = db()
    row = conn.execute(sql, args).fetchone()
    conn.close()
    return row


def execute(sql: str, args=()) -> None:
    conn = db()
    try:
        conn.execute(sql, args)
        conn.commit()
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "UNIQUE constraint failed" in msg:
            raise HTTPException(status_code=409, detail=f"db unique violation: {msg}")
        raise HTTPException(status_code=400, detail=f"db integrity error: {msg}")
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "database is locked" in msg:
            raise HTTPException(status_code=503, detail="database is busy (locked)")
        raise HTTPException(status_code=500, detail=f"db operational error: {msg}")
    finally:
        conn.close()


def redirect_home_or_channel(channel_id: Optional[int]) -> RedirectResponse:
    if channel_id:
        return RedirectResponse(f"/admin/channels/{channel_id}", status_code=303)
    return RedirectResponse("/", status_code=303)


def ensure_default_playlist(channel_id: int, conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT id FROM playlists WHERE channel_id=? ORDER BY id LIMIT 1",
        (channel_id,),
    ).fetchone()
    if row:
        return int(row["id"])

    conn.execute(
        "INSERT INTO playlists(channel_id, name, description, is_active) VALUES (?,?,?,1)",
        (channel_id, "grade_default", "Grade padrao automatica"),
    )
    row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
    return int(row["id"])


def resolve_edge_programming(edge_id: str, channel_id: int) -> Dict[str, Any]:
    row = fetch_one(
        """
        SELECT ep.playlist_id, ep.schedule_start, p.name AS playlist_name
        FROM edge_programming ep
        JOIN playlists p ON p.id = ep.playlist_id
        WHERE ep.edge_id=? AND ep.channel_id=? AND ep.is_active=1
        ORDER BY ep.created_at DESC, ep.id DESC
        LIMIT 1
        """,
        (edge_id, channel_id),
    )
    if row:
        return {
            "playlist_id": row["playlist_id"],
            "playlist_name": row["playlist_name"],
            "schedule_start": row["schedule_start"],
        }

    fallback = fetch_one(
        """
        SELECT p.id AS playlist_id, p.name AS playlist_name
        FROM playlists p
        WHERE p.channel_id=? AND p.is_active=1
        ORDER BY p.id
        LIMIT 1
        """,
        (channel_id,),
    )
    if fallback:
        return {
            "playlist_id": fallback["playlist_id"],
            "playlist_name": fallback["playlist_name"],
            "schedule_start": None,
        }
    return {"playlist_id": None, "playlist_name": None, "schedule_start": None}


def enrich_channel_for_edge(channel_row: sqlite3.Row | Dict[str, Any], edge: sqlite3.Row) -> Dict[str, Any]:
    item = dict(channel_row)
    item["channel_id"] = item.get("channel_number")
    item["channel_pk"] = item.get("id")
    src = item.get("source_url") or ""
    kind = normalize_channel_kind(item.get("kind"), src)
    item["kind"] = kind

    hls_base = edge["hls_base_url"].rstrip("/")
    programming = resolve_edge_programming(edge["edge_id"], int(item["id"]))
    playlist_id = programming.get("playlist_id")
    schedule_start = programming.get("schedule_start") or item.get("schedule_start")

    if kind == "youtube":
        item["playback_url"] = src
        item["embed_url"] = youtube_to_embed(src)
        item.pop("schedule_start", None)
        item["playlist_id"] = None
        item["playlist_name"] = None
        item["now"] = None
    elif kind == "hls":
        item["playback_url"] = f"{hls_base}/{item['provider_id']}/{item['channel_number']}/index.m3u8"
        item["embed_url"] = None
        item.pop("schedule_start", None)
        item["playlist_id"] = None
        item["playlist_name"] = None
        item["now"] = None
    else:
        item["playback_url"] = None
        item["embed_url"] = None
        item["schedule_start"] = schedule_start
        item["playlist_id"] = playlist_id
        item["playlist_name"] = programming.get("playlist_name")

        if playlist_id:
            rows = fetch_all(
                """SELECT position, type, url, duration
                   FROM playlist_items
                   WHERE playlist_id=?
                   ORDER BY position""",
                (playlist_id,),
            )
        else:
            rows = []

        normalized_items = []
        for row in rows:
            payload_item = {
                "type": row["type"],
                "duration": row["duration"],
            }
            if row["type"] == "video" and row["url"]:
                payload_item["url"] = row["url"]
            normalized_items.append(payload_item)
        item["items"] = normalized_items
        item["now"] = compute_linear_now(schedule_start, normalized_items)

    return item


def must_auth_edge(request: Request) -> sqlite3.Row:
    api_key = request.headers.get("X-API-KEY") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(status_code=401, detail="missing X-API-KEY")
    edge = fetch_one("SELECT * FROM edges WHERE api_key=? AND is_active=1", (api_key,))
    if not edge:
        raise HTTPException(status_code=403, detail="invalid api key")
    return edge


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    channel_category = (request.query_params.get("channel_category") or "").strip()
    channel_grade_raw = (request.query_params.get("channel_grade_id") or "").strip()
    edge_country = (request.query_params.get("edge_country") or "").strip()
    edge_state = (request.query_params.get("edge_state") or "").strip()
    edge_city = (request.query_params.get("edge_city") or "").strip()

    channel_grade_id: Optional[int] = None
    if channel_grade_raw:
        try:
            channel_grade_id = int(channel_grade_raw)
        except Exception:
            channel_grade_id = None

    providers = fetch_all("SELECT * FROM providers ORDER BY provider_id")
    channel_grades = fetch_all("SELECT * FROM channel_grades ORDER BY name")
    category_rows = fetch_all(
        """SELECT DISTINCT category
           FROM channels
           WHERE TRIM(COALESCE(category, '')) <> ''
           ORDER BY category"""
    )
    channel_categories = [r["category"] for r in category_rows]

    channel_sql = """
        SELECT c.id, c.channel_number, c.name, c.category, c.provider_id, c.source_url, c.kind, c.schedule_start, c.is_active, c.sort_order, c.created_at,
               p.name AS provider_name,
               (
                   SELECT COUNT(*)
                   FROM playlist_items pi
                   JOIN playlists pl ON pl.id = pi.playlist_id
                   WHERE pl.channel_id = c.id
               ) AS linear_items_count
        FROM channels c
        JOIN providers p ON p.provider_id = c.provider_id
    """
    channel_args: List[Any] = []
    channel_where: List[str] = []
    if channel_grade_id:
        channel_sql += " JOIN channel_grade_channels cgc ON cgc.channel_id = c.id "
        channel_where.append("cgc.grade_id = ?")
        channel_args.append(channel_grade_id)
    if channel_category:
        channel_where.append("c.category = ?")
        channel_args.append(channel_category)
    if channel_where:
        channel_sql += " WHERE " + " AND ".join(channel_where)
    channel_sql += " ORDER BY p.provider_id, c.sort_order, c.channel_number"
    channels = fetch_all(channel_sql, tuple(channel_args))
    channels_all = fetch_all(
        """SELECT id, channel_number, name, provider_id
           FROM channels
           WHERE is_active=1
           ORDER BY provider_id, sort_order, channel_number"""
    )

    edge_sql = """
        SELECT e.*, g.name AS grade_name
        FROM edges e
        LEFT JOIN channel_grades g ON g.id = e.grade_id
    """
    edge_where: List[str] = []
    edge_args: List[Any] = []
    if edge_country:
        edge_where.append("e.country = ?")
        edge_args.append(edge_country)
    if edge_state:
        edge_where.append("e.state = ?")
        edge_args.append(edge_state)
    if edge_city:
        edge_where.append("e.city = ?")
        edge_args.append(edge_city)
    if edge_where:
        edge_sql += " WHERE " + " AND ".join(edge_where)
    edge_sql += " ORDER BY e.country, e.state, e.city, e.edge_id"
    edges = fetch_all(edge_sql, tuple(edge_args))

    country_rows = fetch_all(
        """SELECT DISTINCT country
           FROM edges
           WHERE TRIM(COALESCE(country, '')) <> ''
           ORDER BY country"""
    )
    edge_countries = [r["country"] for r in country_rows]

    state_sql = """
        SELECT DISTINCT state
        FROM edges
        WHERE TRIM(COALESCE(state, '')) <> ''
    """
    state_args: List[Any] = []
    if edge_country:
        state_sql += " AND country = ?"
        state_args.append(edge_country)
    state_sql += " ORDER BY state"
    edge_states = [r["state"] for r in fetch_all(state_sql, tuple(state_args))]

    city_sql = """
        SELECT DISTINCT city
        FROM edges
        WHERE TRIM(COALESCE(city, '')) <> ''
    """
    city_args: List[Any] = []
    if edge_country:
        city_sql += " AND country = ?"
        city_args.append(edge_country)
    if edge_state:
        city_sql += " AND state = ?"
        city_args.append(edge_state)
    city_sql += " ORDER BY city"
    edge_cities = [r["city"] for r in fetch_all(city_sql, tuple(city_args))]

    grade_channels = fetch_all(
        """SELECT gc.grade_id, g.name AS grade_name, gc.channel_id, gc.sort_order,
                  c.channel_number, c.name AS channel_name, c.provider_id, p.name AS provider_name
           FROM channel_grade_channels gc
           JOIN channel_grades g ON g.id = gc.grade_id
           JOIN channels c ON c.id = gc.channel_id
           JOIN providers p ON p.provider_id = c.provider_id
           {where_clause}
           ORDER BY g.name, gc.sort_order, c.channel_number"""
           .format(where_clause="WHERE gc.grade_id = ?" if channel_grade_id else ""),
        (channel_grade_id,) if channel_grade_id else (),
    )

    return templates.TemplateResponse("home.html", {
        "request": request,
        "providers": providers,
        "edges": edges,
        "channels": channels,
        "channels_all": channels_all,
        "channel_categories": channel_categories,
        "channel_grades": channel_grades,
        "grade_channels": grade_channels,
        "selected_channel_category": channel_category,
        "selected_channel_grade_id": str(channel_grade_id) if channel_grade_id else "",
        "edge_countries": edge_countries,
        "edge_states": edge_states,
        "edge_cities": edge_cities,
        "selected_edge_country": edge_country,
        "selected_edge_state": edge_state,
        "selected_edge_city": edge_city,
    })


@app.post("/admin/providers/create")
def create_provider(provider_id: str = Form(...), name: str = Form(...), description: str = Form("")):
    execute("INSERT INTO providers(provider_id,name,description) VALUES (?,?,?)",
            (provider_id.strip(), name.strip(), description.strip()))
    return RedirectResponse("/", status_code=303)


@app.post("/admin/providers/delete")
def delete_provider(provider_id: str = Form(...)):
    execute("DELETE FROM providers WHERE provider_id=?", (provider_id,))
    return RedirectResponse("/", status_code=303)


@app.post("/admin/channel-grades/create")
def create_channel_grade(name: str = Form(...), description: str = Form("")):
    execute(
        "INSERT INTO channel_grades(name,description,is_active) VALUES (?,?,1)",
        (name.strip(), description.strip()),
    )
    return RedirectResponse("/", status_code=303)


@app.post("/admin/channel-grades/delete")
def delete_channel_grade(grade_id: int = Form(...)):
    execute("UPDATE edges SET grade_id=NULL WHERE grade_id=?", (grade_id,))
    execute("DELETE FROM channel_grades WHERE id=?", (grade_id,))
    return RedirectResponse("/", status_code=303)


@app.post("/admin/channel-grades/channels/add")
def add_channel_to_grade(
    grade_id: int = Form(...),
    channel_id: int = Form(...),
    sort_order: int = Form(100),
):
    try:
        sort_order_i = int(sort_order)
    except Exception:
        raise HTTPException(status_code=400, detail="sort_order must be numeric")

    conn = db()
    try:
        conn.execute(
            """
            INSERT INTO channel_grade_channels(grade_id, channel_id, sort_order)
            VALUES (?,?,?)
            ON CONFLICT(grade_id, channel_id) DO UPDATE SET sort_order=excluded.sort_order
            """,
            (grade_id, channel_id, sort_order_i),
        )
        conn.commit()
    finally:
        conn.close()
    return RedirectResponse("/", status_code=303)


@app.post("/admin/channel-grades/channels/remove")
def remove_channel_from_grade(grade_id: int = Form(...), channel_id: int = Form(...)):
    execute(
        "DELETE FROM channel_grade_channels WHERE grade_id=? AND channel_id=?",
        (grade_id, channel_id),
    )
    return RedirectResponse("/", status_code=303)


@app.post("/admin/channels/create")
def create_channel(
    channel_number: str = Form(...),
    name: str = Form(...),
    category: str = Form(...),
    provider_id: str = Form(...),
    source_url: str = Form(""),
    kind: str = Form("auto"),
    schedule_start: str = Form(""),
    sort_order: int = Form(100),
    is_active: int = Form(1),
):
    try:
        sort_order_i = int(sort_order)
        is_active_i = int(is_active)
    except Exception:
        raise HTTPException(status_code=400, detail="sort_order/is_active must be numeric")

    conn = db()
    try:
        kind_clean = (kind or "auto").strip().lower()
        if kind_clean not in ("auto", "hls", "youtube", "youtube_linear"):
            kind_clean = "auto"
        schedule_clean = schedule_start.strip() or None
        conn.execute(
            """INSERT INTO channels(channel_number,name,category,provider_id,source_url,kind,schedule_start,sort_order,is_active)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                channel_number.strip(),
                name.strip(),
                category.strip(),
                provider_id,
                source_url.strip(),
                kind_clean,
                schedule_clean,
                sort_order_i,
                is_active_i,
            ),
        )
        channel_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        ensure_default_playlist(int(channel_id), conn)
        conn.commit()
    except sqlite3.IntegrityError as e:
        msg = str(e)
        if "UNIQUE constraint failed" in msg and "channels.provider_id" in msg:
            raise HTTPException(
                status_code=409,
                detail="Numero de canal ja existe neste Provider. Apague o canal antigo ou use outro numero.",
            )
        raise HTTPException(status_code=400, detail=f"db integrity error: {msg}")
    finally:
        conn.close()
    return RedirectResponse("/", status_code=303)


@app.post("/admin/channels/delete")
def delete_channel(id: int = Form(...)):
    execute("DELETE FROM channels WHERE id=?", (id,))
    return RedirectResponse("/", status_code=303)


@app.get("/admin/channels/{channel_id}", response_class=HTMLResponse)
def channel_detail(channel_id: int, request: Request):
    channel = fetch_one(
        """SELECT c.*, p.name AS provider_name
           FROM channels c
           JOIN providers p ON p.provider_id = c.provider_id
           WHERE c.id=?""",
        (channel_id,),
    )
    if not channel:
        return RedirectResponse("/", status_code=303)

    playlists = fetch_all(
        """SELECT id, channel_id, name, description, is_active, created_at
           FROM playlists
           WHERE channel_id=?
           ORDER BY id""",
        (channel_id,),
    )
    playlist_items = fetch_all(
        """SELECT pi.id, pi.playlist_id, p.name AS playlist_name, pi.position, pi.type, pi.url, pi.duration
           FROM playlist_items pi
           JOIN playlists p ON p.id = pi.playlist_id
           WHERE p.channel_id=?
           ORDER BY p.id, pi.position""",
        (channel_id,),
    )
    return templates.TemplateResponse(
        "channel_detail.html",
        {
            "request": request,
            "channel": channel,
            "playlists": playlists,
            "playlist_items": playlist_items,
        },
    )


@app.post("/admin/channel-items/create")
def create_channel_item(
    channel_id: Optional[int] = Form(None),
    playlist_id: Optional[int] = Form(None),
    return_channel_id: Optional[int] = Form(None),
    position: int = Form(...),
    item_type: str = Form(..., alias="type"),
    url: str = Form(None)
):
    # NOTE: duração NÃO vem mais do form. É 100% automática aqui.
    try:
        position_i = int(position)
    except Exception:
        raise HTTPException(status_code=400, detail="position must be numeric")

    if position_i < 1:
        raise HTTPException(status_code=400, detail="position must be >= 1")

    item_type_clean = (item_type or "").strip().lower()
    if item_type_clean not in ("video", "ad"):
        raise HTTPException(status_code=400, detail="type must be video or ad")

    url_clean = (url or "").strip()
    if item_type_clean == "video" and not url_clean:
        raise HTTPException(status_code=400, detail="video items require url")

    # AUTO duration logic (robust)
    if item_type_clean == "ad":
        duration_i = 30
        url_db = None
    else:
        url_db = url_clean
        if is_youtube_url(url_clean):
            try:
                d = youtube_duration_seconds(url_clean)
                duration_i = d if d and d > 0 else 600
            except Exception:
                duration_i = 600
        else:
            duration_i = 600

    conn = db()
    try:
        target_playlist_id = playlist_id
        if not target_playlist_id:
            if not channel_id:
                raise HTTPException(status_code=400, detail="playlist_id or channel_id is required")
            target_playlist_id = ensure_default_playlist(int(channel_id), conn)

        conn.execute(
            "INSERT INTO playlist_items(playlist_id, position, type, url, duration) VALUES (?,?,?,?,?)",
            (int(target_playlist_id), position_i, item_type_clean, url_db, int(duration_i)),
        )
        conn.commit()
    finally:
        conn.close()

    return redirect_home_or_channel(return_channel_id)


@app.post("/admin/channel-items/delete")
def delete_channel_item(id: int = Form(...), return_channel_id: Optional[int] = Form(None)):
    execute("DELETE FROM playlist_items WHERE id=?", (id,))
    return redirect_home_or_channel(return_channel_id)


@app.post("/admin/playlists/create")
def create_playlist(
    channel_id: int = Form(...),
    name: str = Form(...),
    description: str = Form(""),
    return_channel_id: Optional[int] = Form(None),
    is_active: int = Form(1),
):
    try:
        active_i = int(is_active)
    except Exception:
        raise HTTPException(status_code=400, detail="is_active must be numeric")

    execute(
        "INSERT INTO playlists(channel_id,name,description,is_active) VALUES (?,?,?,?)",
        (channel_id, name.strip(), description.strip(), active_i),
    )
    return redirect_home_or_channel(return_channel_id or channel_id)


@app.post("/admin/playlists/delete")
def delete_playlist(id: int = Form(...), return_channel_id: Optional[int] = Form(None)):
    execute("DELETE FROM playlists WHERE id=?", (id,))
    return redirect_home_or_channel(return_channel_id)


@app.post("/admin/edge-programming/create")
def create_edge_programming(
    edge_id: str = Form(...),
    channel_id: int = Form(...),
    playlist_id: int = Form(...),
    schedule_start: str = Form(...),
    is_active: int = Form(1),
):
    schedule_clean = (schedule_start or "").strip()
    if not parse_datetime_utc(schedule_clean):
        raise HTTPException(status_code=400, detail="schedule_start must be an ISO-8601 datetime")

    try:
        active_i = int(is_active)
    except Exception:
        raise HTTPException(status_code=400, detail="is_active must be numeric")

    row = fetch_one("SELECT id FROM playlists WHERE id=? AND channel_id=?", (playlist_id, channel_id))
    if not row:
        raise HTTPException(status_code=400, detail="playlist does not belong to channel")

    scope_ok = fetch_one(
        """
        SELECT 1
        FROM channels c
        JOIN edge_providers ep ON ep.provider_id = c.provider_id
        WHERE c.id=? AND ep.edge_id=?
        LIMIT 1
        """,
        (channel_id, edge_id),
    )
    if not scope_ok:
        raise HTTPException(status_code=400, detail="edge does not have permission for this channel provider")

    execute(
        "INSERT INTO edge_programming(edge_id,channel_id,playlist_id,schedule_start,is_active) VALUES (?,?,?,?,?)",
        (edge_id, channel_id, playlist_id, schedule_clean, active_i),
    )
    return RedirectResponse("/", status_code=303)


@app.post("/admin/edge-programming/delete")
def delete_edge_programming(id: int = Form(...)):
    execute("DELETE FROM edge_programming WHERE id=?", (id,))
    return RedirectResponse("/", status_code=303)


def gen_api_key() -> str:
    return "edge_" + secrets.token_hex(12)


@app.post("/admin/edges/create")
def create_edge(
    edge_id: str = Form(...),
    name: str = Form(...),
    hls_base_url: str = Form(...),
    country: str = Form(""),
    state: str = Form(""),
    city: str = Form(""),
):
    api_key = gen_api_key()
    execute(
        """INSERT INTO edges(edge_id,name,api_key,hls_base_url,country,state,city,is_active)
           VALUES (?,?,?,?,?,?,?,1)""",
        (
            edge_id.strip(),
            name.strip(),
            api_key,
            hls_base_url.strip(),
            country.strip(),
            state.strip(),
            city.strip(),
        ),
    )
    return RedirectResponse("/", status_code=303)


@app.post("/admin/edges/rotate_key")
def rotate_key(edge_id: str = Form(...)):
    api_key = gen_api_key()
    execute("UPDATE edges SET api_key=? WHERE edge_id=?", (api_key, edge_id))
    return RedirectResponse("/", status_code=303)


@app.post("/admin/edges/delete")
def delete_edge(edge_id: str = Form(...)):
    execute("DELETE FROM edges WHERE edge_id=?", (edge_id,))
    return RedirectResponse("/", status_code=303)


@app.post("/admin/edges/update-location")
def update_edge_location(
    edge_id: str = Form(...),
    country: str = Form(""),
    state: str = Form(""),
    city: str = Form(""),
):
    execute(
        "UPDATE edges SET country=?, state=?, city=? WHERE edge_id=?",
        (country.strip(), state.strip(), city.strip(), edge_id),
    )
    return RedirectResponse("/", status_code=303)


@app.post("/admin/edges/assign-grade")
def assign_edge_grade(edge_id: str = Form(...), grade_id: str = Form("")):
    grade_clean = (grade_id or "").strip()
    if grade_clean:
        try:
            grade_int = int(grade_clean)
        except Exception:
            raise HTTPException(status_code=400, detail="grade_id must be numeric")
        exists = fetch_one("SELECT id FROM channel_grades WHERE id=?", (grade_int,))
        if not exists:
            raise HTTPException(status_code=404, detail="grade not found")
        execute("UPDATE edges SET grade_id=? WHERE edge_id=?", (grade_int, edge_id))
    else:
        execute("UPDATE edges SET grade_id=NULL WHERE edge_id=?", (edge_id,))
    return RedirectResponse("/", status_code=303)


@app.get("/admin/edges/{edge_id}/providers", response_class=HTMLResponse)
def edge_providers_page(edge_id: str, request: Request):
    edge = fetch_one("SELECT * FROM edges WHERE edge_id=?", (edge_id,))
    if not edge:
        return RedirectResponse("/", status_code=303)

    providers = fetch_all("SELECT * FROM providers ORDER BY provider_id")
    current = set([r["provider_id"] for r in fetch_all("SELECT provider_id FROM edge_providers WHERE edge_id=?", (edge_id,))])

    return templates.TemplateResponse("edge_providers.html", {
        "request": request,
        "edge": edge,
        "providers": providers,
        "current": current
    })


@app.post("/admin/edges/{edge_id}/providers/save")
def edge_providers_save(edge_id: str, provider_ids: Optional[List[str]] = Form(None)):
    execute("DELETE FROM edge_providers WHERE edge_id=?", (edge_id,))
    if provider_ids:
        conn = db()
        for pid in provider_ids:
            conn.execute("INSERT OR IGNORE INTO edge_providers(edge_id,provider_id) VALUES (?,?)", (edge_id, pid))
        conn.commit()
        conn.close()
    return RedirectResponse(f"/admin/edges/{edge_id}/providers", status_code=303)


@app.get("/api/edge/channels")
def api_edge_channels(request: Request):
    edge = must_auth_edge(request)
    channels_by_provider: Dict[str, List[sqlite3.Row]] = {}
    provider_order: List[str] = []

    # Main workflow: edge bound to one distribution grade (lista de canais).
    if edge["grade_id"]:
        rows = fetch_all(
            """
            SELECT c.id, c.channel_number, c.name, c.category, c.provider_id, c.source_url, c.kind, c.schedule_start, c.is_active, c.sort_order,
                   gc.sort_order AS grade_sort_order
            FROM channel_grade_channels gc
            JOIN channels c ON c.id = gc.channel_id
            WHERE gc.grade_id=? AND c.is_active=1
            ORDER BY gc.sort_order, c.channel_number
            """,
            (edge["grade_id"],),
        )
        for row in rows:
            pid = row["provider_id"]
            if pid not in channels_by_provider:
                channels_by_provider[pid] = []
                provider_order.append(pid)
            channels_by_provider[pid].append(row)
    else:
        # Automatic general grade fallback: all active channels.
        channel_rows = fetch_all(
            """
            SELECT id, channel_number, name, category, provider_id, source_url, kind, schedule_start, is_active, sort_order
            FROM channels
            WHERE is_active=1
            ORDER BY provider_id, sort_order, channel_number
            """
        )
        for row in channel_rows:
            pid = row["provider_id"]
            if pid not in channels_by_provider:
                channels_by_provider[pid] = []
                provider_order.append(pid)
            channels_by_provider[pid].append(row)

    if not channels_by_provider:
        return {"edge_id": edge["edge_id"], "hls_base_url": edge["hls_base_url"], "providers": []}

    placeholders = ",".join(["?"] * len(provider_order))
    p_rows = fetch_all(
        f"SELECT provider_id, name FROM providers WHERE provider_id IN ({placeholders})",
        tuple(provider_order),
    )
    p_names = {r["provider_id"]: r["name"] for r in p_rows}

    result = []
    for pid in provider_order:
        rows = channels_by_provider.get(pid, [])
        enriched = [enrich_channel_for_edge(r, edge) for r in rows]
        result.append({
            "provider_id": pid,
            "provider_name": p_names.get(pid, pid),
            "channels": enriched,
        })

    return {"edge_id": edge["edge_id"], "hls_base_url": edge["hls_base_url"], "providers": result}


@app.get("/iptv/edge.m3u")
def playlist_for_edge(request: Request):
    edge = must_auth_edge(request)
    payload = api_edge_channels(request)
    hls_base = payload["hls_base_url"].rstrip("/")

    lines = ["#EXTM3U"]
    for p in payload["providers"]:
        grp = p["provider_name"]
        for ch in p["channels"]:
            if ch.get("kind") != "hls":
                continue
            name = ch["name"]
            cnum = ch.get("channel_number") or ""
            display = f"{cnum} - {name}".strip(" -")
            url = ch.get("playback_url") or f"{hls_base}/{ch['provider_id']}/{cnum}/index.m3u8"
            lines.append(f'#EXTINF:-1 group-title="{grp}",{display}')
            lines.append(url)

    return PlainTextResponse("\n".join(lines), media_type="audio/x-mpegurl")


@app.get("/health")
def health():
    return {"ok": True}
