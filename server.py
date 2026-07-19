"""Local learning-platform server for the KP regression book.

Run with:

    python3 server.py

The server serves SRC_HTML/ and exposes the /api/* endpoints used by the
compiled pages. Online providers are configured through config.yaml and/or
environment variables; no real keys are stored in this repository.
"""

from __future__ import annotations

import hashlib
import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import shutil
import socket
import sqlite3
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "SRC_HTML"
DB_PATH = ROOT / "learning_platform.db"
CONFIG_YAML_PATH = ROOT / "config.yaml"
CONFIG_JSON_PATH = ROOT / "config.json"
AUDIO_CACHE = ROOT / "site" / "static" / "audio_cache"
ATTACHMENT_DIR = ROOT / "site" / "static" / "note_attachments"
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
SERVICES = ("tutor", "grading", "papers", "tts", "sync")

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {"host": "127.0.0.1", "port": 9001},
    "services": {
        "tutor": {
            "enabled": True,
            "provider": "anthropic",
            "endpoint": "https://api.anthropic.com/v1/messages",
            "api_key": "${ANTHROPIC_API_KEY}",
            "model": "${ANTHROPIC_MODEL}",
            "timeout_seconds": 3,
        },
        "grading": {
            "enabled": True,
            "provider": "anthropic",
            "endpoint": "https://api.anthropic.com/v1/messages",
            "api_key": "${ANTHROPIC_API_KEY}",
            "model": "${ANTHROPIC_MODEL}",
            "timeout_seconds": 3,
        },
        "papers": {
            "enabled": True,
            "provider": "semantic_scholar",
            "endpoint": "https://api.semanticscholar.org/graph/v1/paper/search",
            "timeout_seconds": 3,
        },
        "tts": {
            "enabled": True,
            "provider": "openai",
            "endpoint": "https://api.openai.com/v1/audio/speech",
            "api_key": "${OPENAI_API_KEY}",
            "model": "${OPENAI_TTS_MODEL}",
            "voice": "${OPENAI_TTS_VOICE}",
            "timeout_seconds": 3,
        },
        "sync": {
            "enabled": True,
            "endpoint": "${KP_SYNC_ENDPOINT}",
            "api_key": "${KP_SYNC_API_KEY}",
            "timeout_seconds": 3,
        },
    },
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if value.lower() in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def read_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the small config subset this project uses.

    This avoids adding PyYAML as a dependency. It supports nested mappings via
    two-space indentation and scalar values.
    """

    if not path.exists():
        return {}
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_raw_config() -> dict[str, Any]:
    config = deep_merge(DEFAULT_CONFIG, read_simple_yaml(CONFIG_YAML_PATH))
    return deep_merge(config, read_json_config(CONFIG_JSON_PATH))


def active_config_path() -> Path:
    return CONFIG_JSON_PATH if CONFIG_JSON_PATH.exists() else CONFIG_YAML_PATH


def expand_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return os.path.expandvars(value)


def resolve_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_env(v) for v in obj]
    return expand_env(obj)


CONFIG_RAW = load_raw_config()
CONFIG = resolve_env(CONFIG_RAW)


def service_config(name: str) -> dict[str, Any]:
    return dict(CONFIG.get("services", {}).get(name, {}))


def reload_config() -> None:
    global CONFIG_RAW, CONFIG
    CONFIG_RAW = load_raw_config()
    CONFIG = resolve_env(CONFIG_RAW)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return ""
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]"]) or text.strip() != text:
        return json.dumps(text)
    return text


def write_simple_yaml(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []

    def emit_mapping(mapping: dict[str, Any], indent: int = 0) -> None:
        for key, value in mapping.items():
            prefix = " " * indent + str(key) + ":"
            if isinstance(value, dict):
                lines.append(prefix)
                emit_mapping(value, indent + 2)
            else:
                lines.append(prefix + (" " + yaml_scalar(value) if yaml_scalar(value) else ""))

    emit_mapping(data)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_yaml_lines(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    def emit_mapping(mapping: dict[str, Any], indent: int = 0) -> None:
        for key, value in mapping.items():
            prefix = " " * indent + str(key) + ":"
            if isinstance(value, dict):
                lines.append(prefix)
                emit_mapping(value, indent + 2)
            else:
                lines.append(prefix + (" " + yaml_scalar(value) if yaml_scalar(value) else ""))

    emit_mapping(data)
    return lines


def mask_secret_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        return value
    if len(value) <= 8:
        return "********"
    return value[:4] + "..." + value[-4:]


def mask_embedded_secrets(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    patterns = [
        r"AIza[0-9A-Za-z_\-]{20,}",
        r"sk-ant-[0-9A-Za-z_\-]{20,}",
        r"sk-[0-9A-Za-z_\-]{20,}",
        r"xai-[0-9A-Za-z_\-]{20,}",
    ]
    masked = value
    for pattern in patterns:
        masked = re.sub(pattern, lambda m: mask_secret_value(m.group(0)), masked)
    return masked


def clean_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return "" if value == "[object Object]" else value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return ""


def mask_config_secrets(value: Any, key_name: str = "") -> Any:
    secret_keys = {"api_key", "token", "secret", "password"}
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, child in value.items():
            if key.lower() in secret_keys:
                masked[key] = mask_secret_value(child)
            else:
                masked[key] = mask_config_secrets(child, key)
        return masked
    if isinstance(value, list):
        return [mask_config_secrets(child, key_name) for child in value]
    return mask_secret_value(value) if key_name.lower() in secret_keys else mask_embedded_secrets(value)


def config_file_views() -> dict[str, Any]:
    files = []
    for path, kind in [(CONFIG_YAML_PATH, "yaml"), (CONFIG_JSON_PATH, "json")]:
        if kind == "json":
            parsed = read_json_config(path)
            content = json.dumps(mask_config_secrets(parsed), indent=2, ensure_ascii=False) if path.exists() else ""
        else:
            parsed = read_simple_yaml(path)
            content = "\n".join(render_yaml_lines(mask_config_secrets(parsed))) if path.exists() else ""
        files.append(
            {
                "name": path.name,
                "exists": path.exists(),
                "active_write_target": path == active_config_path(),
                "content": content,
            }
        )
    return {"files": files}


def public_settings() -> dict[str, Any]:
    services: dict[str, Any] = {}
    raw_services = CONFIG_RAW.get("services", {})
    resolved_services = CONFIG.get("services", {})
    for service in SERVICES:
        raw = dict(raw_services.get(service, {}))
        resolved = dict(resolved_services.get(service, {}))
        item = {
            "enabled": bool(resolved.get("enabled", True)),
            "provider": clean_config_value(resolved.get("provider", "")),
            "endpoint": mask_embedded_secrets(clean_config_value(resolved.get("endpoint", ""))),
            "model": mask_embedded_secrets(clean_config_value(resolved.get("model", ""))),
            "voice": mask_embedded_secrets(clean_config_value(resolved.get("voice", ""))),
            "timeout_seconds": clean_config_value(resolved.get("timeout_seconds", 3)) or 3,
            "mailto": mask_embedded_secrets(clean_config_value(resolved.get("mailto", ""))),
            "api_key": "",
            "api_key_configured": bool(resolved.get("api_key")),
            "api_key_source": mask_secret_value(raw.get("api_key", "")),
        }
        services[service] = item
    return {"server": CONFIG.get("server", {}), "services": services, "active_config_file": active_config_path().name}


def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    config = deep_merge(CONFIG_RAW, {})
    incoming_services = payload.get("services", {})
    config.setdefault("services", {})
    for service in SERVICES:
        incoming = incoming_services.get(service)
        if not isinstance(incoming, dict):
            continue
        current = dict(config["services"].get(service, {}))
        for key in ["enabled", "provider", "endpoint", "model", "voice", "timeout_seconds", "mailto"]:
            if key in incoming:
                current[key] = clean_config_value(incoming[key])
        if incoming.get("clear_api_key"):
            current["api_key"] = ""
        elif incoming.get("api_key"):
            current["api_key"] = incoming["api_key"]
        config["services"][service] = current
    target = active_config_path()
    if target == CONFIG_JSON_PATH:
        target.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        write_simple_yaml(config, target)
    reload_config()
    return public_settings()


def init_db() -> None:
    AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS service_config (
                service_name TEXT PRIMARY KEY,
                enabled      INTEGER NOT NULL DEFAULT 1,
                last_check   TEXT,
                last_status  TEXT
            );

            CREATE TABLE IF NOT EXISTS notes (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_slug TEXT NOT NULL,
                section_id   TEXT NOT NULL,
                title        TEXT NOT NULL DEFAULT 'Untitled note',
                body         TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS note_attachments (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id       INTEGER NOT NULL,
                original_name TEXT NOT NULL,
                stored_name   TEXT NOT NULL,
                content_type  TEXT NOT NULL,
                size_bytes    INTEGER NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY(note_id) REFERENCES notes(id)
            );

            CREATE TABLE IF NOT EXISTS completed_sections (
                chapter_slug TEXT NOT NULL,
                section_id   TEXT NOT NULL,
                completed    INTEGER NOT NULL DEFAULT 0,
                updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (chapter_slug, section_id)
            );

            CREATE TABLE IF NOT EXISTS tutor_conversations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_slug  TEXT NOT NULL,
                section_id    TEXT NOT NULL,
                role          TEXT NOT NULL,
                body          TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS exercise_submissions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_slug  TEXT NOT NULL,
                section_id    TEXT NOT NULL,
                submission    TEXT NOT NULL,
                feedback      TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS paper_cache (
                citation_key  TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                fetched_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS review_queue (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter_slug  TEXT NOT NULL,
                section_id    TEXT NOT NULL,
                due_at        TEXT,
                state         TEXT NOT NULL DEFAULT 'pending',
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        for service in SERVICES:
            cfg = service_config(service)
            conn.execute(
                """
                INSERT INTO service_config(service_name, enabled, last_status)
                VALUES (?, ?, COALESCE((SELECT last_status FROM service_config WHERE service_name = ?), 'provider_error'))
                ON CONFLICT(service_name) DO UPDATE SET enabled = excluded.enabled
                """,
                (service, 1 if cfg.get("enabled", True) else 0, service),
            )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
        if "id" not in columns:
            conn.execute("ALTER TABLE notes RENAME TO notes_legacy")
            conn.execute(
                """
                CREATE TABLE notes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    chapter_slug TEXT NOT NULL,
                    section_id   TEXT NOT NULL,
                    title        TEXT NOT NULL DEFAULT 'Untitled note',
                    body         TEXT NOT NULL DEFAULT '',
                    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                INSERT INTO notes(chapter_slug, section_id, title, body, created_at, updated_at)
                SELECT chapter_slug, section_id, 'Section note', body, COALESCE(updated_at, datetime('now')), COALESCE(updated_at, datetime('now'))
                FROM notes_legacy
                WHERE COALESCE(body, '') != ''
                """
            )
            conn.execute("DROP TABLE notes_legacy")


def db_rows(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(query, params).fetchall()


def db_execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(query, params)


def update_service_status(service: str, status: str) -> None:
    db_execute(
        """
        UPDATE service_config
        SET last_check = datetime('now'), last_status = ?
        WHERE service_name = ?
        """,
        (status, service),
    )


def needs_api_key(cfg: dict[str, Any]) -> bool:
    provider = str(cfg.get("provider", "")).lower()
    return provider in {"anthropic", "openai", "elevenlabs", "gemini", "google_tts", "lmstudio_auth"} or "api_key" in cfg


def has_required_config(service: str, cfg: dict[str, Any]) -> tuple[bool, str | None]:
    if not cfg.get("enabled", True):
        return False, "provider_error"
    if service == "sync" and (not cfg.get("endpoint") or not cfg.get("api_key")):
        return False, "no_api_key"
    if service == "tts" and not cfg.get("api_key"):
        return False, "no_api_key"
    if service in {"tutor", "grading"}:
        provider = str(cfg.get("provider", "")).lower()
        if provider == "lmstudio":
            return True, None
        if needs_api_key(cfg) and not cfg.get("api_key"):
            return False, "no_api_key"
    return True, None


def probe_service(service: str) -> dict[str, Any]:
    cfg = service_config(service)
    ok, reason = has_required_config(service, cfg)
    if not ok:
        update_service_status(service, reason or "provider_error")
        return {"available": False, "reason": reason}

    provider = str(cfg.get("provider", "")).lower()
    endpoint = str(cfg.get("endpoint", "")).strip()
    if "{model}" in endpoint:
        endpoint = endpoint.replace("{model}", urllib.parse.quote(str(cfg.get("model") or "gemini-flash-latest"), safe=""))
    if not endpoint:
        update_service_status(service, "provider_error")
        return {"available": False, "reason": "provider_error"}

    timeout = float(cfg.get("timeout_seconds") or 3)
    headers = {"User-Agent": "KPRegressionBook/1.0"}
    probe_endpoint = endpoint
    probe_method = "HEAD"
    if cfg.get("api_key"):
        if provider == "anthropic":
            headers["x-api-key"] = str(cfg["api_key"])
            headers["anthropic-version"] = "2023-06-01"
        elif provider == "gemini":
            model = urllib.parse.quote(str(cfg.get("model") or "gemini-flash-latest"), safe="")
            probe_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}?key={urllib.parse.quote(str(cfg['api_key']))}"
            probe_method = "GET"
        elif provider == "google_tts":
            sep = "&" if "?" in probe_endpoint else "?"
            probe_endpoint = f"{probe_endpoint}{sep}key={urllib.parse.quote(str(cfg['api_key']))}"
        else:
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
    request = urllib.request.Request(probe_endpoint, method=probe_method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = "ok" if response.status < 500 else "provider_error"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            status = "no_api_key"
        elif exc.code >= 500:
            status = "provider_error"
        else:
            status = "ok"
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        status = "no_connectivity"

    update_service_status(service, status)
    return {"available": status == "ok", "reason": None if status == "ok" else status}


def probe_all_services() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SERVICES)) as executor:
        futures = {executor.submit(probe_service, service): service for service in SERVICES}
        for future in concurrent.futures.as_completed(futures):
            service = futures[future]
            try:
                results[service] = future.result()
            except Exception:
                update_service_status(service, "provider_error")
                results[service] = {"available": False, "reason": "provider_error"}
    return {service: results.get(service, {"available": False, "reason": "provider_error"}) for service in SERVICES}


def read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def safe_filename(name: str) -> str:
    base = Path(name or "attachment").name.strip() or "attachment"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return cleaned[:180] or "attachment"


def parse_disposition_value(disposition: str, key: str) -> str:
    match = re.search(rf'{key}="([^"]*)"', disposition)
    if match:
        return match.group(1)
    match = re.search(rf"{key}=([^;]+)", disposition)
    return match.group(1).strip() if match else ""


def parse_multipart(handler: SimpleHTTPRequestHandler) -> tuple[dict[str, str], list[dict[str, Any]]]:
    content_type = handler.headers.get("Content-Type", "")
    boundary_match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type)
    if not boundary_match:
        raise ValueError("missing_boundary")
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}, []
    if length > MAX_ATTACHMENT_BYTES:
        raise ValueError("payload_too_large")

    boundary = (boundary_match.group(1) or boundary_match.group(2)).encode("utf-8")
    body = handler.rfile.read(length)
    fields: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    for raw_part in body.split(b"--" + boundary):
        if not raw_part or raw_part in {b"--", b"--\r\n"}:
            continue
        part = raw_part
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"--"):
            part = part[:-2]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        header_blob, separator, content = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers: dict[str, str] = {}
        for line in header_blob.decode("utf-8", errors="replace").split("\r\n"):
            key, sep, value = line.partition(":")
            if sep:
                headers[key.strip().lower()] = value.strip()
        disposition = headers.get("content-disposition", "")
        field_name = parse_disposition_value(disposition, "name")
        if not field_name:
            continue
        filename = parse_disposition_value(disposition, "filename")
        if filename:
            files.append(
                {
                    "field_name": field_name,
                    "filename": filename,
                    "content_type": headers.get("content-type") or mimetypes.guess_type(filename)[0] or "application/octet-stream",
                    "content": content,
                }
            )
        else:
            fields[field_name] = content.decode("utf-8", errors="replace")
    return fields, files


def attachment_payload(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["url"] = f"/api/attachments/{item['id']}"
    item["is_image"] = str(item.get("content_type") or "").startswith("image/")
    return item


def attachment_rows(note_id: str) -> list[dict[str, Any]]:
    rows = db_rows(
        """
        SELECT id, note_id, original_name, content_type, size_bytes, created_at
        FROM note_attachments
        WHERE note_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (note_id,),
    )
    return [attachment_payload(row) for row in rows]


def json_response(handler: SimpleHTTPRequestHandler, data: Any, status: int = 200) -> None:
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_stream_response(handler: SimpleHTTPRequestHandler, text: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    encoded = text.encode("utf-8")
    for start in range(0, len(encoded), 256):
        handler.wfile.write(encoded[start : start + 256])
        handler.wfile.flush()
        time.sleep(0.01)


def unavailable(handler: SimpleHTTPRequestHandler, service: str) -> bool:
    status = probe_service(service)
    if status["available"]:
        return False
    json_response(handler, {"error": f"{service} unavailable", "reason": status["reason"]}, 503)
    return True


def call_llm(service: str, prompt: str) -> str:
    cfg = service_config(service)
    provider = str(cfg.get("provider", "anthropic")).lower()
    timeout = max(10, int(cfg.get("request_timeout_seconds") or 45))
    if provider == "gemini":
        model = cfg.get("model") or "gemini-flash-latest"
        endpoint = str(cfg.get("endpoint") or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
        endpoint = endpoint.replace("{model}", urllib.parse.quote(str(model), safe=""))
        sep = "&" if "?" in endpoint else "?"
        endpoint = f"{endpoint}{sep}key={urllib.parse.quote(str(cfg.get('api_key', '')))}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        candidates = data.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return "\n".join(part.get("text", "") for part in parts).strip()
    if provider in {"lmstudio", "openai-compatible", "openai"}:
        endpoint = str(cfg.get("endpoint") or "http://127.0.0.1:1234/v1/chat/completions")
        payload = {
            "model": cfg.get("model") or "local-model",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {"Content-Type": "application/json"}
        if cfg.get("api_key"):
            headers["Authorization"] = f"Bearer {cfg['api_key']}"
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    payload = {
        "model": cfg.get("model") or "claude-3-5-sonnet-latest",
        "max_tokens": int(cfg.get("max_tokens") or 1200),
        "system": "You are a precise regression/ML tutor. Ground every answer in the supplied book section.",
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": str(cfg.get("api_key", "")),
        "anthropic-version": "2023-06-01",
    }
    request = urllib.request.Request(str(cfg["endpoint"]), data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    parts = data.get("content", [])
    return "\n".join(part.get("text", "") for part in parts if part.get("type") == "text").strip()


def paper_from_semantic_scholar(query: str, timeout: float) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {"query": query, "limit": 1, "fields": "title,authors,year,abstract,url,externalIds"}
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "KPRegressionBook/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    item = (data.get("data") or [{}])[0]
    authors = ", ".join(author.get("name", "") for author in item.get("authors", []) if author.get("name"))
    return {
        "title": item.get("title"),
        "authors": authors,
        "year": item.get("year"),
        "abstract": item.get("abstract"),
        "url": item.get("url"),
        "external_ids": item.get("externalIds") or {},
    }


def paper_from_crossref(query: str, timeout: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"query.bibliographic": query, "rows": 1})
    cfg = service_config("papers")
    mailto = str(cfg.get("mailto") or "").strip()
    if mailto:
        params = f"{params}&{urllib.parse.urlencode({'mailto': mailto})}"
    base_url = str(cfg.get("endpoint") or "https://api.crossref.org/works")
    url = f"{base_url}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": f"KPRegressionBook/1.0 ({mailto or 'mailto:none@example.com'})"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    item = (data.get("message", {}).get("items") or [{}])[0]
    authors = ", ".join(
        " ".join(filter(None, [author.get("given"), author.get("family")]))
        for author in item.get("author", [])
    )
    year_parts = item.get("published-print", item.get("published-online", {})).get("date-parts", [[None]])
    return {
        "title": (item.get("title") or [None])[0],
        "authors": authors,
        "year": year_parts[0][0],
        "abstract": item.get("abstract"),
        "url": item.get("URL"),
        "external_ids": {"DOI": item.get("DOI")},
    }


def collect_sync_payload() -> dict[str, Any]:
    return {
        "notes": [dict(row) for row in db_rows("SELECT * FROM notes")],
        "completed_sections": [dict(row) for row in db_rows("SELECT * FROM completed_sections")],
        "tutor_conversations": [dict(row) for row in db_rows("SELECT * FROM tutor_conversations")],
    }


def merge_sync_payload(payload: dict[str, Any]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        for row in payload.get("notes", []):
            if row.get("id"):
                conn.execute(
                    """
                    INSERT INTO notes(id, chapter_slug, section_id, title, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, COALESCE(?, datetime('now')), COALESCE(?, datetime('now')))
                    ON CONFLICT(id) DO UPDATE
                    SET title = excluded.title, body = excluded.body, updated_at = excluded.updated_at
                    """,
                    (
                        row["id"],
                        row["chapter_slug"],
                        row["section_id"],
                        row.get("title") or "Untitled note",
                        row.get("body", ""),
                        row.get("created_at"),
                        row.get("updated_at"),
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO notes(chapter_slug, section_id, title, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, COALESCE(?, datetime('now')), COALESCE(?, datetime('now')))
                    """,
                    (
                        row["chapter_slug"],
                        row["section_id"],
                        row.get("title") or "Untitled note",
                        row.get("body", ""),
                        row.get("created_at"),
                        row.get("updated_at"),
                    ),
                )
        for row in payload.get("completed_sections", []):
            conn.execute(
                """
                INSERT INTO completed_sections(chapter_slug, section_id, completed, updated_at)
                VALUES (?, ?, ?, COALESCE(?, datetime('now')))
                ON CONFLICT(chapter_slug, section_id) DO UPDATE
                SET completed = excluded.completed, updated_at = excluded.updated_at
                """,
                (row["chapter_slug"], row["section_id"], int(row.get("completed", 0)), row.get("updated_at")),
            )
        for row in payload.get("tutor_conversations", []):
            conn.execute(
                """
                INSERT INTO tutor_conversations(chapter_slug, section_id, role, body, created_at)
                VALUES (?, ?, ?, ?, COALESCE(?, datetime('now')))
                """,
                (row["chapter_slug"], row["section_id"], row["role"], row.get("body", ""), row.get("created_at")),
            )


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            # Browsers can cancel polling/fetch requests during navigation or
            # reload. That is a client disconnect, not a server-side failure.
            return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api_get(parsed)
        if parsed.path.startswith("/vendor/") or parsed.path.startswith("/site/static/"):
            return self.serve_project_file(parsed.path.lstrip("/"), include_body=True)
        if parsed.path == "/":
            self.path = "/chapter0.html"
        return super().do_GET()

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/vendor/") or parsed.path.startswith("/site/static/"):
            return self.serve_project_file(parsed.path.lstrip("/"), include_body=False)
        return super().do_HEAD()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api_post(parsed)
        json_response(self, {"error": "not found"}, 404)

    def serve_project_file(self, rel_path: str, include_body: bool) -> None:
        path = (ROOT / rel_path).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def handle_api_get(self, parsed: urllib.parse.ParseResult) -> None:
        params = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/api/services/status":
            json_response(self, probe_all_services())
            return
        if parsed.path == "/api/settings":
            json_response(self, public_settings())
            return
        if parsed.path == "/api/settings/config-files":
            json_response(self, config_file_views())
            return
        if parsed.path == "/api/notes":
            rows = db_rows(
                """
                SELECT id, chapter_slug, section_id, title, created_at, updated_at,
                       substr(replace(replace(body, '<', ' <'), '>', '> '), 1, 180) AS preview
                FROM notes
                WHERE chapter_slug = ? AND section_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (params.get("chapter_slug", [""])[0], params.get("section_id", [""])[0]),
            )
            json_response(self, {"notes": [dict(row) for row in rows]})
            return
        note_attachments_match = re.fullmatch(r"/api/notes/(\d+)/attachments", parsed.path)
        if note_attachments_match:
            note_id = note_attachments_match.group(1)
            if not db_rows("SELECT id FROM notes WHERE id = ?", (note_id,)):
                json_response(self, {"error": "note not found"}, 404)
                return
            json_response(self, {"attachments": attachment_rows(note_id)})
            return
        attachment_match = re.fullmatch(r"/api/attachments/(\d+)", parsed.path)
        if attachment_match:
            rows = db_rows(
                "SELECT id, original_name, stored_name, content_type, size_bytes FROM note_attachments WHERE id = ?",
                (attachment_match.group(1),),
            )
            if not rows:
                json_response(self, {"error": "attachment not found"}, 404)
                return
            row = rows[0]
            path = (ATTACHMENT_DIR / row["stored_name"]).resolve()
            try:
                path.relative_to(ATTACHMENT_DIR.resolve())
            except ValueError:
                json_response(self, {"error": "attachment not found"}, 404)
                return
            if not path.is_file():
                json_response(self, {"error": "attachment not found"}, 404)
                return
            body = path.read_bytes()
            filename = str(row["original_name"]).replace('"', "")
            self.send_response(200)
            self.send_header("Content-Type", row["content_type"] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'inline; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/api/notes/"):
            note_id = parsed.path.rsplit("/", 1)[-1]
            rows = db_rows(
                "SELECT id, chapter_slug, section_id, title, body, created_at, updated_at FROM notes WHERE id = ?",
                (note_id,),
            )
            if not rows:
                json_response(self, {"error": "note not found"}, 404)
                return
            json_response(self, {"note": dict(rows[0])})
            return
        if parsed.path == "/api/progress":
            row = db_rows(
                "SELECT completed FROM completed_sections WHERE chapter_slug = ? AND section_id = ?",
                (params.get("chapter_slug", [""])[0], params.get("section_id", [""])[0]),
            )
            json_response(self, {"completed": bool(row[0]["completed"]) if row else False})
            return
        if parsed.path == "/api/tutor/history":
            rows = db_rows(
                """
                SELECT role, body, created_at
                FROM tutor_conversations
                WHERE chapter_slug = ? AND section_id = ?
                ORDER BY id ASC
                """,
                (params.get("chapter_slug", [""])[0], params.get("section_id", [""])[0]),
            )
            json_response(self, {"messages": [dict(row) for row in rows]})
            return
        if parsed.path.startswith("/api/papers/"):
            key = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            query = params.get("query", [key.replace("_", " ")])[0]
            cached = db_rows("SELECT metadata_json FROM paper_cache WHERE citation_key = ?", (key,))
            if cached:
                json_response(self, {"cached": True, "metadata": json.loads(cached[0]["metadata_json"])})
                return
            if unavailable(self, "papers"):
                return
            timeout = float(service_config("papers").get("timeout_seconds") or 3)
            provider = str(service_config("papers").get("provider") or "semantic_scholar").lower()
            try:
                if provider == "crossref":
                    metadata = paper_from_crossref(query, timeout)
                else:
                    metadata = paper_from_semantic_scholar(query, timeout)
            except Exception:
                metadata = paper_from_semantic_scholar(query, timeout) if provider == "crossref" else paper_from_crossref(query, timeout)
            db_execute(
                """
                INSERT INTO paper_cache(citation_key, metadata_json, fetched_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(citation_key) DO UPDATE
                SET metadata_json = excluded.metadata_json, fetched_at = excluded.fetched_at
                """,
                (key, json.dumps(metadata)),
            )
            json_response(self, {"cached": False, "metadata": metadata})
            return
        json_response(self, {"error": "not found"}, 404)

    def handle_api_post(self, parsed: urllib.parse.ParseResult) -> None:
        if parsed.path == "/api/notes":
            data = read_json(self)
            title = str(data.get("title") or "").strip() or "Untitled note"
            body = str(data.get("body") or "")
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO notes(chapter_slug, section_id, title, body, created_at, updated_at)
                    VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                    """,
                    (data.get("chapter_slug", ""), data.get("section_id", ""), title, body),
                )
                note_id = cur.lastrowid
            json_response(self, {"ok": True, "id": note_id})
            return
        note_attachments_match = re.fullmatch(r"/api/notes/(\d+)/attachments", parsed.path)
        if note_attachments_match:
            note_id = note_attachments_match.group(1)
            if not db_rows("SELECT id FROM notes WHERE id = ?", (note_id,)):
                json_response(self, {"error": "note not found"}, 404)
                return
            try:
                _fields, files = parse_multipart(self)
            except ValueError as exc:
                if str(exc) == "payload_too_large":
                    json_response(self, {"error": "attachment too large", "max_bytes": MAX_ATTACHMENT_BYTES}, 413)
                else:
                    json_response(self, {"error": "invalid multipart upload"}, 400)
                return
            upload = next((item for item in files if item["field_name"] == "file"), files[0] if files else None)
            if not upload or not upload["content"]:
                json_response(self, {"error": "missing file"}, 400)
                return
            original_name = safe_filename(str(upload["filename"]))
            stored_name = f"{note_id}_{uuid.uuid4().hex}_{original_name}"
            content = upload["content"]
            if len(content) > MAX_ATTACHMENT_BYTES:
                json_response(self, {"error": "attachment too large", "max_bytes": MAX_ATTACHMENT_BYTES}, 413)
                return
            (ATTACHMENT_DIR / stored_name).write_bytes(content)
            with sqlite3.connect(DB_PATH) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO note_attachments(note_id, original_name, stored_name, content_type, size_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (note_id, original_name, stored_name, upload["content_type"], len(content)),
                )
                attachment_id = cur.lastrowid
                conn.execute("UPDATE notes SET updated_at = datetime('now') WHERE id = ?", (note_id,))
            rows = db_rows(
                """
                SELECT id, note_id, original_name, content_type, size_bytes, created_at
                FROM note_attachments
                WHERE id = ?
                """,
                (attachment_id,),
            )
            json_response(self, {"ok": True, "attachment": attachment_payload(rows[0])})
            return
        if parsed.path.startswith("/api/notes/"):
            note_id = parsed.path.rsplit("/", 1)[-1]
            data = read_json(self)
            title = str(data.get("title") or "").strip() or "Untitled note"
            body = str(data.get("body") or "")
            db_execute(
                """
                UPDATE notes
                SET title = ?, body = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (title, body, note_id),
            )
            json_response(self, {"ok": True})
            return
        if parsed.path == "/api/settings":
            data = read_json(self)
            json_response(self, update_settings(data))
            return
        if parsed.path == "/api/progress":
            data = read_json(self)
            db_execute(
                """
                INSERT INTO completed_sections(chapter_slug, section_id, completed, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(chapter_slug, section_id) DO UPDATE
                SET completed = excluded.completed, updated_at = excluded.updated_at
                """,
                (data.get("chapter_slug", ""), data.get("section_id", ""), 1 if data.get("completed") else 0),
            )
            json_response(self, {"ok": True})
            return
        if parsed.path == "/api/tutor/ask":
            if unavailable(self, "tutor"):
                return
            data = read_json(self)
            section = data.get("section_text", "")
            question = data.get("question", "")
            mode = data.get("mode", "ask")
            prompt = (
                f"Book section:\n{section}\n\n"
                f"Reader mode: {mode}\n"
                f"Reader question or reasoning:\n{question}\n\n"
                "Answer with section-grounded explanation. If critiquing reasoning, identify the strongest part, the mistake or gap, and a concrete correction."
            )
            db_execute(
                "INSERT INTO tutor_conversations(chapter_slug, section_id, role, body) VALUES (?, ?, 'user', ?)",
                (data.get("chapter_slug", ""), data.get("section_id", ""), question),
            )
            try:
                answer = call_llm("tutor", prompt)
            except Exception as exc:
                json_response(self, {"error": str(exc), "reason": "provider_error"}, 502)
                return
            db_execute(
                "INSERT INTO tutor_conversations(chapter_slug, section_id, role, body) VALUES (?, ?, 'assistant', ?)",
                (data.get("chapter_slug", ""), data.get("section_id", ""), answer),
            )
            text_stream_response(self, answer)
            return
        if parsed.path == "/api/exercises/grade":
            if unavailable(self, "grading"):
                return
            data = read_json(self)
            prompt = (
                f"Exercise or section text:\n{data.get('exercise_text', '')}\n\n"
                f"Reader submission:\n{data.get('submission', '')}\n\n"
                "Give specific, section-grounded feedback. Do not return a pass/fail score. Point out reasoning gaps and next revision steps."
            )
            try:
                feedback = call_llm("grading", prompt)
            except Exception as exc:
                json_response(self, {"error": str(exc), "reason": "provider_error"}, 502)
                return
            db_execute(
                """
                INSERT INTO exercise_submissions(chapter_slug, section_id, submission, feedback)
                VALUES (?, ?, ?, ?)
                """,
                (data.get("chapter_slug", ""), data.get("section_id", ""), data.get("submission", ""), feedback),
            )
            json_response(self, {"feedback": feedback})
            return
        if parsed.path == "/api/tts":
            if unavailable(self, "tts"):
                return
            data = read_json(self)
            text = str(data.get("section_text") or "")
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
            cache_path = AUDIO_CACHE / f"{data.get('chapter_slug', 'chapter')}_{data.get('section_id', 'section')}_{content_hash}.mp3"
            if not cache_path.exists():
                cfg = service_config("tts")
                try:
                    provider = str(cfg.get("provider") or "openai").lower()
                    if provider == "google_tts":
                        endpoint = str(cfg.get("endpoint") or "https://texttospeech.googleapis.com/v1/text:synthesize")
                        sep = "&" if "?" in endpoint else "?"
                        endpoint = f"{endpoint}{sep}key={urllib.parse.quote(str(cfg.get('api_key', '')))}"
                        payload = {
                            "input": {"text": text[:5000]},
                            "voice": {"languageCode": cfg.get("voice") or "en-US", "name": cfg.get("model") or ""},
                            "audioConfig": {"audioEncoding": "MP3"},
                        }
                        if not payload["voice"]["name"]:
                            del payload["voice"]["name"]
                        request = urllib.request.Request(
                            endpoint,
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json"},
                            method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=90) as response:
                            data = json.loads(response.read().decode("utf-8"))
                        cache_path.write_bytes(base64.b64decode(data["audioContent"]))
                    elif provider == "elevenlabs":
                        endpoint = str(cfg.get("endpoint") or "https://api.elevenlabs.io/v1/text-to-speech").rstrip("/")
                        voice = str(cfg.get("voice") or "").strip()
                        if voice and not endpoint.endswith(voice):
                            endpoint = f"{endpoint}/{urllib.parse.quote(voice)}"
                        payload = {"text": text[:5000], "model_id": cfg.get("model") or "eleven_multilingual_v2"}
                        request = urllib.request.Request(
                            endpoint,
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json", "xi-api-key": str(cfg.get("api_key", ""))},
                            method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=90) as response, cache_path.open("wb") as fh:
                            shutil.copyfileobj(response, fh)
                    else:
                        payload = {
                            "model": cfg.get("model") or "tts-1",
                            "voice": cfg.get("voice") or "alloy",
                            "input": text[:12000],
                        }
                        request = urllib.request.Request(
                            str(cfg["endpoint"]),
                            data=json.dumps(payload).encode("utf-8"),
                            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.get('api_key', '')}"},
                            method="POST",
                        )
                        with urllib.request.urlopen(request, timeout=90) as response, cache_path.open("wb") as fh:
                            shutil.copyfileobj(response, fh)
                except Exception as exc:
                    json_response(self, {"error": str(exc), "reason": "provider_error"}, 502)
                    return
            body = cache_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=31536000")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in {"/api/sync/push", "/api/sync/pull"}:
            if unavailable(self, "sync"):
                return
            cfg = service_config("sync")
            direction = parsed.path.rsplit("/", 1)[-1]
            endpoint = str(cfg["endpoint"]).rstrip("/") + f"/{direction}"
            payload = collect_sync_payload() if direction == "push" else {}
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.get('api_key', '')}"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    remote = json.loads(response.read().decode("utf-8") or "{}")
                if direction == "pull":
                    merge_sync_payload(remote)
                json_response(self, {"ok": True, "message": remote.get("message") or f"Sync {direction} complete."})
            except Exception as exc:
                json_response(self, {"error": str(exc), "reason": "provider_error"}, 502)
            return
        json_response(self, {"error": "not found"}, 404)


def main() -> None:
    init_db()
    host = str(CONFIG.get("server", {}).get("host") or "127.0.0.1")
    port = int(CONFIG.get("server", {}).get("port") or 9001)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving KP regression platform at http://{host}:{port}/")
    print("Static root:", STATIC_ROOT)
    present_configs = [str(path) for path in [CONFIG_YAML_PATH, CONFIG_JSON_PATH] if path.exists()]
    print("Config:", ", ".join(present_configs) if present_configs else "defaults + environment variables")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
