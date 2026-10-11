"""A polite, auditable HTTP cache.

Three properties matter more than speed here:

1. Re-running the pipeline must not re-hit upstream. A judge who runs
   `python -m setu validate` twice should get identical numbers, and a demo on
   a conference wifi connection should not fail because the API rate-limited.
2. Every response carries provenance - URL, fetch time, status - because the
   evidence pack has to be able to say where a number came from.
3. It must degrade honestly. Offline means "serve the cache or raise", never
   "silently synthesise something plausible".
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from ..errors import UpstreamUnavailable

# cache.py sits at <repo>/src/setu/io/cache.py, so the repository root is three
# parents up, not two. Getting this wrong writes the cache inside the package.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

# Identified honestly: if an operator ever needs to ask us to back off, the
# User-Agent says who to email and the contact is in the README.
USER_AGENT = "SETU-air-prototype/0.1 (independent research prototype; +https://github.com/bhogesararam23/SETU)"

#: Set SETU_OFFLINE=1 to forbid all network access, e.g. on a demo machine.
OFFLINE_ENV = "SETU_OFFLINE"

#: Point the cache somewhere else without touching code. CI uses this to read
#: the committed snapshot in data/fixtures/upstream, so the whole pipeline runs
#: with no network and no keys.
CACHE_DIR_ENV = "SETU_CACHE_DIR"


def default_cache_dir() -> Path:
    override = os.environ.get(CACHE_DIR_ENV, "").strip()
    return Path(override) if override else DEFAULT_CACHE_DIR


@dataclasses.dataclass(frozen=True)
class CachedResponse:
    url: str
    fetched_at: float
    from_cache: bool
    age_seconds: float
    payload: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "fetched_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.fetched_at)
            ),
            "from_cache": self.from_cache,
            "age_seconds": round(self.age_seconds, 1),
        }


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def offline_only() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {"1", "true", "yes"}


class HttpCache:
    def __init__(self, directory: Path | str | None = None,
                 session: requests.Session | None = None):
        self.dir = Path(directory) if directory is not None else default_cache_dir()
        self._session = session

    # -- plumbing ---------------------------------------------------------
    @property
    def session(self) -> requests.Session:
        if self._session is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": USER_AGENT})
            self._session = sess
        return self._session

    def _paths(self, url: str) -> tuple[Path, Path]:
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = self.dir / _key(url)
        return stem.with_suffix(".json"), stem.with_suffix(".meta.json")

    def _write(self, url: str, payload: Any, fetched_at: float) -> None:
        body, meta = self._paths(url)
        # Write-then-rename so an interrupted run cannot leave a half file that
        # the next run mistakes for a good cached response.
        tmp = body.with_name(body.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(body)
        tmp_meta = meta.with_name(meta.name + ".tmp")
        tmp_meta.write_text(
            json.dumps({"url": url, "fetched_at": fetched_at}), encoding="utf-8"
        )
        tmp_meta.replace(meta)

    def _read(self, url: str) -> tuple[Any, float] | None:
        body, meta = self._paths(url)
        if not body.exists() or not meta.exists():
            return None
        try:
            payload = json.loads(body.read_text(encoding="utf-8"))
            fetched_at = float(json.loads(meta.read_text(encoding="utf-8"))["fetched_at"])
        except (ValueError, OSError, KeyError):
            return None
        return payload, fetched_at

    # -- public API -------------------------------------------------------
    def get_json(self, url: str, ttl_seconds: float = 6 * 3600,
                 timeout: float = 30.0, retries: int = 2) -> CachedResponse:
        """Return parsed JSON for `url`, from cache when fresh enough.

        Raises UpstreamUnavailable when the network fails and there is no cache
        to fall back to. A cached copy older than ttl is still served with a
        warning flag rather than being treated as an error, because for a
        demonstration a stale-but-real number beats a fresh-but-absent one.
        """
        cached = self._read(url)
        now = time.time()

        if cached is not None:
            payload, fetched_at = cached
            age = now - fetched_at
            if age <= ttl_seconds or offline_only():
                return CachedResponse(url, fetched_at, True, age, payload)

        if offline_only():
            raise UpstreamUnavailable(
                "cache", f"{url} is older than its TTL and {OFFLINE_ENV} is set"
            )

        last_error = "not attempted"
        for attempt in range(retries + 1):
            try:
                resp = self.session.get(url, timeout=timeout)
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.8 * (attempt + 1))
                continue

            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(0.8 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise UpstreamUnavailable(url.split("/")[2], f"HTTP {resp.status_code} for {url}")

            try:
                payload = resp.json()
            except ValueError as exc:
                raise UpstreamUnavailable(
                    url.split("/")[2], f"non-JSON response: {exc}"
                ) from exc

            fetched_at = time.time()
            self._write(url, payload, fetched_at)
            return CachedResponse(url, fetched_at, False, 0.0, payload)

        if cached is not None:
            payload, fetched_at = cached
            return CachedResponse(url, fetched_at, True, now - fetched_at, payload)

        raise UpstreamUnavailable(url.split("/")[2], f"unreachable ({last_error})")

    def stats(self) -> dict[str, Any]:
        files = list(self.dir.glob("*.json")) if self.dir.exists() else []
        bodies = [f for f in files if f.suffix == ".json"
                  and not f.name.endswith(".meta.json")]
        return {
            "directory": str(self.dir),
            "entries": len(bodies),
            "bytes": sum(f.stat().st_size for f in bodies) if bodies else 0,
            "offline_only": offline_only(),
        }
