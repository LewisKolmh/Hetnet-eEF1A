"""Shared helper for cached, rate-limited ChEMBL / UniProt / Reactome-KEGG API calls.

All fetch scripts in this package route their HTTP GETs through
`cached_get_json`, which persists raw JSON responses to a cache file so
interrupted runs resume without re-fetching anything already retrieved.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_START = 0.5
BACKOFF_MAX = 30.0


class JsonCache:
    """Simple on-disk key -> JSON-response cache, flushed after every write."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with open(self.path, "r") as f:
                try:
                    self._data: dict[str, Any] = json.load(f)
                except json.JSONDecodeError:
                    log.warning("Cache file %s corrupt, starting fresh", self.path)
                    self._data = {}
        else:
            self._data = {}

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f)
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)


def cached_get_json(
    url: str,
    cache: JsonCache,
    cache_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any | None:
    """GET a URL as JSON, using `cache` to skip already-fetched keys.

    Returns None (and caches None) on a request that fails after retries,
    so callers can distinguish "fetched, empty" from "not yet attempted".
    """
    key = cache_key or url
    if key in cache:
        return cache.get(key)

    delay = BACKOFF_START
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 429:
                wait = min(delay, BACKOFF_MAX)
                log.warning("429 rate-limited on %s, waiting %.1fs", url, wait)
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            cache.set(key, data)
            return data
        except requests.RequestException as exc:
            last_exc = exc
            wait = min(delay, BACKOFF_MAX)
            log.warning(
                "Request failed (attempt %d/%d) for %s: %s - retrying in %.1fs",
                attempt, MAX_RETRIES, url, exc, wait,
            )
            time.sleep(wait)
            delay *= 2

    log.error("Giving up on %s after %d attempts: %s", url, MAX_RETRIES, last_exc)
    cache.set(key, None)
    return None
