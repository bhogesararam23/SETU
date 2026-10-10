"""The cache is what makes a demo reproducible; it has to be tested properly."""

import json
import time

import pytest
import requests

from setu.errors import UpstreamUnavailable
from setu.io import cache as cache_mod
from setu.io.cache import HttpCache


class FakeSession:
    """Stands in for requests.Session and counts real attempts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None):
        self.calls += 1
        item = self.responses.pop(0) if self.responses else {"ok": True}
        if item.get("raise"):
            raise requests.ConnectionError("boom")
        return FakeResponse(item.get("status", 200), item.get("body", {"v": self.calls}))


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def cache(tmp_path):
    c = HttpCache(directory=tmp_path / "cache")
    c._session = FakeSession([])
    return c


def test_first_fetch_hits_the_network_and_the_second_does_not(cache):
    cache._session.responses = [{"body": {"a": 1}}]
    first = cache.get_json("https://example.test/x", ttl_seconds=60)
    assert first.from_cache is False
    assert first.payload == {"a": 1}

    second = cache.get_json("https://example.test/x", ttl_seconds=60)
    assert second.from_cache is True
    assert second.payload == {"a": 1}
    assert cache._session.calls == 1


def test_a_stale_entry_is_refetched(cache):
    cache._session.responses = [{"body": {"v": 1}}, {"body": {"v": 2}}]
    cache.get_json("https://example.test/y", ttl_seconds=0)
    time.sleep(0.01)
    fresh = cache.get_json("https://example.test/y", ttl_seconds=0)
    assert fresh.from_cache is False
    assert fresh.payload == {"v": 2}


def test_network_failure_falls_back_to_a_stale_copy(cache):
    cache._session.responses = [{"body": {"v": 1}}, {"raise": True}, {"raise": True},
                                {"raise": True}]
    cache.get_json("https://example.test/z", ttl_seconds=0)
    recovered = cache.get_json("https://example.test/z", ttl_seconds=0, retries=2)
    assert recovered.from_cache is True
    assert recovered.age_seconds > 0


def test_no_cache_and_no_network_raises_rather_than_inventing(cache):
    cache._session.responses = [{"raise": True}] * 3
    with pytest.raises(UpstreamUnavailable):
        cache.get_json("https://example.test/missing", retries=2)


def test_a_non_5xx_client_error_is_not_retried_into_oblivion(cache):
    cache._session.responses = [{"status": 404}]
    with pytest.raises(UpstreamUnavailable):
        cache.get_json("https://example.test/gone")
    assert cache._session.calls == 1


def test_a_200_with_a_body_that_is_not_json_is_an_error_not_a_none(cache):
    cache._session.responses = [{"body": "<html>portal down</html>"}]
    with pytest.raises(UpstreamUnavailable, match="non-JSON"):
        cache.get_json("https://example.test/html")


def test_offline_refuses_to_go_to_the_network(cache, monkeypatch):
    monkeypatch.setenv(cache_mod.OFFLINE_ENV, "1")
    with pytest.raises(UpstreamUnavailable):
        cache.get_json("https://example.test/never-cached")
    assert cache._session.calls == 0


def test_offline_serves_a_cached_copy_even_if_expired(cache, monkeypatch):
    monkeypatch.delenv(cache_mod.OFFLINE_ENV, raising=False)
    cache._session.responses = [{"body": {"v": 7}}]
    cache.get_json("https://example.test/w", ttl_seconds=0)
    monkeypatch.setenv(cache_mod.OFFLINE_ENV, "1")
    served = cache.get_json("https://example.test/w", ttl_seconds=0)
    assert served.payload == {"v": 7}
    assert served.from_cache is True


def test_an_interrupted_write_leaves_no_readable_entry(tmp_path):
    """A half-written body must never be mistaken for a good cached response."""
    c = HttpCache(directory=tmp_path / "c")
    c._session = FakeSession([])
    body, meta = c._paths("https://example.test/half")
    body.write_text('{"truncated": ', encoding="utf-8")  # no meta, bad JSON
    assert c._read("https://example.test/half") is None


def test_provenance_records_the_url_and_fetch_time(cache):
    cache._session.responses = [{"body": {"v": 1}}]
    got = cache.get_json("https://example.test/p")
    prov = got.as_dict()
    assert prov["url"] == "https://example.test/p"
    assert prov["fetched_at_iso"].endswith("Z")
    assert prov["from_cache"] is False


def test_stats_counts_entries_without_counting_metadata(cache):
    cache._session.responses = [{"body": {"v": 1}}, {"body": {"v": 2}}]
    cache.get_json("https://example.test/one")
    cache.get_json("https://example.test/two")
    st = cache.stats()
    assert st["entries"] == 2
    assert st["bytes"] > 0


def test_default_cache_dir_lands_in_the_repo_not_inside_the_package():
    """This drifted into src/data once, which gitignored nothing and confused
    everyone. Pin the location."""
    resolved = cache_mod.DEFAULT_CACHE_DIR.resolve()
    assert resolved.parent.name == "data"
    assert resolved.name == "cache"
    assert "src" not in resolved.parts


def test_user_agent_identifies_the_project():
    sess = requests.Session()
    sess.headers.update({"User-Agent": cache_mod.USER_AGENT})
    assert "SETU" in sess.headers["User-Agent"]
