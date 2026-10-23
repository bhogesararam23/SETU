"""The cache is what makes offline runs reproducible; it has to be tested properly."""

import json
import shutil
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
def cache(tmp_path, monkeypatch):
    """Fetch-path tests drive a fake session, so they must not be pinned
    offline even when the run points SETU_CACHE_DIR at the committed
    snapshot (which conftest pins to SETU_OFFLINE=1 so nothing rewrites it)."""
    monkeypatch.delenv(cache_mod.OFFLINE_ENV, raising=False)
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


def test_cache_dir_env_moves_the_whole_cache_without_a_code_change(tmp_path,
                                                                   monkeypatch):
    """CI reads the committed snapshot through this. If the override were ignored,
    CI would try to reach the network it is explicitly not allowed to reach."""
    elsewhere = tmp_path / "snapshot"
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(elsewhere))
    assert cache_mod.default_cache_dir() == elsewhere
    assert cache_mod.HttpCache().dir == elsewhere


def test_explicit_directory_beats_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, str(tmp_path / "from_env"))
    chosen = tmp_path / "chosen"
    assert cache_mod.HttpCache(directory=chosen).dir == chosen


def test_no_env_means_the_working_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(cache_mod.CACHE_DIR_ENV, "   ")
    assert cache_mod.default_cache_dir() == cache_mod.DEFAULT_CACHE_DIR


def test_the_committed_snapshot_replays_offline(tmp_path, monkeypatch):
    """CI and the README's reproduction command both read
    data/fixtures/upstream with SETU_OFFLINE=1. This replays a real request
    through that path, so if the snapshot stops resolving, the test fails here
    instead of failing during CI or local development.

    The point is not that a file exists - it is that an *expired* entry is still
    served when egress is forbidden, which is the only reason a checked-in
    snapshot is usable at all. The snapshot is copied into a temp dir and aged
    there; the committed provenance record is never touched.
    """
    source = cache_mod.DEFAULT_CACHE_DIR.parents[0] / "fixtures" / "upstream"
    metas = sorted(source.glob("*.meta.json"))
    if not metas:
        pytest.skip("snapshot not present in this checkout")

    meta = json.loads(metas[0].read_text(encoding="utf-8"))
    key = metas[0].name[: -len(".meta.json")]        # <hash>.meta.json -> <hash>
    shutil.copy(metas[0].parent / f"{key}.json", tmp_path / f"{key}.json")
    aged = dict(meta, fetched_at=time.time() - 400 * 24 * 3600)
    (tmp_path / metas[0].name).write_text(json.dumps(aged), encoding="utf-8")

    monkeypatch.setenv(cache_mod.OFFLINE_ENV, "1")
    cache = HttpCache(directory=tmp_path)

    # Nine months old, so nothing but offline mode can serve it.
    got = cache.get_json(meta["url"], ttl_seconds=6 * 3600)

    assert got.from_cache is True
    assert got.url == meta["url"]
    assert got.payload.get("hourly"), "snapshot entry carries no hourly block"
    assert got.age_seconds > 300 * 24 * 3600


def test_offline_mode_still_refuses_a_url_that_was_never_fetched(monkeypatch,
                                                                 tmp_path):
    """A snapshot is not a substitute for data that was never collected. Serving
    a fabricated payload here would be the worst possible failure mode."""
    monkeypatch.setenv(cache_mod.OFFLINE_ENV, "1")
    cache = HttpCache(directory=tmp_path / "empty")
    with pytest.raises(UpstreamUnavailable):
        cache.get_json("https://example.test/never-fetched", ttl_seconds=60)
