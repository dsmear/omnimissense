import numpy as np

from omnimissense.cache import ScoreCache, default_cache_dir


def test_wt_roundtrip_across_several_files(tmp_path) -> None:
    cache = ScoreCache(tmp_path, "facebook/esm2_t33_650M_UR50D")
    a = np.arange(60, dtype=np.float32).reshape(3, 20)
    b = -np.arange(40, dtype=np.float32).reshape(2, 20)
    cache.put_wt({"hash-a": (a, 1)})
    cache.put_wt({"hash-b": (b, 2)})

    assert cache.wt_hashes() == {"hash-a", "hash-b"}
    found = cache.get_wt(["hash-a", "hash-b", "missing"])
    assert set(found) == {"hash-a", "hash-b"}
    np.testing.assert_array_equal(found["hash-a"][0], a)
    np.testing.assert_array_equal(found["hash-b"][0], b)
    assert found["hash-b"][1] == 2
    assert "facebook--esm2_t33_650M_UR50D" in str(cache.directory)


def test_masked_roundtrip_returns_only_requested_positions(tmp_path) -> None:
    cache = ScoreCache(tmp_path, "m")
    v1 = np.linspace(-5, 0, 20, dtype=np.float32)
    v2 = np.linspace(0, 1, 20, dtype=np.float32)
    cache.put_masked({("h", 3): v1, ("h", 7): v2})
    found = cache.get_masked([("h", 3), ("h", 9)])
    assert list(found) == [("h", 3)]
    np.testing.assert_array_equal(found[("h", 3)], v1)


def test_empty_cache(tmp_path) -> None:
    cache = ScoreCache(tmp_path, "m")
    assert cache.wt_hashes() == set()
    assert cache.get_wt(["x"]) == {}
    assert cache.get_masked([("x", 1)]) == {}


def test_default_cache_dir_respects_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OMNIMISSENSE_CACHE", str(tmp_path))
    assert default_cache_dir() == tmp_path


def test_lookup_reads_only_candidate_row_groups(tmp_path, monkeypatch) -> None:
    """Many proteins in one file: a lookup must not read every row group."""
    import pyarrow.parquet as pq

    cache = ScoreCache(tmp_path, "m")
    scores = {f"{i:064x}": (np.full((2, 20), i, dtype=np.float32), 1) for i in range(1000)}
    cache.put_wt(scores)
    read_groups = []
    original = pq.ParquetFile.read_row_groups

    def spy(self, groups, *args, **kwargs):
        read_groups.extend(groups)
        return original(self, groups, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_groups", spy)
    wanted = f"{500:064x}"
    found = cache.get_wt([wanted])
    assert found[wanted][0][0, 0] == 500
    assert len(read_groups) == 1  # 1000 rows / 128 per group = 8 groups in the file
