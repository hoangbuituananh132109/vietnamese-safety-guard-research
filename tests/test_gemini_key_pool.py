import pytest

from translator.providers.gemini import ApiKeyPool


def test_api_key_pool_selects_and_rotates_original_slots(tmp_path):
    path = tmp_path / "API.txt"
    path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    pool = ApiKeyPool(path, [2, 4])
    assert pool.next() == (2, "two")
    assert pool.next() == (4, "four")
    assert pool.next() == (2, "two")
    pool.disable(2)
    assert pool.next() == (4, "four")
    assert pool.active_size == 1


def test_api_key_pool_rejects_invalid_slots(tmp_path):
    path = tmp_path / "API.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ApiKeyPool(path, [1, 3])


def test_reserve_key_is_dormant_until_activated(tmp_path):
    path = tmp_path / "API.txt"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    pool = ApiKeyPool(path, [1, 2], [3])
    assert [pool.next()[0] for _ in range(4)] == [1, 2, 1, 2]
    assert pool.reserve_active is False
    assert pool.activate_reserve(prefer_next=True) is True
    assert pool.next()[0] == 3
    assert pool.reserve_active is True
