import pytest

from translator.checkpoint import load_checkpoint


def test_conflicting_checkpoint_duplicate_rejected(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"record_uid":"u","x":1}\n{"record_uid":"u","x":2}\n', encoding="utf-8")
    with pytest.raises(ValueError): load_checkpoint(p)

