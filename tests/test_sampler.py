import json
from pathlib import Path

from translator.jsonl_io import write_jsonl
from translator.sampler import prepare, source_ids


def test_source_ids_accepts_pilot_source_id_or_original_id(tmp_path: Path):
    pilot = tmp_path / "pilot.jsonl"
    write_jsonl(pilot, [{"source_id": "a"}, {"id": "b"}, {"source_id": ""}])
    assert source_ids([pilot]) == {"a", "b"}


def test_prepare_excludes_existing_source_ids(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"id": "old", "prompt": "old prompt", "response": None, "tag": "generic"},
        {"id": "new", "prompt": "new prompt", "response": None, "tag": "generic"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepared = prepare(source, {"old"})
    assert [row["source_id"] for row in prepared] == ["new"]
