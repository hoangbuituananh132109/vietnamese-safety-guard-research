import json

from translator.jsonl_io import read_jsonl, write_jsonl


def test_unicode_line_separators_do_not_split_records(tmp_path):
    path = tmp_path / "x.jsonl"
    rows = [{"text": "a\u2028b\u2029c\u0085d"}, {"text": "ok"}]
    write_jsonl(path, rows)
    assert len(list(read_jsonl(path))) == 2
    physical = path.read_text(encoding="utf-8").split("\n")
    assert "\\u2028" in physical[0] and "\\u2029" in physical[0] and "\\u0085" in physical[0]


def test_nan_is_normalized_on_read(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"x":NaN}\n', encoding="utf-8")
    assert list(read_jsonl(path))[0][1]["x"] is None

