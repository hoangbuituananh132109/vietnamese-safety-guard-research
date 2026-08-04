from translator.batching import length_bucket, make_batches


def row(uid, n, bucket=None):
    r = {"record_uid": uid, "prompt": "x" * n, "response": None}
    if bucket: r["length_bucket"] = bucket
    return r


def test_normal_batches_thirty_items():
    batches = make_batches([row(str(i), 10, "normal") for i in range(35)])
    assert [len(x) for x in batches] == [30, 5]


def test_bucket_change_closes_batch():
    batches = make_batches([row("a", 10, "normal"), row("b", 1800, "near_tail")])
    assert [len(x) for x in batches] == [1, 1]


def test_tail_and_oversized_routing():
    assert length_bucket(2343) == "tail"
    assert length_bucket(25_001) == "oversized"
    assert [len(x) for x in make_batches([row("a", 3000, "tail"), row("b", 3000, "tail")])] == [2]
    assert [len(x) for x in make_batches([row("a", 30_000, "oversized"), row("b", 30_000, "oversized")])] == [1, 1]


def test_no_loss_or_duplication():
    rows = [row(str(i), 100, "normal") for i in range(23)]
    flat = [r["record_uid"] for b in make_batches(rows) for r in b]
    assert flat == [r["record_uid"] for r in rows]
