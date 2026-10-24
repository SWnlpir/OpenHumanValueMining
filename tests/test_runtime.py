import argparse
import json
import threading
import time

import pytest

import ohvm_runtime
from ohvm_runtime import paragraph_rows, positive_int, read_jsonl, run_batch, write_jsonl
from value_decomposition import edge_tasks, iter_edges


def test_jsonl_errors_include_line_number(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('\ufeff{}\n\n{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"input.jsonl:3: invalid JSON"):
        read_jsonl(path)
    path.write_text('[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        read_jsonl(path)


def test_generated_ids_and_duplicate_rejection(tmp_path):
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [{"text": str(i)} for i in range(3)])
    assert [row["pack_id"] for row in paragraph_rows(path)] == ["row-1", "row-2", "row-3"]
    write_jsonl(path, [{"id": 0, "text": "0"}, {"id": "0", "text": "1"}])
    with pytest.raises(ValueError, match="duplicate pack_id"):
        list(paragraph_rows(path))


def test_checkpoint_retries_only_failures_and_preserves_order(tmp_path):
    output = tmp_path / "results.jsonl"
    failed = tmp_path / "failed.jsonl"
    tasks = [{"id": i} for i in range(6)]

    def first(row):
        time.sleep((6 - row["id"]) * 0.001)
        if row["id"] == 2:
            raise RuntimeError("temporary failure")
        return row

    assert run_batch(lambda: iter(tasks), first, output, failed, 2, {"model": "test"}) == 1
    assert [row["id"] for row in read_jsonl(output)] == [0, 1, 3, 4, 5]
    assert len(read_jsonl(failed)) == 1
    called = []

    def retry(row):
        called.append(row["id"])
        return row

    assert run_batch(lambda: iter(tasks), retry, output, failed, 2, {"model": "test"}, resume=True) == 0
    assert called == [2]
    assert read_jsonl(output) == tasks
    assert read_jsonl(failed) == []
    # Fully completed steps can export the saved results without credentials.
    assert run_batch(lambda: iter(tasks), retry, output, failed, 2, {"model": "test"}, resume=True,
                     preflight=lambda: pytest.fail("preflight should be skipped")) == 0


def test_checkpoint_rejects_changed_input_or_settings(tmp_path):
    output, failed = tmp_path / "out.jsonl", tmp_path / "failed.jsonl"
    tasks = [{"id": 1}]
    run_batch(lambda: iter(tasks), dict, output, failed, 1, {"model": "first"})
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        run_batch(lambda: iter(tasks), dict, output, failed, 1, {"model": "first"})
    with pytest.raises(ValueError, match="settings changed"):
        run_batch(lambda: iter(tasks), dict, output, failed, 1, {"model": "other"}, resume=True)
    tasks.append({"id": 2})
    with pytest.raises(ValueError, match="settings changed"):
        run_batch(lambda: iter(tasks), dict, output, failed, 1, {"model": "first"}, resume=True)
    assert output.read_bytes() == original


def test_empty_predictions_do_not_fall_back_to_gold():
    row = {"pack_id": "p", "pairs_pred": [], "pairs": [{"premise_text": "a", "conclusion_text": "b"}]}
    assert iter_edges(row) == []


def test_tree_inputs_require_structure_and_unique_edges(tmp_path):
    path = tmp_path / "trees.jsonl"
    write_jsonl(path, [{"text": "0"}])
    with pytest.raises(ValueError, match="missing pairs_pred"):
        list(edge_tasks(path))
    edge = {"edge_id": "e1", "premise_text": "0", "conclusion_text": "1"}
    write_jsonl(path, [{"text": "0", "pairs_pred": [edge, edge]}])
    with pytest.raises(ValueError, match="duplicate edge_id"):
        list(edge_tasks(path))


def test_positive_parameters():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_interrupt_keeps_finished_inflight_tasks(tmp_path, monkeypatch):
    output, failed = tmp_path / "out.jsonl", tmp_path / "failed.jsonl"
    tasks = [{"id": i} for i in range(6)]
    real_wait = ohvm_runtime.wait
    started = threading.Barrier(3)

    def interrupt(*args, **kwargs):
        # Let active workers start, then interrupt the orchestrating thread.
        started.wait(timeout=5)
        raise KeyboardInterrupt

    def worker(row):
        started.wait(timeout=5)
        time.sleep(.05)
        return row

    monkeypatch.setattr(ohvm_runtime, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_batch(lambda: iter(tasks), worker, output, failed, 2, {})
    saved = read_jsonl(output)
    assert len(saved) == 2
    monkeypatch.setattr(ohvm_runtime, "wait", real_wait)
    called = []

    def resume_worker(row):
        called.append(row["id"])
        return row

    assert run_batch(lambda: iter(tasks), resume_worker, output, failed, 2, {}, resume=True) == 0
    assert sorted(called) == [2, 3, 4, 5]
    assert read_jsonl(output) == tasks
