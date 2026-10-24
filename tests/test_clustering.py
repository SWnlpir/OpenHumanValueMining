import json
import subprocess
import sys
import types

import numpy as np
import pytest

from consensus_clustering import build_embeddings
from evaluation import extract_edges
from ohvm_runtime import write_jsonl


def test_embedding_cache_tracks_input_and_model(tmp_path, monkeypatch):
    calls = []

    class Encoder:
        def __init__(self, name):
            calls.append(name)

        def encode(self, texts, **kwargs):
            return np.array([[len(text), 1.] for text in texts], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Encoder))
    rows = [{"attributes": {"goal": str(i)}} for i in range(3)]
    first = build_embeddings(rows, ["goal"], "first", tmp_path, 2)
    second = build_embeddings(rows, ["goal"], "first", tmp_path, 2)
    assert np.array_equal(first["goal"], second["goal"])
    assert calls == ["first"]
    rows[0]["attributes"]["goal"] = "changed"
    changed = build_embeddings(rows, ["goal"], "first", tmp_path, 2)
    assert not np.array_equal(first["goal"], changed["goal"])
    build_embeddings(rows, ["goal"], "other", tmp_path, 2)
    assert calls == ["first", "first", "other"]


def test_evaluation_keeps_empty_predictions_separate_from_gold():
    row = {"pack_id": "p", "pairs_pred": [], "pairs": [{"premise_text": "a", "conclusion_text": "b"}]}
    assert extract_edges([row], "pairs_pred") == []
    assert len(extract_edges([row], "pairs")) == 1


@pytest.mark.parametrize("k", [1, 3, 5])
def test_invalid_cluster_count_fails_before_model_loading(tmp_path, k):
    source = tmp_path / "values.jsonl"
    write_jsonl(source, [{"pack_id": str(i), "edge_id": "e1"} for i in range(3)])
    result = subprocess.run([sys.executable, "-m", "consensus_clustering", "--input-jsonl", str(source),
                             "--output-dir", str(tmp_path / "out"), "--k-final", str(k)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 2
    assert "2 <= K" in result.stderr
    assert "Traceback" not in result.stderr


def test_evaluation_saves_machine_readable_metrics(tmp_path):
    source, vectors, target = tmp_path / "clusters.jsonl", tmp_path / "view.npy", tmp_path / "metrics.json"
    write_jsonl(source, [{"cluster_id": i // 2} for i in range(4)])
    np.save(vectors, np.array([[1., .1], [1., .2], [.1, 1.], [.2, 1.]]))
    result = subprocess.run([sys.executable, "-m", "evaluation", "cluster-summary", "--clusters-jsonl",
                             str(source), "--embedding-npy", str(vectors), "--output-json", str(target)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(target.read_text())["n_samples"] == 4
