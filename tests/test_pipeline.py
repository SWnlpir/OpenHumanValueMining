import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from ohvm_runtime import read_jsonl, write_json, write_jsonl
from pipeline import DEFAULT_CONFIG, load_config


def invoke(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "pipeline", *args], cwd=cwd,
                          text=True, capture_output=True, timeout=90)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "paragraphs.jsonl"
    write_jsonl(source, [{"text": f"test-{i}"} for i in range(8)])
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    rng = np.random.default_rng(7)
    matrix = np.vstack([np.array([1., 0.]) + rng.normal(0, .02, (4, 2)),
                        np.array([0., 1.]) + rng.normal(0, .02, (4, 2))])
    for dim in DEFAULT_CONFIG["dims"].split(","):
        np.save(embedding_dir / f"{dim}.npy", matrix)
    path = tmp_path / "config.json"
    write_json(path, {
        "input_jsonl": source.name, "output_dir": "outputs", "k_final": 2, "env_file": None,
        "api": {"workers": 2, "retries": 1},
        "clustering": {"embedding_dir": embedding_dir.name, "bootstrap": 0,
                       "seeds": "11,23", "k_base": "2", "knn": 7},
    })
    return path


def test_config_paths_resolve_against_config_file(prepared):
    config = load_config(prepared)
    assert config["input_jsonl"] == str(prepared.parent / "paragraphs.jsonl")
    assert config["clustering"]["embedding_dir"] == str(prepared.parent / "embeddings")


def test_dry_run_writes_nothing_and_does_not_require_api(prepared, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = invoke("run", "--config", str(prepared), "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "No API calls" in result.stdout
    assert not (prepared.parent / "outputs").exists()


def test_unknown_config_field_is_actionable(prepared):
    write_json(prepared, {"k_fnal": 2})
    result = invoke("run", "--config", str(prepared), "--dry-run")
    assert result.returncode == 2
    assert "Unknown configuration fields: k_fnal" in result.stderr
    assert "Traceback" not in result.stderr


def test_values_pipeline_runs_offline_from_another_directory(prepared, monkeypatch):
    config = json.loads(prepared.read_text())
    config["input_stage"] = "values"
    write_json(prepared, config)
    write_jsonl(prepared.parent / "paragraphs.jsonl", [
        {"pack_id": str(i), "edge_id": "e1", "premise_text": str(i), "conclusion_text": str(i + 1),
         "attributes": {dim: str(i) for dim in DEFAULT_CONFIG["dims"].split(",")}} for i in range(8)
    ])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    doctor = invoke("doctor", "--config", str(prepared), cwd=prepared.parent)
    assert doctor.returncode == 0, doctor.stdout + doctor.stderr
    result = invoke("run", "--config", str(prepared), cwd=prepared.parent)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((prepared.parent / "outputs/run.json").read_text())
    assert set(state["steps"]) == {"clustering", "evidence"}
    assert state["configuration"]["input_stage"] == "values"
    assert "scikit-learn" in state["versions"]


def test_full_pipeline_resume_with_local_mock_api(prepared, monkeypatch):
    counters = {"extraction": 0, "decomposition": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = payload["messages"][0]["content"]
            if "Split the paragraph" in prompt:
                obj = {"units": [{"unit_id": f"u{i}", "text": str(i)} for i in [1, 2]],
                       "conclusion_unit_ids": ["u2"]}
            elif "Select the root" in prompt:
                obj = {"root_unit_ids": ["u2"]}
            elif "For each target conclusion" in prompt:
                obj = {"expansions": [{"target_unit_id": "u2", "premises": [
                    {"unit_id": "u1", "relation": "support", "stance": "in favor of"}]}]}
            else:
                obj = {field: "test" for field in ["goal", "principle", "tradeoff", "beneficiaries",
                                                  "harmed_groups", "norm", "perspective"]}
            kind = "decomposition" if "Dimension definitions:" in prompt else "extraction"
            with lock:
                counters[kind] += 1
                fail = kind == "decomposition" and counters[kind] == 1
            response = {"error": {"message": "test failure", "type": "test"}} if fail else {
                "id": "test", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": json.dumps(obj)}}],
            }
            body = json.dumps(response).encode()
            self.send_response(500 if fail else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_MODEL", "local-test-model")
    output = prepared.parent / "outputs"
    try:
        failed = invoke("run", "--config", str(prepared))
        assert failed.returncode == 1, failed.stdout + failed.stderr
        assert counters == {"extraction": 24, "decomposition": 8}
        assert len(read_jsonl(output / "pairs_with_attributes.jsonl")) == 7
        assert not (output / "clustering").exists()
        resumed = invoke("run", "--config", str(prepared), "--resume")
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert counters == {"extraction": 24, "decomposition": 9}
        assert len(read_jsonl(output / "clustering/final_clusters.jsonl")) == 8
        assert (output / "cluster_evidence.csv").read_bytes().startswith(b"\xef\xbb\xbf")
        state = json.loads((output / "run.json").read_text())
        assert state["status"] == "completed"
        assert len(state["steps"]) == 4
        changed = json.loads(prepared.read_text())
        changed["api"].update(workers=1, timeout=60, retries=2)
        write_json(prepared, changed)
        again = invoke("run", "--config", str(prepared), "--resume")
        assert again.returncode == 0, again.stderr
        assert again.stdout.count("already complete; skipped") == 4
        assert counters == {"extraction": 24, "decomposition": 9}
        np.save(prepared.parent / "embeddings/goal.npy", np.ones((8, 2)))
        changed_vectors = invoke("run", "--config", str(prepared), "--resume")
        assert changed_vectors.returncode == 2
        assert "Input or settings changed" in changed_vectors.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("module", ["pipeline", "argument_tree_extraction", "value_decomposition",
                                   "consensus_clustering", "cluster_evidence_selection", "evaluation"])
def test_command_help(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
