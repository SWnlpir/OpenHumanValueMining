"""One-command orchestration for the released value-mining scripts."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ohvm_runtime import (
    cli_entry, configure_api, check_api, digest, file_digest, paragraph_rows,
    read_jsonl, write_json,
)

DEFAULT_CONFIG = {
    "input_jsonl": "paragraphs.jsonl",
    "input_stage": "paragraphs",
    "output_dir": "outputs",
    "k_final": 20,
    "text_field": "paragraph_text",
    "dims": "goal,principle,tradeoff,perspective",
    "env_file": ".env",
    "api": {
        "workers": 4, "retries": 3, "timeout": 120, "temperature": 0.0,
        "batch_size": 4,
    },
    "clustering": {
        "embedding_model": "BAAI/bge-m3", "embedding_dir": None,
        "batch_size": 64, "seeds": "11,23,37,51,67", "k_base": None,
        "bootstrap": 2, "bootstrap_frac": 0.8, "knn": 50, "min_observed": 2,
    },
    "evidence": {"top_k": 5},
}


def resolve_path(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path):
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{exc.lineno}: invalid configuration JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(data) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    config = {**DEFAULT_CONFIG, **data}
    sections = {"api": {"model", "base_url"}, "clustering": set(), "evidence": set()}
    for name, extra in sections.items():
        section = data.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"'{name}' must be an object")
        unknown = set(section) - (set(DEFAULT_CONFIG[name]) | extra)
        if unknown:
            raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
        config[name] = {**DEFAULT_CONFIG[name], **section}
    if config["input_stage"] not in {"paragraphs", "trees", "values"}:
        raise ValueError("input_stage must be paragraphs, trees, or values")
    if type(config["k_final"]) is not int or config["k_final"] < 2:
        raise ValueError("k_final must be an integer >= 2")
    for section, names in [("api", ["workers", "retries", "timeout", "batch_size"]),
                           ("clustering", ["batch_size", "knn", "min_observed"]),
                           ("evidence", ["top_k"])]:
        for name in names:
            if type(config[section][name]) is not int or config[section][name] < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
    cluster = config["clustering"]
    if not isinstance(config["api"]["temperature"], (int, float)):
        raise ValueError("api.temperature must be numeric")
    for name in ["model", "base_url"]:
        if name in config["api"] and (not isinstance(config["api"][name], str) or not config["api"][name].strip()):
            raise ValueError(f"api.{name} must be a non-empty string")
    for name in ["seeds", "k_base"]:
        value = cluster[name]
        if value is None and name == "k_base":
            continue
        try:
            numbers = [int(part.strip()) for part in value.split(",")]
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"clustering.{name} must be a comma-separated integer string") from exc
        if any(number < (0 if name == "seeds" else 2) for number in numbers):
            raise ValueError(f"Invalid clustering.{name} values")
    if type(cluster["bootstrap"]) is not int or cluster["bootstrap"] < 0:
        raise ValueError("clustering.bootstrap must be a non-negative integer")
    if not isinstance(cluster["bootstrap_frac"], (int, float)) or not 0 < cluster["bootstrap_frac"] <= 1:
        raise ValueError("clustering.bootstrap_frac must be in (0, 1]")
    dimensions = {"goal", "final_goal", "principle", "tradeoff", "trade_off", "beneficiaries", "harmed_groups", "norm", "perspective"}
    if not isinstance(config["dims"], str):
        raise ValueError("dims must be a comma-separated string")
    dims = [part.strip() for part in config["dims"].split(",")]
    if any(dim not in dimensions for dim in dims) or len(set(dims)) != len(dims):
        raise ValueError("dims must list unique value dimensions")
    for field in ["input_jsonl", "output_dir", "env_file"]:
        if field == "env_file" and config[field] is None:
            continue
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty path")
        config[field] = str(resolve_path(config[field], path.parent))
    if cluster["embedding_dir"]:
        if not isinstance(cluster["embedding_dir"], str):
            raise ValueError("clustering.embedding_dir must be a path string or null")
        cluster["embedding_dir"] = str(resolve_path(cluster["embedding_dir"], path.parent))
        for dim in dims:
            array = Path(cluster["embedding_dir"]) / f"{dim}.npy"
            if not array.is_file():
                raise ValueError(f"Precomputed embedding file not found: {array}")
    if not isinstance(config["text_field"], str) or not config["text_field"].strip():
        raise ValueError("text_field must be a non-empty string")
    return config


def options(values):
    return [item for key, value in values.items() if value is not None
            for item in ("--" + key.replace("_", "-"), str(value))]


def plan(config, resume=False, overwrite=False):
    output = Path(config["output_dir"])
    trees = output / "trees.jsonl"
    values = output / "pairs_with_attributes.jsonl"
    clustering = output / "clustering"
    clusters = clustering / "final_clusters.jsonl"
    embedding = clustering / "view_embedding.npy"
    api = {key: value for key, value in config["api"].items() if key != "batch_size"}
    api["env_file"] = config["env_file"]
    mode = ["--resume"] if resume else ["--overwrite"] if overwrite else []
    commands = []

    def add(name, module, arguments, artifacts, api_step=False):
        commands.append({"name": name, "command": [sys.executable, "-m", module] + arguments,
                         "outputs": [str(path) for path in artifacts], "api": api_step})

    if config["input_stage"] == "paragraphs":
        add("extraction", "argument_tree_extraction", options({
            "input_jsonl": config["input_jsonl"], "output_jsonl": trees,
            "text_field": config["text_field"], "batch_size": config["api"]["batch_size"], **api,
        }) + mode, [trees], True)
    if config["input_stage"] in {"paragraphs", "trees"}:
        source = trees if config["input_stage"] == "paragraphs" else config["input_jsonl"]
        add("decomposition", "value_decomposition", options({
            "input_jsonl": source, "output_jsonl": values, **api,
        }) + mode, [values], True)
    else:
        values = Path(config["input_jsonl"])
    add("clustering", "consensus_clustering", options({
        "input_jsonl": values, "output_dir": clustering, "k_final": config["k_final"],
        "dims": config["dims"], **config["clustering"],
    }), [clusters, embedding, clustering / "consensus_affinity.npz", clustering / "cluster_summary.json"])
    add("evidence", "cluster_evidence_selection", options({
        "clusters_jsonl": clusters, "pairs_jsonl": values, "embedding_npy": embedding,
        "output_csv": output / "cluster_evidence.csv", **config["evidence"],
    }), [output / "cluster_evidence.csv"])
    return commands


def validate_input(config):
    if config["input_stage"] == "paragraphs":
        count = sum(1 for _ in paragraph_rows(config["input_jsonl"], config["text_field"]))
    elif config["input_stage"] == "trees":
        from value_decomposition import edge_tasks
        count = sum(1 for _ in edge_tasks(config["input_jsonl"]))
    else:
        rows = read_jsonl(config["input_jsonl"])
        count = len(rows)
        dims = [part.strip() for part in config["dims"].split(",")]
        seen = set()
        for index, row in enumerate(rows, 1):
            identity = (row.get("pack_id"), row.get("edge_id"))
            if any(value is None for value in identity) or identity in seen:
                raise ValueError(f"Record {index}: missing or duplicate (pack_id, edge_id)")
            seen.add(identity)
            attrs = row.get("attributes")
            if not isinstance(attrs, dict) or any(not attrs.get(dim) and not attrs.get({"goal": "final_goal", "tradeoff": "trade_off"}.get(dim, dim)) for dim in dims):
                raise ValueError(f"Record {index}: missing selected value attributes")
        if not 2 <= config["k_final"] < count:
            raise ValueError(f"k_final must satisfy 2 <= K < number of edges ({count})")
    if not count:
        raise ValueError("Input contains no usable records")
    return count


def missing_dependencies(config):
    modules = {"numpy": "cluster", "scipy": "cluster", "sklearn": "cluster", "pandas": "cluster"}
    if config["input_stage"] != "values":
        modules["openai"] = "api"
        if config["env_file"]:
            modules["dotenv"] = "api"
    if not config["clustering"]["embedding_dir"]:
        modules["sentence_transformers"] = "embeddings"
    return [(module, extra) for module, extra in modules.items() if importlib.util.find_spec(module) is None]


def execute(config, resume=False, overwrite=False, dry_run=False):
    count = validate_input(config)
    commands = plan(config, resume, overwrite)
    print(f"Validated {count} input records/tasks", flush=True)
    for step in commands:
        print(f"[{step['name']}] {shlex.join(step['command'])}", flush=True)
    if dry_run:
        print("Dry run complete. No API calls, model downloads or output writes.")
        return 0
    missing = missing_dependencies(config)
    if missing:
        raise ValueError(f"Missing dependencies: {', '.join(name for name, _ in missing)}. Install with: pip install -e '.[all]'")
    if config["input_stage"] != "values":
        configure_api(argparse.Namespace(env_file=config["env_file"], model=config["api"].get("model"),
                      base_url=config["api"].get("base_url"), temperature=config["api"]["temperature"],
                      retries=config["api"]["retries"], timeout=config["api"]["timeout"]))
        check_api()
    output = Path(config["output_dir"])
    manifest = output / "run.json"
    if Path(config["input_jsonl"]).is_relative_to(output):
        raise ValueError("Keep input data outside output_dir to avoid overwriting inputs")
    semantic_config = {**config, "api": {key: value for key, value in config["api"].items()
                                         if key not in {"workers", "timeout", "retries"}}}
    modules = {step["command"][2] for step in commands} | {"pipeline", "ohvm_runtime"}
    code = {name: file_digest(Path(__file__).with_name(name + ".py")) for name in modules}
    api_used = config["input_stage"] != "values"
    context = {"config": semantic_config, "input_sha256": file_digest(config["input_jsonl"]),
               "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14") if api_used else None,
               "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") if api_used else None, "code": code}
    embedding_dir = config["clustering"]["embedding_dir"]
    if embedding_dir:
        context["embeddings"] = {dim.strip(): file_digest(Path(embedding_dir) / f"{dim.strip()}.npy")
                                 for dim in config["dims"].split(",")}
    signature = digest(context)
    versions = {}
    for name in ["open-human-value-mining", "openai", "numpy", "scipy", "scikit-learn", "pandas", "sentence-transformers"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    state = {"signature": signature, "status": "running", "steps": {}, "configuration": config,
             "input_sha256": context["input_sha256"], "code_sha256": code,
             "api_model": context["model"], "python": sys.version.split()[0], "versions": versions}
    if manifest.exists() and resume:
        state = json.loads(manifest.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("Input or settings changed; choose a new output_dir or use --overwrite")
    elif not overwrite and output.exists() and any(output.iterdir()):
        raise ValueError("output_dir is not empty; use --resume, --overwrite, or a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    state["status"] = "running"
    write_json(manifest, state)
    try:
        upstream_changed = False
        for step in commands:
            previous = state["steps"].get(step["name"], {})
            if resume and not upstream_changed and previous.get("outputs") and all(
                Path(path).is_file() and file_digest(path) == checksum
                for path, checksum in previous["outputs"].items()
            ):
                print(f"[{step['name']}] already complete; skipped", flush=True)
                continue
            upstream_changed = True
            # Once a step reruns, downstream stages must also be recomputed.
            remaining = [item["name"] for item in commands[commands.index(step):]]
            for name in remaining:
                state["steps"].pop(name, None)
            write_json(manifest, state)
            subprocess.run(step["command"], check=True)
            state["steps"][step["name"]] = {
                "outputs": {path: file_digest(path) for path in step["outputs"]},
            }
            write_json(manifest, state)
    except (subprocess.CalledProcessError, KeyboardInterrupt) as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        write_json(manifest, state)
        if isinstance(exc, KeyboardInterrupt):
            raise
        print("Pipeline stopped. Inspect the failure log, then rerun with --resume.", file=sys.stderr)
        return exc.returncode or 1
    state["status"] = "completed"
    write_json(manifest, state)
    print(f"Complete. Evidence: {output / 'cluster_evidence.csv'}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Open Human Value Mining pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create a pipeline configuration")
    init.add_argument("--config", default="ohvm.json")
    for name in ["doctor", "run"]:
        command = sub.add_parser(name, help="Check environment" if name == "doctor" else "Run the pipeline")
        command.add_argument("--config", default="ohvm.json")
        command.add_argument("--env-file", help="Override the configured dotenv path")
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            mode = command.add_mutually_exclusive_group()
            mode.add_argument("--resume", action="store_true")
            mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        path = Path(args.config)
        if path.exists():
            raise ValueError(f"Configuration already exists: {path}")
        write_json(path, DEFAULT_CONFIG)
        print(f"Created {path}. Set input_jsonl and k_final, and configure credentials in .env.")
        return 0
    config = load_config(args.config)
    if args.env_file:
        config["env_file"] = str(Path(args.env_file).expanduser().resolve())
    if args.command == "doctor":
        missing = missing_dependencies(config)
        print(f"Python: {sys.version.split()[0]}")
        print(f"Input: {config['input_jsonl']}")
        print(f"Output: {config['output_dir']}")
        for module, extra in missing:
            print(f"MISSING: {module}; install extra [{extra}]")
        count = validate_input(config)
        print(f"Input check: OK ({count} records/tasks)")
        if config["input_stage"] != "values" and not missing:
            configure_api(argparse.Namespace(env_file=config["env_file"], model=config["api"].get("model"),
                          base_url=config["api"].get("base_url"), temperature=config["api"]["temperature"],
                          retries=config["api"]["retries"], timeout=config["api"]["timeout"]))
            check_api()
            print("API credentials: configured (not contacted)")
        print("No API requests or model downloads were made.")
        return 1 if missing else 0
    return execute(config, args.resume, args.overwrite, args.dry_run)


def entrypoint():
    raise SystemExit(cli_entry(main))


if __name__ == "__main__":
    entrypoint()
