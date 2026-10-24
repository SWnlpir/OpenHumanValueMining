"""Shared input validation, API configuration and durable batch execution."""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def read_jsonl(path):
    return list(iter_jsonl(path))


def _write_text(path, chunks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete write, including when exporting checkpoints.
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(chunk)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path, rows):
    _write_text(path, (json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def write_json(path, obj):
    _write_text(path, [json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + "\n"])


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def paragraph_rows(path, text_field="paragraph_text"):
    seen = set()
    for index, source in enumerate(iter_jsonl(path), 1):
        row = dict(source)
        text = row.get(text_field) or row.get("paragraph_text") or row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: record {index}: missing non-empty text field '{text_field}'")
        pack_id = row.get("pack_id")
        if pack_id is None or str(pack_id).strip() == "":
            pack_id = row.get("id")
        if pack_id is None or str(pack_id).strip() == "":
            pack_id = f"row-{index}"
        row["pack_id"] = str(pack_id)
        if row["pack_id"] in seen:
            raise ValueError(f"{path}: record {index}: duplicate pack_id '{row['pack_id']}'")
        seen.add(row["pack_id"])
        row["paragraph_text"] = text.strip()
        yield row


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def add_api_arguments(parser):
    parser.add_argument("--env-file", help="Load credentials from this dotenv file; existing environment wins.")
    parser.add_argument("--model", help="Override OPENAI_MODEL.")
    parser.add_argument("--base-url", help="Override OPENAI_BASE_URL.")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--retries", type=positive_int, default=3)
    parser.add_argument("--timeout", type=positive_int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse successes and retry failures from the checkpoint.")
    mode.add_argument("--overwrite", action="store_true", help="Discard this step's previous checkpoint and outputs.")


def configure_api(args):
    if args.env_file:
        if not Path(args.env_file).is_file():
            raise ValueError(f"Environment file does not exist: {args.env_file}")
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ValueError('Install API dependencies with: pip install -e ".[api]"') from exc
        load_dotenv(args.env_file, override=False)
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    return {
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "temperature": args.temperature,
    }


def check_api():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in your environment or pass --env-file .env")
    if not os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14").strip():
        raise ValueError("OPENAI_MODEL must not be empty")
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ValueError('Install API dependencies with: pip install -e ".[api]"') from exc


_local = threading.local()


def call_json(prompt, temperature, timeout, retries):
    from openai import APIConnectionError, APIStatusError, OpenAI

    check_api()
    identity = (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    if getattr(_local, "identity", None) != identity:
        _local.client = OpenAI(api_key=identity[0], base_url=identity[1], max_retries=0)
        _local.identity = identity
    for attempt in range(retries):
        try:
            response = _local.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14"),
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            start = content.find("{")
            if start < 0:
                raise ValueError("Response did not contain a JSON object")
            obj, _ = json.JSONDecoder().raw_decode(content[start:])
            if not isinstance(obj, dict):
                raise ValueError("Expected a JSON object")
            return obj
        except APIStatusError as exc:
            # Do not persist response bodies, which may contain provider details.
            error = f"API returned HTTP {exc.status_code}"
            if exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                raise RuntimeError(error) from None
        except APIConnectionError:
            error = "API connection failed or timed out"
        except (ValueError, IndexError):
            error = "API returned invalid or empty JSON"
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"{error} after {retries} attempts")


def run_batch(task_factory, worker, output, failed, workers, context, resume=False, overwrite=False, preflight=None):
    """Bound pending requests, checkpoint each result, and export in input order."""
    output, failed = Path(output), Path(failed)
    checkpoint = output.with_suffix(".checkpoint.sqlite3")
    paths = [output.resolve(), failed.resolve(), checkpoint.resolve()]
    if len(set(paths)) != len(paths):
        raise ValueError("Output, failure log and checkpoint must use different paths")
    fingerprint = hashlib.sha256(digest(context).encode())
    total = 0
    for task in task_factory():
        fingerprint.update(digest(task).encode())
        total += 1
    if not total:
        raise ValueError("No tasks found; check the input format and extracted edges")
    fingerprint = fingerprint.hexdigest()
    if resume and not checkpoint.exists():
        if output.exists() or failed.exists():
            raise ValueError("Cannot resume existing outputs without their checkpoint; use a new output path")
    elif not resume and not overwrite and any(p.exists() for p in [output, failed, checkpoint]):
        raise ValueError("Outputs already exist. Use --resume, --overwrite, or a new output path")
    if (overwrite or not checkpoint.exists()) and preflight:
        preflight()
    output.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in [output, failed, checkpoint]:
            path.unlink(missing_ok=True)
    with sqlite3.connect(checkpoint) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (fingerprint TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS results (idx INTEGER PRIMARY KEY, success INTEGER, payload TEXT)")
        old = db.execute("SELECT fingerprint FROM metadata").fetchone()
        if old and old[0] != fingerprint:
            raise ValueError("Input or API settings changed since the checkpoint; use a new output path or --overwrite")
        if not old:
            db.execute("INSERT INTO metadata VALUES (?)", (fingerprint,))
            db.commit()
        done = {row[0] for row in db.execute("SELECT idx FROM results WHERE success=1")}
        if len(done) < total and preflight:
            preflight()
        tasks = ((index, task) for index, task in enumerate(task_factory()) if index not in done)
        completed = len(done)
        print(f"Tasks: {total}; resumed: {completed}; pending: {total - completed}", flush=True)
        pool = ThreadPoolExecutor(max_workers=workers)
        pending = {}

        def save_result(future, index, task):
            try:
                payload = future.result()
                success = 1
            except Exception as exc:
                payload = {"source": task, "error": str(exc)}
                success = 0
            db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                       (index, success, json.dumps(payload, ensure_ascii=False)))
            db.commit()

        try:
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < workers * 2:
                    item = next(tasks, None)
                    if item is None:
                        exhausted = True
                        break
                    index, task = item
                    pending[pool.submit(worker, task)] = (index, task)
                if not pending:
                    break
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    index, task = pending.pop(future)
                    save_result(future, index, task)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        print(f"Completed {completed}/{total}", flush=True)
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            # Requests already running may finish during an interrupt; retain them too.
            for future, (index, task) in pending.items():
                if future.done() and not future.cancelled():
                    save_result(future, index, task)
            for success, path in [(1, output), (0, failed)]:
                write_jsonl(path, (json.loads(row[0]) for row in db.execute(
                    "SELECT payload FROM results WHERE success=? ORDER BY idx", (success,))))
        failures = db.execute("SELECT COUNT(*) FROM results WHERE success=0").fetchone()[0]
    print(f"Saved: {output}; failures: {failures} ({failed})", flush=True)
    return 1 if failures else 0


def cli_entry(main):
    try:
        result = main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed API tasks are saved; rerun with --resume.", file=sys.stderr)
        return 130
    return result or 0
