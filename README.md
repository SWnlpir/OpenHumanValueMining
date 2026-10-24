# Open Human Value Mining

This repository contains the code release for the Open Human Value Mining
pipeline: argument-tree extraction, value-semantic decomposition, sparse
consensus clustering, evaluation, and cluster evidence selection.

Repository: https://github.com/SWnlpir/OpenHumanValueMining

## Quick Start

Python 3.9 or newer is supported; Python 3.11 or newer is recommended.

```bash
git clone https://github.com/SWnlpir/OpenHumanValueMining.git
cd OpenHumanValueMining
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
cp .env.example .env
ohvm init
```

On Windows, activate the environment with `.venv\Scripts\activate`.

1. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in `.env`.
2. In `ohvm.json`, set `input_jsonl` to your own input file and choose `k_final`.
3. Check the setup, preview the commands, then run:

```bash
ohvm doctor
ohvm run --dry-run
ohvm run
```

The pipeline connects extraction, decomposition, embedding generation,
clustering, and evidence export automatically. The first embedding run downloads
the configured Sentence Transformers model. API extraction and decomposition
use your configured chat-completions service.

`python -m pipeline` can replace `ohvm` in every command.

## Input Format

Use UTF-8 JSONL: one JSON object per line. Blank lines and UTF-8 BOM are accepted.
Provide your own data; no sample dataset is distributed with this release.

| Field | Requirement |
| --- | --- |
| `paragraph_text` | Non-empty paragraph text; `text` is also accepted. |
| `pack_id` | Unique paragraph ID. Falls back to `id`, then a generated `row-N` ID. |

Set `text_field` in the configuration when your text uses a different field.
Duplicate IDs and invalid records are reported before API processing starts.

To start from an intermediate result, set `input_stage` and point `input_jsonl`
at the corresponding file:

| `input_stage` | Expected input | Steps executed |
| --- | --- | --- |
| `paragraphs` (default) | Paragraph objects | Extract, decompose, cluster, export evidence |
| `trees` | Paragraphs with `pairs_pred` or `pairs` arrays | Decompose, cluster, export evidence |
| `values` | Edge objects with `pack_id`, `edge_id`, and `attributes` | Cluster, export evidence |

Tree edges need `premise_text` and `conclusion_text`, or unit IDs that reference
the paragraph's `units_pred`/`units` array. Value rows need all selected dimensions
in their `attributes` object. `(pack_id, edge_id)` must uniquely identify each edge.

## Configuration

`ohvm init --config path/to/ohvm.json` creates an editable configuration. All paths
inside it are resolved relative to the configuration file, including `.env`.
Unknown option names are rejected so spelling errors do not silently use defaults.

| Setting | Purpose |
| --- | --- |
| `k_final` | Final cluster count; must be at least 2 and smaller than the number of value edges. |
| `api.workers` | Concurrent API tasks; default 4. |
| `api.retries`, `api.timeout` | Request attempts and timeout in seconds. |
| `api.model`, `api.base_url` | Optional overrides for the corresponding environment variables. |
| `clustering.embedding_model` | Sentence Transformers model name or local model directory. |
| `clustering.embedding_dir` | Optional directory of precomputed dimension `.npy` files. |
| `evidence.top_k` | Evidence rows selected per cluster and selection method. |

Keep credentials in the environment or the ignored `.env` file. Existing
environment variables take precedence over `.env`; explicit configuration
overrides take precedence over both. For API-free clustering, use
`input_stage: "values"`. To avoid model downloads as well, provide
`clustering.embedding_dir` with one matrix per selected dimension, in input-row
order. See [ENVIRONMENT.md](ENVIRONMENT.md) for installation options.

## Outputs And Resuming

All generated files are placed under `output_dir` (default `outputs/`).

| File | Contents |
| --- | --- |
| `trees.jsonl` | Extracted units, argument edges, roots, and paragraph IDs. |
| `pairs_with_attributes.jsonl` | One argument edge and its value attributes per row. |
| `clustering/final_clusters.jsonl` | Cluster labels and confidence scores. |
| `clustering/cluster_summary.json` | Cluster sizes and cosine silhouette summary. |
| `clustering/view_embedding.npy` | Embeddings aligned with cluster rows. |
| `clustering/consensus_affinity.npz` | Sparse consensus graph. |
| `cluster_evidence.csv` | Center, high-confidence, diverse, and boundary evidence; Excel-compatible UTF-8. |
| `run.json` | Run status, configuration, dependency versions, input/source fingerprints, and completed output checksums. |

API steps save every finished task to a local SQLite checkpoint. Successful and
failed records are exported separately in input order. A task failure stops the
pipeline before clustering incomplete results. Fix the reported problem, then run:

```bash
ohvm run --resume
```

Successful API tasks are reused, failed tasks are retried, and completed pipeline
stages with matching output checksums are skipped. Keep the `.checkpoint.sqlite3`
files alongside the outputs. To intentionally rerun all steps, use
`ohvm run --overwrite`, or choose a new `output_dir`. Changed inputs, models,
precomputed embeddings, result-affecting settings, or pipeline source files require
a new run. Concurrency, timeout, and retry settings can be adjusted when resuming.

Generated embedding caches include an input/model manifest. Supplied `.npy`
files without a manifest are accepted after shape and finite-value checks; ensure
their row order matches your input.

## Individual Scripts

The original script interfaces remain available:

```bash
python argument_tree_extraction.py --input-jsonl path/to/paragraphs.jsonl --output-jsonl outputs/trees.jsonl --env-file .env
python value_decomposition.py --input-jsonl outputs/trees.jsonl --output-jsonl outputs/pairs_with_attributes.jsonl --env-file .env
python consensus_clustering.py --input-jsonl outputs/pairs_with_attributes.jsonl --output-dir outputs/clustering --k-final 20
python cluster_evidence_selection.py --clusters-jsonl outputs/clustering/final_clusters.jsonl --pairs-jsonl outputs/pairs_with_attributes.jsonl --embedding-npy outputs/clustering/view_embedding.npy --output-csv outputs/cluster_evidence.csv
python evaluation.py cluster-summary --clusters-jsonl outputs/clustering/final_clusters.jsonl --embedding-npy outputs/clustering/view_embedding.npy
```

Both API scripts accept `--resume`, `--overwrite`, `--workers`, `--model`, and
`--base-url`. Every script supports `--help`. Evaluation also provides `edge`
and `external-alignment` subcommands.
Pass `--output-json path/to/metrics.json` to an evaluation subcommand to save
its metrics as well as print them.

## Value Decomposition Schema

Each P-C-S edge is decomposed into seven fields:

- `goal`: desired outcome or state of affairs.
- `principle`: normative reason that makes the goal desirable or legitimate.
- `tradeoff`: competing value, cost, risk, or sacrifice.
- `beneficiaries`: entities expected to benefit.
- `harmed_groups`: entities expected to be harmed, burdened, or excluded.
- `norm`: behavioral expectation, duty, or social rule.
- `perspective`: evaluative standpoint used to judge the issue.

The main clustering view used in the paper is `goal,principle,tradeoff,perspective`.
For more than 5,000 edges, the clustering summary estimates cosine silhouette
on a reproducible sample of 5,000 edges.

## Development

```bash
python -m pip install -e ".[api,cluster,dev]"
python -m pytest -q
python -m build
```

Tests generate temporary inputs and use a local mock API. They do not call a paid
service or download embedding models. GitHub Actions runs the suite on Python
3.9, 3.11, and 3.13 and checks that the package builds.

## Data Access

Access to the full dataset requires permission from its creator; please visit
the dataset project page to obtain it.
