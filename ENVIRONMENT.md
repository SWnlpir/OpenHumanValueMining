# Environment

Use Python >= 3.9 in a virtual environment. Python >= 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

On Windows, activate with `.venv\Scripts\activate`.
`pip install -r requirements.txt` remains supported from the repository root.

## Optional Dependencies

The base package installs the CLI and standard-library validation/orchestration
tools. Choose dependencies for the stages you need:

| Installation | Use |
| --- | --- |
| `pip install -e ".[api]"` | Argument extraction and value decomposition. |
| `pip install -e ".[cluster]"` | Clustering/evidence and numerical evaluation with existing embeddings. |
| `pip install -e ".[cluster,embeddings]"` | Local embedding generation and clustering. |
| `pip install -e ".[all]"` | Complete pipeline. |
| `pip install -e ".[api,cluster,dev]"` | Offline development tests and package builds. |

Run `ohvm init`, edit `ohvm.json`, and use `ohvm doctor` to check the
dependencies, input format, and API credential configuration for that run.
`ohvm run --dry-run` validates inputs and prints commands without importing
the embedding model or making network requests.

## API Configuration

Copy `.env.example` to `.env` and configure your OpenAI-compatible
chat-completions service. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_MODEL`; select a model available through your provider.

Environment variables take precedence over values in `.env`. Explicit
`api.model` and `api.base_url` settings in `ohvm.json` override both.
Standalone API scripts load dotenv files only when `--env-file` is supplied.
Neither the doctor command nor tests send real API requests.

## Embedding Models

The default local embedding model is `BAAI/bge-m3`. The first run downloads it
through Sentence Transformers. Set `SENTENCE_TRANSFORMERS_HOME` to choose the
model cache location, or set `clustering.embedding_model` to an existing local
model directory. For constrained hardware, choose a smaller compatible model
and lower `clustering.batch_size`.

Precomputed embeddings avoid this dependency: set `clustering.embedding_dir`
to a directory containing `<dimension>.npy` for each selected dimension.
Each matrix must be two-dimensional, finite, and contain one row per input
edge in exactly the same order. Dimension matrices may have different widths.
Automatically generated caches also contain a manifest with the input digest,
selected dimensions, and model. Keep it with the arrays.

## Troubleshooting

| Message | Action |
| --- | --- |
| Missing dependencies | Install the extra named by `ohvm doctor` in the active environment. |
| Set OPENAI_API_KEY | Configure `.env` or the environment; check the configured `env_file` path. |
| Outputs already exist | Use `--resume`, choose a new output path, or explicitly use `--overwrite`. |
| Input or settings changed | Start a new output directory; a checkpoint belongs to one input/model configuration. |
| HTTP 401/403 | Check the API credential and model access. |
| HTTP 429 | Reduce `api.workers` and resume after the provider's rate limit clears. |
| No tasks found | Check `input_stage`, paragraph fields, and extracted edge arrays. |
| K is outside the valid range | Choose `2 <= k_final < number of value edges`. |

The released package contains no private credentials, local data paths, or
experimental datasets.
