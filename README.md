# Open Human Value Mining

This repository contains the code release for the Open Human Value Mining
pipeline. The implementation focuses on the main algorithmic components used
for argument-tree extraction, value-semantic decomposition, consensus
clustering, evaluation, and cluster evidence selection.

Repository: https://github.com/SWnlpir/OpenHumanValueMining

## Components

- `argument_tree_extraction.py`: tree-constrained extraction of paragraph-level Premise-Conclusion-Stance edges.
- `value_decomposition.py`: value-semantic decomposition into seven dimensions.
- `consensus_clustering.py`: sparse consensus clustering over compositional value-semantic embeddings.
- `evaluation.py`: automatic evaluation for edge extraction, cluster summary, and external alignment.
- `cluster_evidence_selection.py`: selection of center, high-confidence, diverse, and boundary evidence for cluster naming.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9 or newer is supported. 

API-based scripts require an OpenAI-compatible chat-completions endpoint:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4.1-mini-2025-04-14"
```

## Full Pipeline Commands

```bash
python argument_tree_extraction.py \
  --input-jsonl path/to/paragraphs.jsonl \
  --output-jsonl outputs/trees.jsonl

python value_decomposition.py \
  --input-jsonl outputs/trees.jsonl \
  --output-jsonl outputs/pairs_with_attributes.jsonl

python consensus_clustering.py \
  --input-jsonl outputs/pairs_with_attributes.jsonl \
  --output-dir outputs/clustering \
  --embedding-dir path/to/embeddings \
  --k-final 20
```

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

## Notes

Access to the full dataset requires permission from its creator; please visit
the dataset project page to obtain it.
