#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

from ohvm_runtime import (
    add_api_arguments, call_json, check_api, cli_entry, configure_api,
    paragraph_rows, run_batch,
)

ATTRIBUTES = [
    "goal",
    "principle",
    "tradeoff",
    "beneficiaries",
    "harmed_groups",
    "norm",
    "perspective",
]

DIMENSION_DEFINITIONS = """Dimension definitions:
- goal: the desired outcome or state of affairs that the argument supports.
- principle: the normative reason that makes the goal desirable, legitimate, or necessary.
- tradeoff: the competing value, cost, risk, or sacrifice involved in accepting the conclusion.
- beneficiaries: the people, groups, institutions, or entities expected to benefit.
- harmed_groups: the people, groups, institutions, or entities expected to be harmed, burdened, or excluded.
- norm: the behavioral expectation, duty, or social rule implied by the argument.
- perspective: the evaluative standpoint from which the argument judges the issue.
"""

PROMPT = """You are given one argument edge in context.

Task: describe the value expressed by the Premise-Conclusion-Stance edge using seven semantic dimensions. Each value must be exactly one concise English sentence.

{dimension_definitions}

Paragraph:
{paragraph}

Premise:
{premise}

Conclusion:
{conclusion}

Relation: {relation}
Stance: {stance}

Return JSON only:
{{
  "goal": "...",
  "principle": "...",
  "tradeoff": "...",
  "beneficiaries": "...",
  "harmed_groups": "...",
  "norm": "...",
  "perspective": "..."
}}

Rules:
- Ground every field in the argument and paragraph context.
- If a field is not specified, write "Not specified in the paragraph."
- Do not start with "This sentence" or "The unit".
- Output JSON only.
"""


def valid_attributes(obj: dict[str, Any]) -> dict[str, str]:
    out = {}
    for key in ATTRIBUTES:
        value = str(obj.get(key) or "").strip()
        if key == "goal" and not value:
            value = str(obj.get("final_goal") or "").strip()
        if key == "tradeoff" and not value:
            value = str(obj.get("trade_off") or "").strip()
        if not value:
            value = "Not specified in the paragraph."
        out[key] = value
    return out


def iter_edges(row: dict[str, Any]) -> list[dict[str, Any]]:
    paragraph = str(row.get("paragraph_text") or row.get("text") or "").strip()
    pack_id = str(row.get("pack_id") or row.get("id") or "")
    units = {str(u.get("unit_id")): str(u.get("text") or "") for u in row.get("units_pred") or row.get("units") or [] if isinstance(u, dict)}
    pairs = row["pairs_pred"] if "pairs_pred" in row else row.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError(f"pack_id={pack_id}: pairs_pred/pairs must be an array")
    edges = []
    for idx, edge in enumerate(pairs, 1):
        if not isinstance(edge, dict):
            raise ValueError(f"pack_id={pack_id}: edge {idx} must be an object")
        src = str(edge.get("src_unit_id") or edge.get("premise_unit_id") or "").strip()
        tgt = str(edge.get("tgt_unit_id") or edge.get("conclusion_unit_id") or "").strip()
        premise = str(edge.get("premise_text") or units.get(src) or "").strip()
        conclusion = str(edge.get("conclusion_text") or units.get(tgt) or "").strip()
        if not premise or not conclusion:
            raise ValueError(f"pack_id={pack_id}: edge {idx} requires premise and conclusion text")
        edges.append({
            "pack_id": pack_id,
            "edge_id": str(edge.get("edge_id") or f"e{idx}"),
            "src_unit_id": src,
            "tgt_unit_id": tgt,
            "premise_text": premise,
            "conclusion_text": conclusion,
            "relation": str(edge.get("relation") or "support").strip(),
            "stance": str(edge.get("stance") or "in favor of").strip(),
            "paragraph_text": paragraph,
        })
    return edges


def edge_tasks(path):
    for row in paragraph_rows(path):
        if "pairs_pred" not in row and "pairs" not in row:
            raise ValueError(f"pack_id={row['pack_id']}: missing pairs_pred or pairs; run extraction first")
        seen = set()
        for edge in iter_edges(row):
            if edge["edge_id"] in seen:
                raise ValueError(f"pack_id={row['pack_id']}: duplicate edge_id {edge['edge_id']}")
            seen.add(edge["edge_id"])
            yield edge


def build_prompt(edge: dict[str, Any]) -> str:
    return PROMPT.format(
        dimension_definitions=DIMENSION_DEFINITIONS,
        paragraph=edge.get("paragraph_text", ""),
        premise=edge.get("premise_text", ""),
        conclusion=edge.get("conclusion_text", ""),
        relation=edge.get("relation", ""),
        stance=edge.get("stance", ""),
    )


def decompose(edge: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    obj = call_json(build_prompt(edge), args.temperature, args.timeout, args.retries)
    out = dict(edge)
    out["attributes"] = valid_attributes(obj)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate value-semantic attributes for P-C-S edges.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--failed-jsonl", default=None)
    add_api_arguments(parser)
    parser.set_defaults(workers=8)
    args = parser.parse_args()

    output = Path(args.output_jsonl)
    failed = Path(args.failed_jsonl) if args.failed_jsonl else output.with_suffix(".failed.jsonl")
    if Path(args.input_jsonl).resolve() in {output.resolve(), failed.resolve(), output.with_suffix(".checkpoint.sqlite3").resolve()}:
        raise ValueError("Input and output paths must be different")
    context = configure_api(args)
    context.update(step="decomposition", prompt=PROMPT, definitions=DIMENSION_DEFINITIONS)
    return run_batch(lambda: edge_tasks(args.input_jsonl), lambda edge: decompose(edge, args),
                     output, failed, args.workers, context, args.resume, args.overwrite, check_api)


if __name__ == "__main__":
    raise SystemExit(cli_entry(main))
