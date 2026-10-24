#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ohvm_runtime import cli_entry, positive_int, read_jsonl


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def key(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("pack_id") is None or row.get("edge_id") is None:
        raise ValueError("Every pair and cluster row requires pack_id and edge_id")
    return str(row["pack_id"]), str(row["edge_id"])


def build_view_embedding(embedding_dir: Path, dims: list[str]) -> np.ndarray:
    mats = [np.load(embedding_dir / f"{dim}.npy") for dim in dims]
    return l2_normalize(np.concatenate(mats, axis=1).astype(np.float32))


def value_text(row: dict[str, Any]) -> str:
    attrs = row.get("attributes") or {}
    goal = attrs.get("goal") or attrs.get("final_goal")
    tradeoff = attrs.get("tradeoff") or attrs.get("trade_off")
    parts = [
        ("Premise", row.get("premise_text")),
        ("Conclusion", row.get("conclusion_text")),
        ("Stance", row.get("stance")),
        ("Goal", goal),
        ("Principle", attrs.get("principle")),
        ("Tradeoff", tradeoff),
        ("Perspective", attrs.get("perspective")),
    ]
    return " | ".join(f"{name}: {str(value).strip()}" for name, value in parts if str(value or "").strip())


def center_rank(vecs: np.ndarray, idx: np.ndarray) -> list[int]:
    centroid = l2_normalize(vecs[idx].mean(axis=0, keepdims=True))[0]
    scores = vecs[idx] @ centroid
    return [int(idx[i]) for i in np.argsort(-scores)]


def high_conf_rank(idx: np.ndarray, conf_intra: np.ndarray, conf_ratio: np.ndarray) -> list[int]:
    scores = conf_intra[idx] + 0.25 * conf_ratio[idx]
    return [int(idx[i]) for i in np.argsort(-scores)]


def diverse_rank(vecs: np.ndarray, idx: np.ndarray, limit: int) -> list[int]:
    if len(idx) == 0:
        return []
    sub = vecs[idx]
    centroid = l2_normalize(sub.mean(axis=0, keepdims=True))[0]
    selected = [int(np.argmax(sub @ centroid))]
    remaining = set(range(len(idx))) - set(selected)
    while remaining and len(selected) < limit:
        rem = np.array(sorted(remaining), dtype=np.int32)
        dist = 1.0 - sub[rem] @ sub[np.array(selected)].T
        pick = int(rem[np.argmax(dist.min(axis=1))])
        selected.append(pick)
        remaining.remove(pick)
    return [int(idx[i]) for i in selected]


def boundary_rank(vecs: np.ndarray, idx: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> list[int]:
    cid = int(labels[idx[0]])
    sub = vecs[idx]
    own = centroids[cid]
    own_dist = 1.0 - sub @ own
    other_ids = [i for i in range(centroids.shape[0]) if i != cid]
    if not other_ids:
        return [int(i) for i in idx]
    other_dist = 1.0 - sub @ centroids[other_ids].T
    margin = other_dist.min(axis=1) - own_dist
    return [int(idx[i]) for i in np.argsort(margin)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative evidence for cluster naming.")
    parser.add_argument("--clusters-jsonl", required=True)
    parser.add_argument("--pairs-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--embedding-npy", default=None)
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--dims", default="goal,principle,tradeoff,perspective")
    parser.add_argument("--top-k", type=positive_int, default=5)
    args = parser.parse_args()

    cluster_rows = read_jsonl(Path(args.clusters_jsonl))
    pair_rows = read_jsonl(Path(args.pairs_jsonl))
    pair_map = {key(row): row for row in pair_rows}
    ordered_keys = [key(row) for row in cluster_rows]
    if not ordered_keys:
        raise ValueError("No cluster rows found")
    if len(pair_map) != len(pair_rows) or len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("Duplicate (pack_id, edge_id) keys in pairs or clusters")
    missing = [k for k in ordered_keys if k not in pair_map]
    if missing:
        raise ValueError(f"Missing pair rows for {len(missing)} cluster entries; first={missing[0]}")

    if args.embedding_npy:
        vecs = l2_normalize(np.load(args.embedding_npy).astype(np.float32))
    else:
        if not args.embedding_dir:
            raise ValueError("Either --embedding-npy or --embedding-dir must be provided.")
        dims = [x.strip() for x in args.dims.split(",") if x.strip()]
        vecs = build_view_embedding(Path(args.embedding_dir), dims)
    if vecs.ndim != 2 or vecs.shape[0] != len(cluster_rows) or not np.isfinite(vecs).all():
        raise ValueError("Embeddings must be a finite 2-D matrix aligned with clusters-jsonl rows")

    labels = np.array([int(row["cluster_id"]) for row in cluster_rows], dtype=np.int32)
    if set(labels) != set(range(int(labels.max()) + 1)):
        raise ValueError("Cluster IDs must be consecutive non-negative integers starting at 0")
    conf_intra = np.array([float(row.get("conf_intra", 0.0)) for row in cluster_rows], dtype=np.float32)
    conf_ratio = np.array([float(row.get("conf_ratio", 0.0)) for row in cluster_rows], dtype=np.float32)
    n_clusters = int(labels.max()) + 1

    centroids = np.zeros((n_clusters, vecs.shape[1]), dtype=np.float32)
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        centroids[cid] = l2_normalize(vecs[idx].mean(axis=0, keepdims=True))[0]

    rows = []
    reserve = max(args.top_k * 4, 20)
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        groups = {
            "center": center_rank(vecs, idx)[: args.top_k],
            "high_confidence": high_conf_rank(idx, conf_intra, conf_ratio)[: args.top_k],
            "diverse": diverse_rank(vecs, idx, reserve)[: args.top_k],
            "boundary": boundary_rank(vecs, idx, labels, centroids)[: args.top_k],
        }
        for evidence_type, selected in groups.items():
            for rank, row_idx in enumerate(selected, 1):
                pair = pair_map[ordered_keys[row_idx]]
                rows.append({
                    "cluster_id": cid,
                    "evidence_type": evidence_type,
                    "rank": rank,
                    "pack_id": pair.get("pack_id"),
                    "edge_id": pair.get("edge_id"),
                    "premise_text": pair.get("premise_text"),
                    "conclusion_text": pair.get("conclusion_text"),
                    "stance": pair.get("stance"),
                    "value_text": value_text(pair),
                })

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Evidence saved to {output}", flush=True)


if __name__ == "__main__":
    raise SystemExit(cli_entry(main))
