#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from ohvm_runtime import cli_entry, digest, positive_int, read_jsonl, write_json, write_jsonl

DIM_FIELDS = {
    "goal": "goal",
    "final_goal": "final_goal",
    "principle": "principle",
    "tradeoff": "tradeoff",
    "trade_off": "trade_off",
    "beneficiaries": "beneficiaries",
    "harmed_groups": "harmed_groups",
    "norm": "norm",
    "perspective": "perspective",
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def parse_list(value: str, cast=str) -> list[Any]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def dimension_text(row: dict[str, Any], dim: str) -> str:
    attrs = row.get("attributes") or {}
    field = DIM_FIELDS.get(dim, dim)
    value = str(attrs.get(field) or "").strip()
    if not value and dim == "goal":
        value = str(attrs.get("final_goal") or "").strip()
    if not value and dim == "tradeoff":
        value = str(attrs.get("trade_off") or "").strip()
    return value


def build_embeddings(rows: list[dict[str, Any]], dims: list[str], model_name: str, out_dir: Path, batch_size: int) -> dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = embedding_signature(rows, dims, model_name)
    manifest = out_dir / "manifest.json"
    reusable = manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) == expected
    if reusable and all((out_dir / f"{dim}.npy").is_file() for dim in dims):
        return load_embeddings(out_dir, dims)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = {}
    # Remove the completion marker before replacing any arrays.
    manifest.unlink(missing_ok=True)
    for dim in dims:
        path = out_dir / f"{dim}.npy"
        texts = [dimension_text(row, dim) for row in rows]
        arr = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        arr = np.asarray(arr, dtype=np.float32)
        np.save(path, arr)
        vecs[dim] = arr
    write_json(manifest, expected)
    return vecs


def embedding_signature(rows, dims, model_name):
    return {"model": model_name, "dims": dims, "input_digest": digest(rows)}


def load_embeddings(embedding_dir: Path, dims: list[str]) -> dict[str, np.ndarray]:
    return {dim: np.load(embedding_dir / f"{dim}.npy") for dim in dims}


def make_view_embedding(dim_vecs: dict[str, np.ndarray], dims: list[str]) -> np.ndarray:
    mats = [dim_vecs[dim] for dim in dims]
    return l2_normalize(np.concatenate(mats, axis=1).astype(np.float32))


def knn_candidates(vecs: np.ndarray, k: int, batch_size: int) -> list[tuple[int, int]]:
    n = vecs.shape[0]
    k = min(k, max(n - 1, 1))
    edges = set()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = vecs[start:end] @ vecs.T
        for r, i in enumerate(range(start, end)):
            sims[r, i] = -np.inf
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        for r, i in enumerate(range(start, end)):
            for j in idx[r]:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a != b:
                    edges.add((a, b))
    return sorted(edges)


def kmeans_partition(vecs: np.ndarray, k: int, seed: int, subset: Optional[np.ndarray]) -> np.ndarray:
    labels = np.full(vecs.shape[0], -1, dtype=np.int32)
    data = vecs if subset is None else vecs[subset]
    if data.shape[0] < k:
        return labels
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    part = model.fit_predict(data)
    if subset is None:
        labels[:] = part
    else:
        labels[subset] = part
    return labels


def build_partitions(vecs: np.ndarray, k_values: list[int], seeds: list[int], bootstrap: int, bootstrap_frac: float) -> list[np.ndarray]:
    rng = np.random.default_rng(12345)
    n = vecs.shape[0]
    partitions = []
    for k in k_values:
        for seed in seeds:
            partitions.append(kmeans_partition(vecs, k, seed, None))
            for _ in range(bootstrap):
                size = max(k + 1, int(n * bootstrap_frac))
                size = min(size, n)
                subset = np.sort(rng.choice(n, size=size, replace=False))
                partitions.append(kmeans_partition(vecs, k, seed + int(rng.integers(1, 1_000_000)), subset))
    return partitions


def consensus_matrix(partitions: list[np.ndarray], edges: list[tuple[int, int]], n: int, min_observed: int) -> sparse.csr_matrix:
    votes = np.zeros(len(edges), dtype=np.float32)
    denom = np.zeros(len(edges), dtype=np.float32)
    a = np.array([e[0] for e in edges], dtype=np.int32)
    b = np.array([e[1] for e in edges], dtype=np.int32)
    for labels in partitions:
        observed = (labels[a] >= 0) & (labels[b] >= 0)
        denom[observed] += 1.0
        votes[observed & (labels[a] == labels[b])] += 1.0
    keep = denom >= float(min_observed)
    weights = np.zeros_like(votes)
    weights[keep] = votes[keep] / np.maximum(denom[keep], 1.0)
    diag = np.arange(n, dtype=np.int32)
    rows = np.concatenate([a[keep], b[keep], diag])
    cols = np.concatenate([b[keep], a[keep], diag])
    vals = np.concatenate([weights[keep], weights[keep], np.ones(n, dtype=np.float32)])
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return mat


def confidence_scores(affinity: sparse.csr_matrix, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = labels.shape[0]
    conf_intra = np.zeros(n, dtype=np.float32)
    conf_ratio = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row = affinity.getrow(i)
        js = row.indices
        ws = row.data
        mask_self = js != i
        js = js[mask_self]
        ws = ws[mask_self]
        if len(js) == 0:
            continue
        own = ws[labels[js] == labels[i]]
        other = ws[labels[js] != labels[i]]
        own_mean = float(own.mean()) if len(own) else 0.0
        other_mean = float(other.mean()) if len(other) else 0.0
        conf_intra[i] = own_mean
        conf_ratio[i] = own_mean / (own_mean + other_mean + 1e-8)
    return conf_intra, conf_ratio


def cluster_summary(labels: np.ndarray, vecs: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(labels)
    summary = {
        "n_samples": int(labels.shape[0]),
        "n_clusters": int(len(counts)),
        "max_cluster_ratio": float(counts.max() / labels.shape[0]),
        "cluster_sizes": {str(i): int(v) for i, v in enumerate(counts)},
    }
    if len(counts) > 1 and min(counts) > 1:
        indices = np.random.RandomState(0).permutation(len(labels))[:5000] if len(labels) > 5000 else np.arange(len(labels))
        sampled_labels = labels[indices]
        summary["silhouette"] = (
            float(silhouette_score(vecs[indices], sampled_labels, metric="cosine"))
            if 1 < len(np.unique(sampled_labels)) < len(indices) else None
        )
        summary["silhouette_sample_size"] = min(len(labels), 5000)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse consensus clustering over value-semantic vectors.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dims", default="goal,principle,tradeoff,perspective")
    parser.add_argument("--k-final", type=positive_int, required=True)
    parser.add_argument("--k-base", default=None, help="Comma-separated K values for base partitions.")
    parser.add_argument("--seeds", default="11,23,37,51,67")
    parser.add_argument("--bootstrap", type=int, default=2)
    parser.add_argument("--bootstrap-frac", type=float, default=0.8)
    parser.add_argument("--knn", type=positive_int, default=50)
    parser.add_argument("--min-observed", type=positive_int, default=2)
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=positive_int, default=64)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_jsonl))
    dims = parse_list(args.dims)
    n = len(rows)
    if not 2 <= args.k_final < n:
        raise ValueError(f"--k-final must satisfy 2 <= K < number of edges ({n})")
    if not dims or len(set(dims)) != len(dims) or any(dim not in DIM_FIELDS for dim in dims):
        raise ValueError(f"--dims must contain unique fields from: {', '.join(DIM_FIELDS)}")
    if args.bootstrap < 0 or not 0 < args.bootstrap_frac <= 1:
        raise ValueError("--bootstrap must be >= 0 and --bootstrap-frac must be in (0, 1]")
    ids = [(row.get("pack_id"), row.get("edge_id")) for row in rows]
    if any(a is None or b is None for a, b in ids) or len(set(ids)) != n:
        raise ValueError("Each value row requires a unique (pack_id, edge_id)")
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("attributes"), dict) or any(not dimension_text(row, dim) for dim in dims):
            raise ValueError(f"Record {index}: missing value attributes for selected dimensions")
    k_values = parse_list(args.k_base, int) if args.k_base else sorted({args.k_final - 2, args.k_final, args.k_final + 2})
    if args.k_base and (not k_values or any(k < 2 or k > n for k in k_values)):
        raise ValueError(f"--k-base values must be between 2 and {n}")
    k_values = [k for k in k_values if 2 <= k <= n]
    seeds = parse_list(args.seeds, int)
    if not seeds or any(seed < 0 or seed >= 2 ** 32 for seed in seeds):
        raise ValueError("--seeds requires integers in [0, 2**32)")
    if args.min_observed > len(k_values) * len(seeds) * (1 + args.bootstrap):
        raise ValueError("--min-observed exceeds the total number of base partitions")
    out_dir = Path(args.output_dir)
    emb_dir = Path(args.embedding_dir) if args.embedding_dir else out_dir / "embeddings"
    if args.embedding_dir:
        manifest = emb_dir / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != embedding_signature(rows, dims, args.embedding_model):
            raise ValueError("Embedding manifest does not match the input rows, dimensions or model")
        dim_vecs = load_embeddings(emb_dir, dims)
    else:
        dim_vecs = build_embeddings(rows, dims, args.embedding_model, emb_dir, args.batch_size)

    for dim, arr in dim_vecs.items():
        if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] < 1 or not np.isfinite(arr).all():
            raise ValueError(f"{dim}.npy must be a finite 2-D matrix with {n} rows")
    vecs = make_view_embedding(dim_vecs, dims)

    print(f"Clustering {n} edges; dimensions: {', '.join(dims)}; K={args.k_final}", flush=True)
    edges = knn_candidates(vecs, args.knn, args.batch_size)
    partitions = build_partitions(vecs, k_values, seeds, args.bootstrap, args.bootstrap_frac)
    affinity = consensus_matrix(partitions, edges, vecs.shape[0], args.min_observed)
    labels = SpectralClustering(
        n_clusters=args.k_final,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    ).fit_predict(affinity)
    conf_intra, conf_ratio = confidence_scores(affinity, labels)

    out_rows = []
    for i, row in enumerate(rows):
        out = {
            "pack_id": row.get("pack_id"),
            "edge_id": row.get("edge_id"),
            "cluster_id": int(labels[i]),
            "conf_intra": float(conf_intra[i]),
            "conf_ratio": float(conf_ratio[i]),
        }
        out_rows.append(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "final_clusters.jsonl", out_rows)
    np.save(out_dir / "view_embedding.npy", vecs)
    sparse.save_npz(out_dir / "consensus_affinity.npz", affinity)
    with (out_dir / "cluster_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cluster_summary(labels, vecs), f, indent=2, ensure_ascii=False)
    print(f"Clustering outputs saved to {out_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(cli_entry(main))
