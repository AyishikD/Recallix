"""
Reference comparison: hnswlib (the reference HNSW implementation) on the same
data distribution and parameters as backend/infra_cpp/benchmark_hnsw.

  pip install hnswlib
  python eval/hnsw_reference.py 100000 128 clustered 50 >> eval/results/hnsw_reference.jsonl

Prints one JSON line: {"impl": "hnswlib", n, dim, data, M, efConstruction, efSearch,
recall_at_10, build_seconds, search_avg_ms}. Single-threaded for a fair comparison.
"""
import json
import sys
import time

import numpy as np


def make(n, dim, kind, rng, n_clusters):
    if kind == "uniform":
        x = rng.uniform(-1, 1, (n, dim)).astype(np.float32)
    else:
        centers = rng.normal(size=(n_clusters, dim)).astype(np.float32)
        x = centers[rng.integers(0, n_clusters, n)] + 0.35 * rng.normal(size=(n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    import hnswlib
    n, dim, kind = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ef = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    M, efc, nq = 16, 100, 300
    rng = np.random.default_rng(1)
    allx = make(n + nq, dim, kind, rng, max(10, n // 1000))
    data, q = allx[:n], allx[n:]
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=n, M=M, ef_construction=efc, random_seed=42)
    index.set_num_threads(1)
    t = time.time()
    index.add_items(data)
    build = time.time() - t
    index.set_ef(ef)
    lat = []
    for row in q:
        s = time.perf_counter()
        index.knn_query(row[None, :], k=10)
        lat.append((time.perf_counter() - s) * 1000)
    labels, _ = index.knn_query(q, k=10)
    exact = np.argsort(-(q @ data.T), axis=1)[:, :10]
    recall = float(np.mean([len(set(labels[i]) & set(exact[i])) / 10 for i in range(nq)]))
    print(json.dumps({"impl": "hnswlib", "n": n, "dim": dim, "data": kind, "M": M, "efConstruction": efc,
                      "efSearch": ef, "recall_at_10": round(recall, 4), "build_seconds": round(build, 2),
                      "search_avg_ms": round(float(np.mean(lat)), 4)}))


if __name__ == "__main__":
    main()
