#ifndef VECTOR_ENGINE_H
#define VECTOR_ENGINE_H

// HNSW approximate nearest-neighbour index (Malkov & Yashunin, 2018) over
// unit-normalised float vectors, distance = 1 - dot(a, b).
//
// Concurrency model: one FairSharedMutex per index (below).
//   * add / remove / indexPending (graph insertion) take the lock exclusively.
//   * search takes it shared and never re-locks.
// Every read of graph state therefore happens under the lock that guards it.
//
// Two-phase writes: add() appends the vector to a pending list in O(dim);
// indexPending() later links pending vectors into the graph. search() scans
// pending vectors exactly, so a write is visible to search immediately.

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// A shared mutex that cannot starve writers.
//
// std::shared_mutex makes no fairness promise, and glibc's implementation
// prefers readers: under continuous search traffic a writer waiting for
// exclusive access may never get in (new readers keep overlapping), so new
// memories would stop being added or linked. macOS's implementation is fair,
// which is why the problem only shows on Linux.
//
// Turnstile: every acquisition passes through `gate_`. A writer holds the gate
// while it waits for exclusive access, so new readers queue behind it while the
// readers already inside finish. The gate is held only during acquisition,
// never while the lock is in use, so it adds one uncontended mutex per call.
// Satisfies SharedMutex for std::unique_lock / std::shared_lock.
class FairSharedMutex {
public:
  void lock() {
    std::lock_guard<std::mutex> g(gate_);
    m_.lock();
  }
  bool try_lock() {
    std::unique_lock<std::mutex> g(gate_, std::try_to_lock);
    return g.owns_lock() && m_.try_lock();
  }
  void unlock() { m_.unlock(); }

  void lock_shared() {
    std::lock_guard<std::mutex> g(gate_);
    m_.lock_shared();
  }
  bool try_lock_shared() {
    std::unique_lock<std::mutex> g(gate_, std::try_to_lock);
    return g.owns_lock() && m_.try_lock_shared();
  }
  void unlock_shared() { m_.unlock_shared(); }

private:
  std::mutex gate_;
  std::shared_mutex m_;
};

struct IndexStats {
  std::size_t live;      // searchable vectors (indexed + pending, minus deleted)
  std::size_t indexed;   // linked into the graph
  std::size_t pending;   // waiting for graph insertion
  std::size_t deleted;   // tombstoned nodes still used for graph traversal
  int dim;
  int M;
  int efConstruction;
  int efSearch;
  int max_level;
};

class VectorEngine {
public:
  VectorEngine(int dim, int M, int efConstruction, int efSearch,
               unsigned seed = std::random_device{}());

  // Adds (or replaces) a vector. Returns false if vec.size() != dim().
  bool add(const std::string &id, const std::vector<float> &vec);
  // Tombstones a vector. Returns false if the id is unknown.
  bool remove(const std::string &id);

  // k nearest live vectors as (id, cosine similarity), best first.
  // ef_search <= 0 uses the index default; the beam is always >= k.
  std::vector<std::pair<std::string, float>>
  search(const std::vector<float> &query, int k, int ef_search = -1) const;

  // Exact k nearest live vectors by a full scan (no graph). Deterministic: the
  // result depends only on the stored vectors, never on graph shape, background
  // indexing progress or the level RNG; ties break by insertion order. Used where
  // reproducibility matters more than speed (e.g. the novelty feature at store time).
  std::vector<std::pair<std::string, float>>
  searchExact(const std::vector<float> &query, int k) const;

  // Links up to max_items pending vectors into the graph. Returns how many.
  std::size_t indexPending(std::size_t max_items);

  void setEfSearch(int ef);
  IndexStats stats() const;
  std::size_t pendingCount() const;
  int dim() const { return dim_; }

private:
  using Candidate = std::pair<float, int>; // (distance, node)

  const int dim_;
  const int M_;
  const int maxM0_; // max links on layer 0 (2 * M)
  const int efConstruction_;
  int efSearch_;
  const double levelMult_;

  mutable FairSharedMutex mu_;

  std::vector<float> data_; // node i occupies [i*dim, (i+1)*dim)
  std::vector<std::string> ids_;
  std::unordered_map<std::string, int> idToNode_;
  std::vector<int> level_;                          // -1 while pending
  std::vector<std::vector<std::vector<int>>> links_; // links_[node][layer]
  std::vector<char> deleted_;
  std::deque<int> pending_;
  std::size_t deletedCount_ = 0;
  std::size_t indexedCount_ = 0;

  int entry_ = -1;
  int maxLevel_ = -1;
  std::mt19937 rng_;

  const float *vec(int node) const { return data_.data() + (std::size_t)node * dim_; }
  float distance(const float *a, const float *b) const;
  int randomLevel();

  void insertNode(int node);
  int greedyClosest(const float *q, int entry, int layer) const;
  std::vector<Candidate> searchLayer(const float *q, int entry, int ef, int layer) const;
  std::vector<int> selectNeighbors(const float *q, std::vector<Candidate> candidates, int m) const;
  void addLink(int from, int to, int layer);
};

// Name of the compiled SIMD dot-product backend ("NEON", "AVX2", "SCALAR").
const char *simdBackend();
float dotProduct(const float *a, const float *b, int n);

#endif
