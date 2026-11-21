#include "vector_engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <queue>

// ============================================================================
// SIMD dot product
// ============================================================================
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
const char *simdBackend() { return "NEON"; }
float dotProduct(const float *a, const float *b, int n) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4)
    acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
  float r = vaddvq_f32(acc);
  for (; i < n; ++i)
    r += a[i] * b[i];
  return r;
}
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
const char *simdBackend() { return "AVX2"; }
float dotProduct(const float *a, const float *b, int n) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float r = _mm_cvtss_f32(s);
  for (; i < n; ++i)
    r += a[i] * b[i];
  return r;
}
#else
const char *simdBackend() { return "SCALAR"; }
float dotProduct(const float *a, const float *b, int n) {
  float r = 0.0f;
  for (int i = 0; i < n; ++i)
    r += a[i] * b[i];
  return r;
}
#endif

namespace {

// Reusable per-thread visited marks: O(1) reset via a version counter instead
// of allocating an O(n) array for every search.
struct VisitedMarks {
  std::vector<std::uint32_t> tag;
  std::uint32_t version = 0;
  void reset(std::size_t n) {
    if (tag.size() < n)
      tag.resize(n, 0);
    if (++version == 0) { // wrapped: clear and restart
      std::fill(tag.begin(), tag.end(), 0);
      version = 1;
    }
  }
  bool visit(int node) { // returns true if newly visited
    if (tag[node] == version)
      return false;
    tag[node] = version;
    return true;
  }
};
thread_local VisitedMarks t_visited;

std::vector<float> normalized(const std::vector<float> &v) {
  double sq = 0.0;
  for (float x : v)
    sq += (double)x * x;
  std::vector<float> out(v);
  if (sq > 1e-20) {
    float inv = (float)(1.0 / std::sqrt(sq));
    for (float &x : out)
      x *= inv;
  }
  return out;
}

} // namespace

// ============================================================================
// VectorEngine
// ============================================================================
VectorEngine::VectorEngine(int dim, int M, int efConstruction, int efSearch, unsigned seed)
    : dim_(dim), M_(std::max(2, M)), maxM0_(2 * std::max(2, M)),
      efConstruction_(std::max(efConstruction, M)), efSearch_(std::max(1, efSearch)),
      levelMult_(1.0 / std::log((double)std::max(2, M))), rng_(seed) {}

float VectorEngine::distance(const float *a, const float *b) const {
  return 1.0f - dotProduct(a, b, dim_);
}

int VectorEngine::randomLevel() {
  std::uniform_real_distribution<double> u(std::numeric_limits<double>::min(), 1.0);
  return std::min(32, (int)std::floor(-std::log(u(rng_)) * levelMult_));
}

bool VectorEngine::add(const std::string &id, const std::vector<float> &vec_in) {
  if ((int)vec_in.size() != dim_)
    return false;
  std::vector<float> v = normalized(vec_in);

  std::unique_lock<FairSharedMutex> lock(mu_);
  auto it = idToNode_.find(id);
  if (it != idToNode_.end() && !deleted_[it->second]) { // upsert: tombstone old version
    deleted_[it->second] = 1;
    ++deletedCount_;
  }
  int node = (int)ids_.size();
  data_.insert(data_.end(), v.begin(), v.end());
  ids_.push_back(id);
  idToNode_[id] = node;
  level_.push_back(-1);
  links_.emplace_back();
  deleted_.push_back(0);
  pending_.push_back(node);
  return true;
}

bool VectorEngine::remove(const std::string &id) {
  std::unique_lock<FairSharedMutex> lock(mu_);
  auto it = idToNode_.find(id);
  if (it == idToNode_.end())
    return false;
  if (!deleted_[it->second]) {
    deleted_[it->second] = 1;
    ++deletedCount_;
  }
  idToNode_.erase(it);
  return true;
}

std::size_t VectorEngine::indexPending(std::size_t max_items) {
  std::size_t done = 0;
  while (done < max_items) {
    std::unique_lock<FairSharedMutex> lock(mu_); // per node, so searches interleave
    if (pending_.empty())
      break;
    int node = pending_.front();
    pending_.pop_front();
    if (!deleted_[node])
      insertNode(node);
    ++done;
  }
  return done;
}

int VectorEngine::greedyClosest(const float *q, int ep, int layer) const {
  float best = distance(q, vec(ep));
  for (bool changed = true; changed;) {
    changed = false;
    for (int n : links_[ep][layer]) {
      float d = distance(q, vec(n));
      if (d < best) {
        best = d;
        ep = n;
        changed = true;
      }
    }
  }
  return ep;
}

std::vector<VectorEngine::Candidate>
VectorEngine::searchLayer(const float *q, int ep, int ef, int layer) const {
  VisitedMarks &visited = t_visited;
  visited.reset(ids_.size());

  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> frontier; // min-heap
  std::priority_queue<Candidate> best;                                                    // max-heap
  float d0 = distance(q, vec(ep));
  frontier.push({d0, ep});
  best.push({d0, ep});
  visited.visit(ep);

  while (!frontier.empty()) {
    Candidate c = frontier.top();
    if (c.first > best.top().first && (int)best.size() >= ef)
      break;
    frontier.pop();
    for (int n : links_[c.second][layer]) {
      if (!visited.visit(n))
        continue;
      float d = distance(q, vec(n));
      if ((int)best.size() < ef || d < best.top().first) {
        frontier.push({d, n});
        best.push({d, n});
        if ((int)best.size() > ef)
          best.pop();
      }
    }
  }
  std::vector<Candidate> out;
  out.reserve(best.size());
  while (!best.empty()) {
    out.push_back(best.top());
    best.pop();
  }
  std::reverse(out.begin(), out.end()); // closest first
  return out;
}

// Neighbour-selection heuristic (paper, Algorithm 4, keepPrunedConnections):
// keep a candidate only if it is closer to the query than to every neighbour
// already kept, which spreads links across directions; then top up with the
// closest pruned candidates.
std::vector<int> VectorEngine::selectNeighbors(const float *q, std::vector<Candidate> cands, int m) const {
  (void)q;
  std::sort(cands.begin(), cands.end());
  std::vector<int> kept;
  std::vector<int> pruned;
  for (const auto &c : cands) {
    if ((int)kept.size() >= m)
      break;
    bool diverse = true;
    for (int s : kept) {
      if (distance(vec(c.second), vec(s)) < c.first) {
        diverse = false;
        break;
      }
    }
    (diverse ? kept : pruned).push_back(c.second);
  }
  for (int p : pruned) {
    if ((int)kept.size() >= m)
      break;
    kept.push_back(p);
  }
  return kept;
}

void VectorEngine::addLink(int from, int to, int layer) {
  auto &l = links_[from][layer];
  l.push_back(to);
  int cap = layer == 0 ? maxM0_ : M_;
  if ((int)l.size() <= cap)
    return;
  std::vector<Candidate> cands;
  cands.reserve(l.size());
  for (int n : l)
    cands.push_back({distance(vec(from), vec(n)), n});
  l = selectNeighbors(vec(from), std::move(cands), cap);
}

void VectorEngine::insertNode(int node) {
  int lvl = randomLevel();
  level_[node] = lvl;
  links_[node].assign(lvl + 1, {});
  ++indexedCount_;

  if (entry_ < 0) {
    entry_ = node;
    maxLevel_ = lvl;
    return;
  }
  const float *q = vec(node);
  int ep = entry_;
  for (int l = maxLevel_; l > lvl; --l)
    ep = greedyClosest(q, ep, l);
  for (int l = std::min(lvl, maxLevel_); l >= 0; --l) {
    std::vector<Candidate> found = searchLayer(q, ep, efConstruction_, l);
    links_[node][l] = selectNeighbors(q, found, M_);
    for (int n : links_[node][l])
      addLink(n, node, l);
    ep = found.front().second;
  }
  if (lvl > maxLevel_) {
    entry_ = node;
    maxLevel_ = lvl;
  }
}

std::vector<std::pair<std::string, float>>
VectorEngine::search(const std::vector<float> &query, int k, int ef_search) const {
  if ((int)query.size() != dim_ || k <= 0)
    return {};
  std::vector<float> q = normalized(query);

  std::shared_lock<FairSharedMutex> lock(mu_);
  std::vector<Candidate> hits;

  if (entry_ >= 0) {
    int ef = std::max(ef_search > 0 ? ef_search : efSearch_, k);
    // Tombstones still occupy beam slots; widen the beam so k live results survive.
    ef += (int)std::min<std::size_t>(deletedCount_, (std::size_t)ef * 4);
    int ep = entry_;
    for (int l = maxLevel_; l > 0; --l)
      ep = greedyClosest(q.data(), ep, l);
    for (const auto &c : searchLayer(q.data(), ep, ef, 0))
      if (!deleted_[c.second])
        hits.push_back(c);
  }
  for (int n : pending_) // not yet in the graph: exact scan
    if (!deleted_[n])
      hits.push_back({distance(q.data(), vec(n)), n});

  std::size_t take = std::min<std::size_t>((std::size_t)k, hits.size());
  std::partial_sort(hits.begin(), hits.begin() + take, hits.end());
  std::vector<std::pair<std::string, float>> out;
  out.reserve(take);
  for (std::size_t i = 0; i < take; ++i)
    out.push_back({ids_[hits[i].second], 1.0f - hits[i].first});
  return out;
}

void VectorEngine::setEfSearch(int ef) {
  std::unique_lock<FairSharedMutex> lock(mu_);
  efSearch_ = std::max(1, ef);
}

std::size_t VectorEngine::pendingCount() const {
  std::shared_lock<FairSharedMutex> lock(mu_);
  return pending_.size();
}

IndexStats VectorEngine::stats() const {
  std::shared_lock<FairSharedMutex> lock(mu_);
  return {ids_.size() - deletedCount_, indexedCount_, pending_.size(), deletedCount_,
          dim_, M_, efConstruction_, efSearch_, maxLevel_};
}
