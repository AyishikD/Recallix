// Correctness tests for VectorEngine / IndexRegistry. Exit code 0 = all passed.
// Build the `test_vector_engine` target; the `test_vector_engine_tsan` target
// runs the same tests under ThreadSanitizer.

#include "index_registry.h"
#include "vector_engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg)                                                                                     \
  do {                                                                                                       \
    if (!(cond)) {                                                                                           \
      std::cerr << "  FAIL " << __LINE__ << ": " << msg << std::endl;                                        \
      ++g_failures;                                                                                          \
    }                                                                                                        \
  } while (0)

static std::vector<float> unit(std::vector<float> v) {
  double s = 0;
  for (float x : v) s += x * x;
  for (float &x : v) x /= (float)std::sqrt(s);
  return v;
}

// Gaussian clusters: closer to real embedding geometry than uniform noise.
static std::vector<std::vector<float>> clustered(int n, int dim, int clusters, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
  for (auto &c : centers) for (float &x : c) x = g(rng);
  std::vector<std::vector<float>> out(n, std::vector<float>(dim));
  for (int i = 0; i < n; ++i) {
    auto &c = centers[rng() % clusters];
    for (int d = 0; d < dim; ++d) out[i][d] = c[d] + 0.35f * g(rng);
    out[i] = unit(out[i]);
  }
  return out;
}

static std::vector<int> bruteForce(const std::vector<std::vector<float>> &data, const std::vector<float> &q, int k,
                                   const std::set<int> &skip = {}) {
  std::vector<std::pair<float, int>> s;
  for (int i = 0; i < (int)data.size(); ++i)
    if (!skip.count(i)) s.push_back({-dotProduct(data[i].data(), q.data(), (int)q.size()), i});
  std::partial_sort(s.begin(), s.begin() + std::min<size_t>(k, s.size()), s.end());
  std::vector<int> ids;
  for (int i = 0; i < k && i < (int)s.size(); ++i) ids.push_back(s[i].second);
  return ids;
}

static void indexAll(VectorEngine &e) {
  while (e.indexPending(1024) > 0) {}
}

static void testRecall() {
  std::cout << "recall@10 vs exact search" << std::endl;
  const int n = 20000, dim = 64, k = 10;
  // Queries come from the same distribution as the data but are held out of the index.
  auto all = clustered(n + 200, dim, 50, 1);
  std::vector<std::vector<float>> data(all.begin(), all.begin() + n), queries(all.begin() + n, all.end());
  VectorEngine e(dim, 16, 100, 64, 42);
  for (int i = 0; i < n; ++i) e.add(std::to_string(i), data[i]);
  indexAll(e);
  double hits = 0;
  for (auto &q : queries) {
    auto truth = bruteForce(data, q, k);
    std::set<std::string> got;
    for (auto &r : e.search(q, k)) got.insert(r.first);
    for (int t : truth) hits += got.count(std::to_string(t));
  }
  double recall = hits / (queries.size() * k);
  std::cout << "  recall@10 = " << recall << std::endl;
  CHECK(recall >= 0.95, "recall@10 below 0.95: " << recall);
}

static void testPendingVisible() {
  std::cout << "pending writes are searchable before indexing" << std::endl;
  VectorEngine e(8, 8, 32, 16, 3);
  std::vector<float> v = unit({1, 2, 3, 4, 5, 6, 7, 8});
  e.add("a", v);
  CHECK(e.pendingCount() == 1, "expected 1 pending");
  auto r = e.search(v, 1);
  CHECK(r.size() == 1 && r[0].first == "a", "pending vector not found");
  CHECK(r.size() == 1 && std::fabs(r[0].second - 1.0f) < 1e-4, "self-similarity should be 1");
}

static void testDeleteAndUpsert() {
  std::cout << "delete / upsert" << std::endl;
  const int n = 3000, dim = 32;
  auto data = clustered(n, dim, 10, 4);
  VectorEngine e(dim, 12, 64, 32, 5);
  for (int i = 0; i < n; ++i) e.add(std::to_string(i), data[i]);
  indexAll(e);
  std::set<int> removed;
  for (int i = 0; i < n; i += 3) {
    CHECK(e.remove(std::to_string(i)), "remove failed");
    removed.insert(i);
  }
  CHECK(!e.remove("does-not-exist"), "removing unknown id should fail");
  int returned_deleted = 0, short_results = 0;
  for (int i = 0; i < 100; ++i) {
    auto res = e.search(data[i], 10);
    if (res.size() < 10) ++short_results;
    for (auto &r : res) returned_deleted += removed.count(std::stoi(r.first));
  }
  CHECK(returned_deleted == 0, "deleted ids returned: " << returned_deleted);
  CHECK(short_results == 0, "searches returned fewer than k live results: " << short_results);
  CHECK(e.stats().live == (size_t)(n - removed.size()), "live count wrong");

  // Upsert: re-adding an id replaces its vector.
  std::vector<float> other = data[1];
  e.add("2", other);
  auto r = e.search(other, 2);
  bool has2 = false;
  for (auto &x : r) has2 |= x.first == "2";
  CHECK(has2, "upserted id not found at its new position");
  CHECK(e.stats().live == (size_t)(n - removed.size()), "upsert changed live count");
}

static void testDimensionAndK() {
  std::cout << "dimension checks / k larger than ef" << std::endl;
  VectorEngine e(16, 8, 32, 8, 6);
  CHECK(!e.add("bad", std::vector<float>(15, 1.f)), "wrong dimension accepted");
  CHECK(e.search(std::vector<float>(15, 1.f), 3).empty(), "wrong-dimension query returned results");
  auto data = clustered(500, 16, 5, 7);
  for (int i = 0; i < 500; ++i) e.add(std::to_string(i), data[i]);
  indexAll(e);
  CHECK(e.search(data[0], 200).size() == 200, "k > efSearch must still return k results");

  IndexRegistry reg({8, 32, 16});
  CHECK(reg.getOrCreate("u1", 16) != nullptr, "create failed");
  CHECK(reg.getOrCreate("u1", 32) == nullptr, "dimension mismatch on existing namespace accepted");
  reg.getOrCreate("u2", 16)->add("x", data[0]);
  CHECK(reg.drop("u1"), "drop failed");
  CHECK(reg.get("u1") == nullptr && reg.get("u2") != nullptr, "drop affected another namespace");
}

static void testConcurrency() {
  std::cout << "concurrent add / remove / index / search" << std::endl;
  const int dim = 32;
  auto data = clustered(6000, dim, 20, 8);
  IndexRegistry reg({12, 48, 24});
  auto idx = reg.getOrCreate("u", dim);
  std::atomic<bool> stop{false};
  std::atomic<int> searches{0}, bad{0};

  std::thread writer([&] {
    for (int i = 0; i < 6000; ++i) {
      idx->add(std::to_string(i), data[i]);
      if (i % 7 == 0) idx->remove(std::to_string(i / 2));
      reg.notifyWork();
    }
  });
  std::vector<std::thread> readers;
  for (int t = 0; t < 4; ++t)
    readers.emplace_back([&, t] {
      std::mt19937 rng(t);
      while (!stop) {
        auto res = idx->search(data[rng() % 6000], 10);
        for (size_t i = 1; i < res.size(); ++i)
          if (res[i].second > res[i - 1].second + 1e-5f) ++bad; // must be sorted best-first
        ++searches;
      }
    });
  writer.join();
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (idx->pendingCount() > 0 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  stop = true;
  for (auto &r : readers) r.join();
  CHECK(idx->pendingCount() == 0, "background worker did not drain pending vectors");
  CHECK(bad == 0, "unsorted results under concurrency");
  CHECK(searches > 0, "no searches ran");
  std::cout << "  " << searches << " searches during concurrent writes" << std::endl;
}

int main() {
  std::cout << "SIMD backend: " << simdBackend() << std::endl;
  testPendingVisible();
  testDeleteAndUpsert();
  testDimensionAndK();
  testConcurrency();
  testRecall();
  if (g_failures) {
    std::cerr << g_failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "all tests passed" << std::endl;
  return 0;
}
