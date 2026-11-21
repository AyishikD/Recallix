// HNSW benchmark: build throughput, query latency and recall against exact
// search. Every number printed is measured in this run.
//
//   ./benchmark_hnsw [n=100000] [dim=128] [M=16] [efC=100] [efS=50] [data=clustered|uniform] [queries=500]
//
// Prints a human summary and one JSON line (prefixed "RESULT ") for tooling.

#include "vector_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::vector<std::vector<float>> makeData(int n, int dim, bool clustered, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> g(0.f, 1.f);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  int clusters = std::max(10, n / 1000);
  std::vector<std::vector<float>> centers(clusters, std::vector<float>(dim));
  for (auto &c : centers) for (float &x : c) x = g(rng);
  std::vector<std::vector<float>> out(n, std::vector<float>(dim));
  for (int i = 0; i < n; ++i) {
    auto &c = centers[rng() % clusters];
    double s = 0;
    for (int d = 0; d < dim; ++d) {
      out[i][d] = clustered ? c[d] + 0.35f * g(rng) : u(rng);
      s += out[i][d] * out[i][d];
    }
    for (float &x : out[i]) x /= (float)std::sqrt(s);
  }
  return out;
}

int main(int argc, char **argv) {
  auto arg = [&](int i, int def) { return argc > i ? std::atoi(argv[i]) : def; };
  const int n = arg(1, 100000), dim = arg(2, 128), M = arg(3, 16), efC = arg(4, 100), efS = arg(5, 50);
  const bool clustered = !(argc > 6 && std::string(argv[6]) == "uniform");
  const int nq = arg(7, 500), k = 10;

  std::cout << "n=" << n << " dim=" << dim << " M=" << M << " efC=" << efC << " efS=" << efS
            << " data=" << (clustered ? "clustered" : "uniform") << " SIMD=" << simdBackend()
            << " threads(hw)=" << std::thread::hardware_concurrency() << std::endl;

  // Queries are drawn from the same distribution as the data and held out of the index.
  auto all = makeData(n + nq, dim, clustered, 1);
  std::vector<std::vector<float>> data(all.begin(), all.begin() + n), queries(all.begin() + n, all.end());
  all.clear();
  all.shrink_to_fit();
  VectorEngine engine(dim, M, efC, efS, 42);

  auto t0 = Clock::now();
  for (int i = 0; i < n; ++i) engine.add(std::to_string(i), data[i]);
  double ingest_s = std::chrono::duration<double>(Clock::now() - t0).count();

  auto t1 = Clock::now();
  while (engine.indexPending(4096) > 0) {}
  double build_s = std::chrono::duration<double>(Clock::now() - t1).count();

  std::vector<double> lat;
  double hits = 0;
  for (auto &q : queries) {
    auto s = Clock::now();
    auto res = engine.search(q, k);
    lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - s).count());

    std::vector<std::pair<float, int>> exact;
    exact.reserve(n);
    for (int i = 0; i < n; ++i) exact.push_back({-dotProduct(data[i].data(), q.data(), dim), i});
    std::partial_sort(exact.begin(), exact.begin() + k, exact.end());
    std::set<std::string> got;
    for (auto &r : res) got.insert(r.first);
    for (int i = 0; i < k; ++i) hits += got.count(std::to_string(exact[i].second));
  }
  std::sort(lat.begin(), lat.end());
  double avg = 0;
  for (double l : lat) avg += l;
  avg /= lat.size();
  double p50 = lat[lat.size() / 2], p99 = lat[std::min(lat.size() - 1, (size_t)(lat.size() * 0.99))];
  double recall = hits / (nq * (double)k);

  std::cout << "ingest (append to pending): " << ingest_s * 1e6 / n << " us/vector\n"
            << "graph build: " << build_s << " s (" << n / build_s << " vectors/s, single thread)\n"
            << "search k=10: avg " << avg << " ms, p50 " << p50 << " ms, p99 " << p99 << " ms\n"
            << "recall@10 vs exact: " << recall << std::endl;
  std::cout << "RESULT {\"n\":" << n << ",\"dim\":" << dim << ",\"M\":" << M << ",\"efConstruction\":" << efC
            << ",\"efSearch\":" << efS << ",\"data\":\"" << (clustered ? "clustered" : "uniform")
            << "\",\"simd\":\"" << simdBackend() << "\",\"ingest_us_per_vector\":" << ingest_s * 1e6 / n
            << ",\"build_seconds\":" << build_s << ",\"search_avg_ms\":" << avg << ",\"search_p50_ms\":" << p50
            << ",\"search_p99_ms\":" << p99 << ",\"recall_at_10\":" << recall << "}" << std::endl;
  return 0;
}
