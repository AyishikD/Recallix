#ifndef INDEX_REGISTRY_H
#define INDEX_REGISTRY_H

// One HNSW index per namespace (Recallix uses one namespace per user), plus a
// single background worker that links pending vectors into whichever indexes
// have work. Dropping a namespace affects nobody else.

#include "vector_engine.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

struct IndexParams {
  int M;
  int efConstruction;
  int efSearch;
};

class IndexRegistry {
public:
  explicit IndexRegistry(IndexParams defaults) : defaults_(defaults) {
    worker_ = std::thread([this] { workerLoop(); });
  }
  ~IndexRegistry() {
    {
      std::lock_guard<std::mutex> g(wakeMu_);
      stop_ = true;
    }
    wake_.notify_all();
    worker_.join();
  }

  // Existing index, or nullptr.
  std::shared_ptr<VectorEngine> get(const std::string &ns) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = indexes_.find(ns);
    return it == indexes_.end() ? nullptr : it->second;
  }

  // Existing index, or a new one with the current default parameters.
  // Returns nullptr if the namespace exists with a different dimension.
  std::shared_ptr<VectorEngine> getOrCreate(const std::string &ns, int dim) {
    {
      std::shared_lock<std::shared_mutex> lock(mu_);
      auto it = indexes_.find(ns);
      if (it != indexes_.end())
        return it->second->dim() == dim ? it->second : nullptr;
    }
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto &slot = indexes_[ns];
    if (!slot)
      slot = std::make_shared<VectorEngine>(dim, defaults_.M, defaults_.efConstruction, defaults_.efSearch);
    return slot->dim() == dim ? slot : nullptr;
  }

  bool drop(const std::string &ns) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    return indexes_.erase(ns) > 0;
  }

  std::size_t dropAll() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    std::size_t n = indexes_.size();
    indexes_.clear();
    return n;
  }

  // New defaults apply to namespaces created afterwards; efSearch also applies
  // to existing ones (it is a query-time parameter).
  void configure(IndexParams p) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    defaults_ = p;
    for (auto &kv : indexes_)
      kv.second->setEfSearch(p.efSearch);
  }

  IndexParams defaults() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return defaults_;
  }

  std::map<std::string, IndexStats> stats() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    std::map<std::string, IndexStats> out;
    for (const auto &kv : indexes_)
      out.emplace(kv.first, kv.second->stats());
    return out;
  }

  void notifyWork() {
    {
      std::lock_guard<std::mutex> g(wakeMu_);
      workAvailable_ = true;
    }
    wake_.notify_one();
  }

private:
  mutable std::shared_mutex mu_;
  std::map<std::string, std::shared_ptr<VectorEngine>> indexes_;
  IndexParams defaults_;

  std::thread worker_;
  std::mutex wakeMu_;
  std::condition_variable wake_;
  bool workAvailable_ = false;
  bool stop_ = false;

  void workerLoop() {
    for (;;) {
      {
        std::unique_lock<std::mutex> g(wakeMu_);
        wake_.wait(g, [this] { return stop_ || workAvailable_; });
        if (stop_)
          return;
        workAvailable_ = false;
      }
      // Round-robin in small batches so one large rebuild cannot starve others.
      for (bool progressed = true; progressed;) {
        progressed = false;
        std::vector<std::shared_ptr<VectorEngine>> snapshot;
        {
          std::shared_lock<std::shared_mutex> lock(mu_);
          for (auto &kv : indexes_)
            snapshot.push_back(kv.second);
        }
        for (auto &idx : snapshot)
          if (idx->indexPending(64) > 0)
            progressed = true;
        std::lock_guard<std::mutex> g(wakeMu_);
        if (stop_)
          return;
      }
    }
  }
};

#endif
