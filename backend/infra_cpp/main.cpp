// Recallix vector engine service.
//
// Configuration (environment):
//   RECALLIX_INTERNAL_KEY   required. Shared secret; every request except
//                           GET /health must send it as X-Internal-Key.
//   RECALLIX_ENGINE_HOST    bind address (default 127.0.0.1)
//   RECALLIX_ENGINE_PORT    port (default 8080)
//   RECALLIX_HNSW_M, RECALLIX_HNSW_EF_CONSTRUCTION, RECALLIX_HNSW_EF_SEARCH
//                           initial index parameters; the Python service also
//                           pushes config.yaml values via POST /configure.
//   RECALLIX_MAX_PAYLOAD_MB request body limit (default 256)
//
// API (JSON). Every vector operation is scoped to a namespace.
//   POST /configure   {M, efConstruction, efSearch}
//   POST /add         {namespace, id, vector}
//   POST /bulk_add    {namespace, items: [{id, vector}]}
//   POST /search      {namespace, query, top_k, ef_search?} -> {results: [{id, score}]}
//   POST /remove      {namespace, id}
//   POST /drop        {namespace}
//   POST /drop_all    {}
//   GET  /namespaces  -> {namespace: {live, indexed, pending, deleted, dim, M, ...}}
//   GET  /status      -> {status, instance, simd, pending_total, namespaces}
//   GET  /health      (no auth)
// Every response carries X-Engine-Instance (random per process start) so
// clients can detect a restart and rebuild the in-memory indexes.

#include "httplib.h"
#include "index_registry.h"
#include "json.hpp"
#include "vector_engine.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>

using json = nlohmann::json;

namespace {

std::string envOr(const char *name, const std::string &fallback) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::string(v) : fallback;
}

int envInt(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v)
    return fallback;
  try {
    return std::stoi(v);
  } catch (...) {
    std::cerr << "Invalid integer in " << name << "; using " << fallback << std::endl;
    return fallback;
  }
}

std::string randomHex(int bytes) {
  std::random_device rd;
  std::ostringstream os;
  for (int i = 0; i < bytes; ++i)
    os << std::hex << ((rd() & 0xF0) >> 4) << (rd() & 0x0F);
  return os.str();
}

bool constantTimeEquals(const std::string &a, const std::string &b) {
  if (a.size() != b.size())
    return false;
  unsigned char diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    diff |= (unsigned char)(a[i] ^ b[i]);
  return diff == 0;
}

void reply(httplib::Response &res, int status, const json &body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

json statsJson(const IndexStats &s) {
  return {{"live", s.live},         {"indexed", s.indexed}, {"pending", s.pending},
          {"deleted", s.deleted},   {"dim", s.dim},         {"M", s.M},
          {"efConstruction", s.efConstruction}, {"efSearch", s.efSearch}, {"max_level", s.max_level}};
}

// Wraps a handler: parses JSON, maps exceptions to 400 with a message.
template <typename F> httplib::Server::Handler jsonHandler(F f) {
  return [f](const httplib::Request &req, httplib::Response &res) {
    try {
      json body = req.body.empty() ? json::object() : json::parse(req.body);
      f(body, res);
    } catch (const std::exception &e) {
      reply(res, 400, {{"error", e.what()}});
    }
  };
}

} // namespace

int main() {
  const std::string key = envOr("RECALLIX_INTERNAL_KEY", "");
  if (key.size() < 16) {
    std::cerr << "RECALLIX_INTERNAL_KEY must be set (>= 16 chars). "
                 "Generate one with: python -m backend.tools.setup_env" << std::endl;
    return 1;
  }
  const std::string host = envOr("RECALLIX_ENGINE_HOST", "127.0.0.1");
  const int port = envInt("RECALLIX_ENGINE_PORT", 8080);
  const std::string instance = randomHex(8);

  IndexRegistry registry({envInt("RECALLIX_HNSW_M", 16), envInt("RECALLIX_HNSW_EF_CONSTRUCTION", 100),
                          envInt("RECALLIX_HNSW_EF_SEARCH", 50)});
  httplib::Server svr;

  svr.set_pre_routing_handler([&](const httplib::Request &req, httplib::Response &res) {
    res.set_header("X-Engine-Instance", instance);
    if (req.path == "/health")
      return httplib::Server::HandlerResponse::Unhandled;
    if (!constantTimeEquals(req.get_header_value("X-Internal-Key"), key)) {
      reply(res, 401, {{"error", "unauthorized"}});
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  svr.Post("/configure", jsonHandler([&](const json &b, httplib::Response &res) {
    IndexParams d = registry.defaults();
    IndexParams p{b.value("M", d.M), b.value("efConstruction", d.efConstruction), b.value("efSearch", d.efSearch)};
    if (p.M < 2 || p.efConstruction < 1 || p.efSearch < 1)
      return reply(res, 400, {{"error", "invalid parameters"}});
    registry.configure(p);
    reply(res, 200, {{"status", "ok"}, {"M", p.M}, {"efConstruction", p.efConstruction}, {"efSearch", p.efSearch}});
  }));

  auto addItems = [&](const std::string &ns, const json &items, httplib::Response &res) {
    if (items.empty())
      return reply(res, 200, {{"status", "ok"}, {"added", 0}});
    int dim = (int)items.at(0).at("vector").size();
    auto idx = registry.getOrCreate(ns, dim);
    if (!idx)
      return reply(res, 400, {{"error", "dimension mismatch for namespace"}, {"namespace", ns}});
    std::size_t added = 0;
    for (const auto &it : items) {
      if (!idx->add(it.at("id").get<std::string>(), it.at("vector").get<std::vector<float>>()))
        return reply(res, 400, {{"error", "dimension mismatch"}, {"expected", idx->dim()}, {"added", added}});
      ++added;
    }
    registry.notifyWork();
    reply(res, 200, {{"status", "ok"}, {"added", added}});
  };

  svr.Post("/add", jsonHandler([&](const json &b, httplib::Response &res) {
    json item = json::object();
    item["id"] = b.at("id");
    item["vector"] = b.at("vector");
    json items = json::array();
    items.push_back(item);
    addItems(b.at("namespace").get<std::string>(), items, res);
  }));

  svr.Post("/bulk_add", jsonHandler([&](const json &b, httplib::Response &res) {
    addItems(b.at("namespace").get<std::string>(), b.at("items"), res);
  }));

  svr.Post("/search", jsonHandler([&](const json &b, httplib::Response &res) {
    auto idx = registry.get(b.at("namespace").get<std::string>());
    json out = json::array();
    if (idx) {
      for (auto &r : idx->search(b.at("query").get<std::vector<float>>(), b.value("top_k", 10), b.value("ef_search", -1)))
        out.push_back({{"id", r.first}, {"score", r.second}});
    }
    reply(res, 200, {{"results", out}});
  }));

  svr.Post("/remove", jsonHandler([&](const json &b, httplib::Response &res) {
    auto idx = registry.get(b.at("namespace").get<std::string>());
    bool removed = idx && idx->remove(b.at("id").get<std::string>());
    reply(res, 200, {{"status", "ok"}, {"removed", removed}});
  }));

  svr.Post("/drop", jsonHandler([&](const json &b, httplib::Response &res) {
    reply(res, 200, {{"status", "ok"}, {"dropped", registry.drop(b.at("namespace").get<std::string>())}});
  }));

  svr.Post("/drop_all", jsonHandler([&](const json &, httplib::Response &res) {
    reply(res, 200, {{"status", "ok"}, {"dropped", registry.dropAll()}});
  }));

  svr.Get("/namespaces", [&](const httplib::Request &, httplib::Response &res) {
    json out = json::object();
    for (auto &kv : registry.stats())
      out[kv.first] = statsJson(kv.second);
    reply(res, 200, out);
  });

  svr.Get("/status", [&](const httplib::Request &, httplib::Response &res) {
    std::size_t pending = 0;
    auto all = registry.stats();
    for (auto &kv : all)
      pending += kv.second.pending;
    IndexParams d = registry.defaults();
    reply(res, 200, {{"status", "online"}, {"instance", instance}, {"simd", simdBackend()},
                     {"pending_total", pending}, {"namespaces", all.size()},
                     {"defaults", {{"M", d.M}, {"efConstruction", d.efConstruction}, {"efSearch", d.efSearch}}}});
  });

  svr.Get("/health", [&](const httplib::Request &, httplib::Response &res) {
    reply(res, 200, {{"status", "healthy"}, {"simd", simdBackend()}});
  });

  svr.set_payload_max_length((std::size_t)envInt("RECALLIX_MAX_PAYLOAD_MB", 256) * 1024 * 1024);

  std::cout << "Recallix vector engine on http://" << host << ":" << port << " (SIMD " << simdBackend()
            << ", instance " << instance << ")" << std::endl;
  if (!svr.listen(host, port)) {
    std::cerr << "Failed to bind " << host << ":" << port << std::endl;
    return 1;
  }
  return 0;
}
