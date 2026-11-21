def _store(client, headers, content, **extra):
    r = client.post("/memory/store", json={"content": content, **extra}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def _recall(client, headers, query, **extra):
    r = client.post("/memory/recall", json={"query": query, **extra}, headers=headers)
    assert r.status_code == 200, r.text
    return [m["content"] for m in r.json()["data"]["memories"]]


def test_users_never_see_each_others_memories(client, api_keys, engine):
    _store(client, api_keys["alice"], "My bank PIN is 4321.")
    _store(client, api_keys["bob"], "My favourite colour is green.")
    assert not any("4321" in c for c in _recall(client, api_keys["bob"], "What is my bank PIN?"))
    assert not any("green" in c for c in _recall(client, api_keys["alice"], "What is my favourite colour?"))
    spaces = engine.get("/namespaces")
    assert spaces["alice"]["live"] == 1 and spaces["bob"]["live"] == 1


def test_private_memories_are_agent_scoped_and_shared_ones_are_not(client, api_keys):
    h = api_keys["alice"]
    _store(client, h, "The Atlas password is swordfish.", agent_id="planner", memory_type="private")
    _store(client, h, "The Atlas deadline is June 30.", agent_id="planner", memory_type="shared")

    planner = _recall(client, h, "Atlas password deadline", agent_id="planner", limit=10)
    researcher = _recall(client, h, "Atlas password deadline", agent_id="researcher", limit=10)
    assert any("swordfish" in c for c in planner)
    assert not any("swordfish" in c for c in researcher)
    assert any("June 30" in c for c in researcher)

    r = client.post("/agents/recall", json={"query": "Atlas password", "agent_id": "researcher"}, headers=h)
    assert not any("swordfish" in m["content"] for m in r.json()["data"]["memories"])


def test_invalid_memory_type_rejected(client, api_keys):
    r = client.post("/memory/store", json={"content": "x y z", "memory_type": "public"}, headers=api_keys["alice"])
    assert r.status_code == 422
