"""
Public demo usage per day, from the permanent totals the demo gate keeps in Upstash Redis
(deploy/cloudflare/demo-gate/src/index.js, `track`). Counted since 2026-11-28.

    python3 scripts/demo_stats.py

Needs UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN in the environment or in .env. Prints totals only;
nothing per visitor is stored.
"""
import json
import os
import pathlib
import urllib.request

FIELDS = ("visitors", "sessions", "chatted", "messages", "questions")


def env(name):
    if os.environ.get(name):
        return os.environ[name]
    f = pathlib.Path(__file__).resolve().parents[1] / ".env"
    for line in f.read_text().splitlines() if f.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    raise SystemExit(f"{name} is not set")


URL, TOKEN = env("UPSTASH_REDIS_REST_URL").rstrip("/"), env("UPSTASH_REDIS_REST_TOKEN")


def redis(*commands):
    req = urllib.request.Request(f"{URL}/pipeline", data=json.dumps(list(commands)).encode(),
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    return [x["result"] for x in json.load(urllib.request.urlopen(req))]


def main():
    days, cursor = set(), "0"
    while True:
        cursor, keys = redis(["SCAN", cursor, "MATCH", "stats:2*", "COUNT", "1000"])[0]
        days |= {k.split(":", 1)[1] for k in keys}
        if cursor == "0":
            break
    days = sorted(days)
    if not days:
        print("No demo usage counted yet.")
        return
    results = redis(*[c for d in days for c in (["HGETALL", f"stats:{d}"], ["PFCOUNT", f"stats:visitors:{d}"])])
    total = dict.fromkeys(FIELDS, 0)
    print(f"{'day':<12}" + "".join(f"{f:>11}" for f in FIELDS))
    for i, d in enumerate(days):
        flat = results[2 * i] or []
        row = {k: int(v) for k, v in zip(flat[::2], flat[1::2])}
        row["visitors"] = int(results[2 * i + 1] or 0)
        for f in FIELDS:
            total[f] += row.get(f, 0)
        print(f"{d:<12}" + "".join(f"{row.get(f, 0):>11}" for f in FIELDS))
    print(f"{'total':<12}" + "".join(f"{total[f]:>11}" for f in FIELDS))
    print("visitors: estimated distinct visitors per day (summed over days, so one person on two days counts twice)")


if __name__ == "__main__":
    main()
