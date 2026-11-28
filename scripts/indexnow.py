"""
Tell IndexNow (Bing, Yandex, Seznam, Naver) that recallic.com's pages changed, so they are re-crawled now instead
of whenever the crawler comes by. Google does not use IndexNow; it reads the sitemap.

    python scripts/indexnow.py                 # every URL in the live sitemap
    python scripts/indexnow.py URL [URL ...]   # just these

The key is public by design: the same value is served at https://www.recallic.com/<key>.txt to prove ownership.
Run by .github/workflows/indexnow.yml after each successful production deploy.
"""
import json
import re
import sys
import urllib.error
import urllib.request

HOST = "www.recallic.com"
KEY = "a3dd7a728c9ce714bc46dc687948c8fc"
SITEMAP = f"https://{HOST}/sitemap.xml"


def sitemap_urls():
    with urllib.request.urlopen(SITEMAP, timeout=30) as r:
        return re.findall(r"<loc>([^<]+)</loc>", r.read().decode())


def main(argv):
    urls = argv or sitemap_urls()
    body = json.dumps({"host": HOST, "key": KEY, "keyLocation": f"https://{HOST}/{KEY}.txt", "urlList": urls}).encode()
    req = urllib.request.Request("https://api.indexnow.org/indexnow", data=body, method="POST",
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"IndexNow: HTTP {r.status} for {len(urls)} URLs")  # 200 or 202 = accepted
    except urllib.error.HTTPError as e:
        # 403: the key file is not live yet; 422: a URL is not on this host; 429: too many submissions
        sys.exit(f"IndexNow refused: HTTP {e.code} {e.read().decode()[:200]}")


if __name__ == "__main__":
    main(sys.argv[1:])
