"""
Request size cap (config api.max_request_bytes; api.max_import_bytes for the bulk writes POST /memory/import
and POST /memory/store_batch).

A request that declares a larger Content-Length is refused with 413 before its body is read.
One without a length (chunked) is counted as it arrives and cut off at the limit, so an
oversized body never sits in memory either way.
"""
import json

from backend import settings

BULK_PATHS = ("/memory/import", "/memory/store_batch")


def _limit(path):
    if path in BULK_PATHS:
        return int(settings.get("api.max_import_bytes", 10 * 1024 * 1024))
    return int(settings.get("api.max_request_bytes", 1024 * 1024))


async def _too_large(send, limit):
    body = json.dumps({"detail": f"Request body larger than {limit} bytes"}).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class BodySizeLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = _limit(scope.get("path", ""))
        if limit <= 0:
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > limit:
                    return await _too_large(send, limit)
            except ValueError:
                return await _too_large(send, limit)
        seen = 0

        async def counted():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:  # no (honest) Content-Length: stop reading, the handler sees a disconnect
                    return {"type": "http.disconnect"}
            return message
        return await self.app(scope, counted, send)
