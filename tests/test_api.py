"""Light HTTP contract tests for the FastAPI review-analysis endpoint."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from backend.main import app


async def _post_json(path: str, payload: dict[str, object]):
    body = json.dumps(payload).encode("utf-8")
    request_sent = False
    messages: list[dict[str, object]] = []

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    await app(scope, receive, send)
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


class ReviewAnalysisEndpointTests(unittest.TestCase):
    def test_existing_and_new_routes_are_registered(self) -> None:
        routes = {(route.path, tuple(route.methods or ())) for route in app.routes}
        paths = {path for path, _ in routes}
        self.assertIn("/places", paths)
        self.assertIn("/places/{place_id}", paths)
        self.assertIn("/analyze-review", paths)

    def test_valid_review_returns_final_analysis_structure(self) -> None:
        fake_result = {
            "text": "The coffee was excellent but the service was very slow.",
            "sentence_count": 1,
            "sentences": ["The coffee was excellent but the service was very slow."],
            "sentence_aspect_probabilities": [],
            "aspects": {
                aspect: {"present": False}
                for aspect in ("taste", "atmosphere", "service", "value")
            },
            "no_target_aspects_detected": True,
            "sentiment_calls": 0,
        }
        with patch("backend.main.run_review_analysis", return_value=fake_result) as run:
            status, response = asyncio.run(
                _post_json(
                    "/analyze-review",
                    {"text": "The coffee was excellent but the service was very slow."},
                )
            )

        self.assertEqual(status, 200)
        self.assertEqual(response, fake_result)
        run.assert_called_once_with(
            "The coffee was excellent but the service was very slow."
        )
        self.assertEqual(
            tuple(response["aspects"]),
            ("taste", "atmosphere", "service", "value"),
        )

    def test_whitespace_text_returns_422_without_inference(self) -> None:
        with patch("backend.main.run_review_analysis") as run:
            status, response = asyncio.run(
                _post_json("/analyze-review", {"text": "  \n  "})
            )

        self.assertEqual(status, 422)
        self.assertEqual(response["detail"], "text must be a non-empty string")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
