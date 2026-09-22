import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_scenario_from_llm as generator


def make_response(body, content_type="application/json", status=200):
    response = requests.Response()
    response.status_code = status
    response._content = body.encode("utf-8")
    response.encoding = "utf-8"
    response.headers["Content-Type"] = content_type
    response.url = "https://example.invalid/test"
    return response


class ResponseTests(unittest.TestCase):
    def decode(self, response):
        return generator.response_json(
            response, provider="Test API", endpoint=response.url, api_key="SECRET"
        )

    def test_empty_and_html_responses(self):
        for body, content_type in [("", "text/plain"), ("<html>SECRET</html>", "text/html")]:
            with self.subTest(body=body):
                with self.assertRaises(RuntimeError) as caught:
                    self.decode(make_response(body, content_type))
                message = str(caught.exception)
                self.assertIn("non-JSON", message)
                self.assertIn("status=200", message)
                self.assertIn(content_type, message)
                self.assertNotIn("SECRET", message)
                self.assertIn("<empty>" if not body else "<redacted-api-key>", message)

    def test_json_object(self):
        self.assertEqual(self.decode(make_response('{"ok": true}')), {"ok": True})

    def test_non_object_json(self):
        for body in ["[]", "null", '"text"']:
            with self.subTest(body=body), self.assertRaisesRegex(RuntimeError, "expected an object"):
                self.decode(make_response(body))

    def test_redaction_before_truncation(self):
        summary = generator.response_summary(make_response("x" * 498 + "SECRET"), "SECRET")
        self.assertNotIn("SE", summary)
        self.assertIn("...", summary)

    def test_endpoint_redaction(self):
        response = make_response("")
        response.url += "?key=SECRET"
        with self.assertRaises(RuntimeError) as caught:
            self.decode(response)
        self.assertNotIn("SECRET", str(caught.exception))

    def test_both_provider_call_paths(self):
        config = {"api_base_url": "https://example.invalid", "api_key": "SECRET", "model_id": "test"}
        result = {"agents": [], "robot": {}}
        envelopes = [
            (generator.call_gemini_llm, {"candidates": [{"content": {"parts": [{"text": json.dumps(result)}]}}]}),
            (generator.call_openai_responses_llm, {"output": [{"content": [{"text": json.dumps(result)}]}]}),
        ]
        for call, envelope in envelopes:
            with self.subTest(provider=call.__name__):
                with patch.object(generator.requests, "post", return_value=make_response(json.dumps(envelope))):
                    self.assertEqual(call(config, "test", []), result)
                with patch.object(generator.requests, "post", return_value=make_response("<html>error</html>", "text/html")):
                    with self.assertRaisesRegex(RuntimeError, "non-JSON"):
                        call(config, "test", [])
                with patch.object(generator.requests, "post", return_value=make_response("gateway SECRET", "text/plain", 502)):
                    with self.assertRaises(RuntimeError) as caught:
                        call(config, "test", [])
                    self.assertIn("status=502", str(caught.exception))
                    self.assertNotIn("SECRET", str(caught.exception))

    def test_openai_chat_multimodal_request(self):
        config = {
            "api_base_url": "https://jkwl.dmxapi.cn/v1",
            "api_key": "SECRET",
            "model_id": "gpt-6-astra",
            "image_detail": "high",
        }
        result = {"agents": [], "robot": {}}
        envelope = {"choices": [{"message": {"content": json.dumps(result)}}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.png"
            image_path.write_bytes(b"image-bytes")
            with patch.object(
                generator.requests,
                "post",
                return_value=make_response(json.dumps(envelope)),
            ) as post:
                self.assertEqual(
                    generator.call_openai_chat_llm(config, "test prompt", [image_path]),
                    result,
                )

        endpoint = post.call_args.args[0]
        payload = post.call_args.kwargs["json"]
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(endpoint, "https://jkwl.dmxapi.cn/v1/chat/completions")
        self.assertEqual(payload["model"], "gpt-6-astra")
        self.assertEqual(payload["messages"][0]["content"][0], {"type": "text", "text": "test prompt"})
        image_part = payload["messages"][0]["content"][1]
        self.assertEqual(image_part["type"], "image_url")
        self.assertTrue(image_part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(image_part["image_url"]["detail"], "high")
        self.assertEqual(headers["Authorization"], "Bearer SECRET")
        self.assertEqual(post.call_args.kwargs["timeout"], (30.0, 180.0))
        self.assertNotIn("temperature", payload)
        self.assertNotIn("response_format", payload)

    def test_request_timeout_config(self):
        self.assertEqual(generator.request_timeout({}), (30.0, 180.0))
        self.assertEqual(
            generator.request_timeout({"connect_timeout": 10, "read_timeout": 600}),
            (10.0, 600.0),
        )
        self.assertEqual(generator.request_timeout({"timeout": 45}), (30.0, 45.0))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            generator.request_timeout({"read_timeout": 0})

    def test_openai_chat_timeout_message(self):
        config = {
            "api_base_url": "https://jkwl.dmxapi.cn/v1",
            "api_key": "SECRET",
            "model_id": "gpt-6-astra",
            "connect_timeout": 30,
            "read_timeout": 600,
        }
        with patch.object(generator.requests, "post", side_effect=requests.ReadTimeout("slow")):
            with self.assertRaises(RuntimeError) as caught:
                generator.call_openai_chat_llm(config, "prompt", [])
        message = str(caught.exception)
        self.assertIn("model gpt-6-astra", message)
        self.assertIn("read timeout=600.0s", message)
        self.assertIn("supports image input", message)
        self.assertNotIn("SECRET", message)


if __name__ == "__main__":
    unittest.main()
