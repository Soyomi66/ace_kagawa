from __future__ import annotations

import http.server
import importlib.util
import io
import json
import math
import socket
import struct
import sys
import threading
import unittest
import wave
from dataclasses import replace
from pathlib import Path

try:
    import fastapi
except ModuleNotFoundError:
    fastapi = None


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


irodori_server = load_module("irodori_server", ROOT / "infra" / "tts" / "irodori_server.py")


def sine_wav(sample_rate: int, seconds: float = 0.5) -> bytes:
    samples = int(sample_rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(
            b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * 220 * i / sample_rate))) for i in range(samples))
        )
    return buffer.getvalue()


class FakeSokutou(http.server.ThreadingHTTPServer):
    """sokutou's voice server as engine/serve.py answers it (commit 68b871c).

    /health answers 503 for the first `warming` calls. Input containing TOOLONG answers 400,
    BABBLE answers 500 "implausible duration" `babbles` times, BROKEN always answers 500.
    """

    def __init__(self, *, num_steps: int = 12, warming: int = 0, sample_rate: int = 48000, babbles: int = 1) -> None:
        super().__init__(("127.0.0.1", 0), FakeSokutouHandler)
        self.num_steps = num_steps
        self.warming = warming
        self.sample_rate = sample_rate
        self.babbles = babbles
        self.health_calls = 0
        self.bodies: list[dict] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class FakeSokutouHandler(http.server.BaseHTTPRequestHandler):
    server: FakeSokutou

    def log_message(self, *_args) -> None:
        pass

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode())

    def do_GET(self) -> None:
        fake = self.server
        if self.path == "/health":
            fake.health_calls += 1
            ready = fake.health_calls > fake.warming
            self._json(200 if ready else 503, {"ok": ready, "voices": ["kagawa"], "default_voice": "kagawa"})
        elif self.path == "/v1/engine":
            self._json(
                200,
                {
                    "engine": "sokutou fused-bf16",
                    "ready": True,
                    "num_steps": fake.num_steps,
                    "requests": len(fake.bodies),
                    "irregular_requests": 0,
                    "fallbacks": 0,
                    "startup": {"num_steps": fake.num_steps, "watermark": False},
                },
            )
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self) -> None:
        fake = self.server
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        fake.bodies.append(body)
        text = body["input"]
        if "TOOLONG" in text:
            self._json(400, {"detail": "input is longer than 200 characters"})
        elif "BROKEN" in text:
            self._json(500, {"detail": "CUDA error: an illegal memory access was encountered"})
        elif "BABBLE" in text and fake.babbles > 0:
            fake.babbles -= 1
            self._json(500, {"detail": "implausible duration 30.0s for 9 chars"})
        else:
            self._send(200, sine_wav(fake.sample_rate), "audio/wav")


def closed_port_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


class SokutouBackendTests(unittest.TestCase):
    def start_fake(self, **kwargs) -> FakeSokutou:
        fake = FakeSokutou(**kwargs)
        thread = threading.Thread(target=fake.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(fake.server_close)
        self.addCleanup(fake.shutdown)
        return fake

    def settings(self, url: str, **overrides):
        values = {
            "IRODORI_TTS_BACKEND": "sokutou",
            "IRODORI_TTS_SOKUTOU_URL": url,
            "IRODORI_TTS_SOKUTOU_STARTUP_TIMEOUT_SECONDS": "5",
            "IRODORI_TTS_NUM_STEPS": "12",
        }
        values.update(overrides)
        return irodori_server.IrodoriSettings.from_env(values)

    def synthesizer(self, settings):
        synth = irodori_server.IrodoriSynthesizer(settings)
        self.addCleanup(synth.close)
        return synth

    def test_settings_read_the_sokutou_environment(self) -> None:
        settings = self.settings(
            "http://127.0.0.1:9999/",
            IRODORI_TTS_SOKUTOU_STARTUP_TIMEOUT_SECONDS="30",
            IRODORI_TTS_SOKUTOU_TIMEOUT_SECONDS="7.5",
        )

        self.assertEqual(settings.backend, "sokutou")
        self.assertEqual(settings.sokutou_url, "http://127.0.0.1:9999")
        self.assertEqual(settings.sokutou_startup_timeout_seconds, 30.0)
        self.assertEqual(settings.sokutou_timeout_seconds, 7.5)

    def test_settings_default_to_the_local_container(self) -> None:
        settings = irodori_server.IrodoriSettings.from_env({"IRODORI_TTS_BACKEND": "sokutou", "IRODORI_TTS_NUM_STEPS": "12"})

        self.assertEqual(settings.sokutou_url, "http://127.0.0.1:8026")
        self.assertEqual(settings.sokutou_startup_timeout_seconds, 100.0)
        self.assertEqual(settings.sokutou_timeout_seconds, 60.0)

    def test_settings_refuse_what_sokutou_cannot_do(self) -> None:
        url = "http://127.0.0.1:9999"
        with self.assertRaisesRegex(ValueError, "caption"):
            self.settings(url, IRODORI_TTS_CAPTION="落ち着いた声で")
        with self.assertRaisesRegex(ValueError, "LoRA"):
            self.settings(url, IRODORI_TTS_LORA_ADAPTER="/tmp/adapter.safetensors")
        with self.assertRaisesRegex(ValueError, "IRODORI_TTS_NUM_STEPS"):
            self.settings(url, IRODORI_TTS_NUM_STEPS="")
        with self.assertRaisesRegex(ValueError, "IRODORI_TTS_SHORT_CACHE_PREWARM_NUM_STEPS"):
            self.settings(url, IRODORI_TTS_SHORT_CACHE_PREWARM_NUM_STEPS="20")
        with self.assertRaisesRegex(ValueError, "IRODORI_TTS_DURATION_SCALE"):
            self.settings(url, IRODORI_TTS_DURATION_SCALE="2.5")
        with self.assertRaisesRegex(ValueError, "IRODORI_TTS_SHORT_DURATION_SCALE"):
            self.settings(url, IRODORI_TTS_SHORT_DURATION_SCALE="0.4")

    def test_prewarm_steps_equal_to_the_engine_steps_are_accepted(self) -> None:
        settings = self.settings("http://127.0.0.1:9999", IRODORI_TTS_SHORT_CACHE_PREWARM_NUM_STEPS="12")

        self.assertEqual(settings.short_cache_prewarm_num_steps, 12)

    def test_short_text_gets_the_mora_cap_and_the_style_emoji_without_a_seed(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(self.settings(fake.url, IRODORI_TTS_STYLE_EMOJI="😊"))

        pcm, media_type, sample_rate = synth.synthesize("少々お待ちください。", response_format="pcm")

        normalized = irodori_server.normalize_speech_input("少々お待ちください。")
        (body,) = fake.bodies
        self.assertEqual(body["input"], "😊" + normalized)
        self.assertEqual(body["voice"], "kagawa")
        self.assertEqual(body["response_format"], "wav")
        self.assertEqual(body["seconds"], synth._max_seconds_for(normalized))
        self.assertNotIn("speed", body)
        self.assertNotIn("seed", body)
        self.assertEqual((media_type, sample_rate), ("audio/L16", 16000))
        # 0.5 s of 48 kHz audio comes back as at most 0.5 s of 16 kHz PCM after trimming.
        self.assertGreater(len(pcm), 0)
        self.assertLessEqual(len(pcm) // 2, 8000)

    def test_short_text_with_an_exempt_word_keeps_the_predicted_length(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(
            self.settings(fake.url, IRODORI_TTS_DURATION_CAP_EXEMPT_WORDS="ゆたか|ほか")
        )

        synth.synthesize("私は香川豊です。")
        synth.synthesize("少々お待ちください。")

        exempt, capped = fake.bodies
        self.assertEqual(exempt["input"], "私は香川ゆたかです。")
        self.assertNotIn("seconds", exempt)
        self.assertIn("seconds", capped)
        self.assertEqual(synth._max_seconds_for("私は香川ゆたかです。"), irodori_server.MAX_SYNTHESIS_SECONDS)

    def test_long_text_keeps_the_predicted_length(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(self.settings(fake.url))

        synth.synthesize("東京工科大学では、専門分野の知識だけでなく、実践的な学びを大切にしています。")

        (body,) = fake.bodies
        self.assertNotIn("seconds", body)
        self.assertNotIn("speed", body)

    def test_a_duration_scale_becomes_its_inverse_speed(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(self.settings(fake.url, IRODORI_TTS_DURATION_SCALE="1.25"))

        synth.synthesize("東京工科大学では、専門分野の知識だけでなく、実践的な学びを大切にしています。")

        (body,) = fake.bodies
        self.assertEqual(body["speed"], 0.8)
        self.assertNotIn("seconds", body)

    def test_the_rate_comes_from_the_wav_header(self) -> None:
        fake = self.start_fake(sample_rate=24000)
        settings = replace(self.settings(fake.url), trim_enabled=False, fade_ms=0.0)
        synth = self.synthesizer(settings)

        pcm, _media_type, sample_rate = synth.synthesize("東京工科大学では、実践的な学びを大切にしています。", response_format="pcm")

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(pcm) // 2, 8000)

    def test_start_waits_for_the_engine_to_warm_up(self) -> None:
        fake = self.start_fake(warming=3)
        synth = self.synthesizer(self.settings(fake.url))

        synth.load_runtime()

        self.assertEqual(fake.health_calls, 4)
        self.assertTrue(synth.model_loaded())

    def test_start_refuses_an_engine_with_other_steps(self) -> None:
        fake = self.start_fake(num_steps=20)
        synth = self.synthesizer(self.settings(fake.url))

        with self.assertRaisesRegex(RuntimeError, r"20 steps.*IRODORI_TTS_NUM_STEPS=12"):
            synth.load_runtime()
        self.assertFalse(synth.model_loaded())

    def test_start_gives_up_after_the_startup_timeout(self) -> None:
        fake = self.start_fake(warming=10**6)
        synth = self.synthesizer(self.settings(fake.url, IRODORI_TTS_SOKUTOU_STARTUP_TIMEOUT_SECONDS="0.3"))

        with self.assertRaisesRegex(RuntimeError, rf"{fake.url}.*0.3 s"):
            synth.load_runtime()

    def test_an_unreachable_engine_names_its_url(self) -> None:
        url = closed_port_url()
        synth = self.synthesizer(self.settings(url, IRODORI_TTS_SOKUTOU_STARTUP_TIMEOUT_SECONDS="0.2"))

        with self.assertRaisesRegex(RuntimeError, url):
            synth.synthesize("こんにちは。")

    def test_a_refused_input_is_a_value_error_with_the_engine_detail(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(self.settings(fake.url))

        with self.assertRaisesRegex(ValueError, "longer than 200 characters"):
            synth.synthesize("TOOLONG")

    def test_an_implausible_duration_is_retried_once(self) -> None:
        fake = self.start_fake(babbles=1)
        synth = self.synthesizer(self.settings(fake.url))

        pcm, _media_type, _rate = synth.synthesize("BABBLE")

        self.assertEqual(len(fake.bodies), 2)
        self.assertGreater(len(pcm), 0)

    def test_a_second_implausible_duration_fails_with_the_detail(self) -> None:
        fake = self.start_fake(babbles=2)
        synth = self.synthesizer(self.settings(fake.url))

        with self.assertRaisesRegex(RuntimeError, "implausible duration"):
            synth.synthesize("BABBLE")
        self.assertEqual(len(fake.bodies), 2)

    def test_other_engine_errors_are_not_retried(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(self.settings(fake.url))

        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            synth.synthesize("BROKEN")
        self.assertEqual(len(fake.bodies), 1)

    def test_a_request_for_other_steps_is_refused_before_sending(self) -> None:
        fake = self.start_fake()
        synth = self.synthesizer(self.settings(fake.url))

        with self.assertRaisesRegex(ValueError, "12"):
            synth._synthesize_normalized("こんにちは。", response_format="pcm", num_steps=20)
        self.assertEqual(fake.bodies, [])

    @unittest.skipIf(fastapi is None, "fastapi is not installed")
    def test_healthz_reports_the_engine(self) -> None:
        from fastapi.testclient import TestClient

        fake = self.start_fake()
        settings = self.settings(fake.url)
        synth = self.synthesizer(settings)
        client = TestClient(irodori_server.create_app(settings, synth))

        before = client.get("/healthz").json()
        response = client.post("/v1/audio/speech", json={"input": "こんにちは。"})
        after = client.get("/healthz").json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Sample-Rate-Hz"], "16000")
        self.assertEqual(before["backend"], "sokutou")
        self.assertFalse(before["model_loaded"])
        self.assertTrue(after["model_loaded"])
        self.assertEqual(after["sokutou_url"], fake.url)
        self.assertEqual(after["sokutou_num_steps"], 12)
        self.assertFalse(after["watermark"])
        self.assertIsNone(after["speechcpp_device"])
        self.assertIsNone(after["caption"])

    @unittest.skipIf(fastapi is None, "fastapi is not installed")
    def test_a_refused_input_answers_400(self) -> None:
        from fastapi.testclient import TestClient

        fake = self.start_fake()
        settings = self.settings(fake.url)
        client = TestClient(irodori_server.create_app(settings, self.synthesizer(settings)))

        response = client.post("/v1/audio/speech", json={"input": "TOOLONG"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("longer than 200 characters", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
