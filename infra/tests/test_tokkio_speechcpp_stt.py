from __future__ import annotations

import asyncio
import enum
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Language(str, enum.Enum):
    JA = "ja"
    EN = "en"


class FakeTranscriptions:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return FakeTranscription(FakeOpenAISTTService.next_text)


class FakeAsyncOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.audio = types.SimpleNamespace(transcriptions=FakeTranscriptions())


class FakeTranscription:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeOpenAISTTService:
    """pipecat 0.0.68's OpenAISTTService, down to what the subclass relies on."""

    next_text = ""
    super_calls = 0

    def __init__(self, *, model, api_key=None, base_url=None, language=Language.EN, prompt=None, temperature=None, **kwargs):
        self.model_name = model
        self.init_kwargs = {"api_key": api_key, "base_url": base_url, "language": language, "prompt": prompt, **kwargs}
        self._client = self._create_client(api_key, base_url)
        # BaseWhisperSTTService keeps the Whisper code ("ja"), not the enum.
        self._language = language.value
        self._prompt = prompt
        self._temperature = temperature
        # SegmentedSTTService: the audio of the segment, with up to a second before it.
        self._audio_buffer = bytearray()
        self.sample_rate = 16000
        self.transcribed: list[bytes] = []

    def _create_client(self, api_key, base_url):
        raise AssertionError("the subclass must build its own client")

    async def _handle_user_stopped_speaking(self, frame) -> None:
        if frame.emulated:
            return
        self.transcribed.append(bytes(self._audio_buffer))
        self._audio_buffer.clear()

    async def _transcribe(self, audio: bytes) -> FakeTranscription:
        FakeOpenAISTTService.super_calls += 1
        return FakeTranscription(FakeOpenAISTTService.next_text)


def load_tokkio_speechcpp_stt():
    class FakeLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def info(self, message, *args, **kwargs) -> None:
            self.infos.append(message)

    openai_module = types.ModuleType("openai")
    openai_module.AsyncOpenAI = FakeAsyncOpenAI
    loguru_module = types.ModuleType("loguru")
    loguru_module.logger = FakeLogger()
    stt_module = types.ModuleType("pipecat.services.openai.stt")
    stt_module.OpenAISTTService = FakeOpenAISTTService
    language_module = types.ModuleType("pipecat.transcriptions.language")
    language_module.Language = Language

    stubs = {
        "openai": openai_module,
        "loguru": loguru_module,
        "pipecat": types.ModuleType("pipecat"),
        "pipecat.services": types.ModuleType("pipecat.services"),
        "pipecat.services.openai": types.ModuleType("pipecat.services.openai"),
        "pipecat.services.openai.stt": stt_module,
        "pipecat.transcriptions": types.ModuleType("pipecat.transcriptions"),
        "pipecat.transcriptions.language": language_module,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "tokkio_speechcpp_stt_under_test", ROOT / "infra" / "tokkio" / "tokkio_speechcpp_stt.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load tokkio_speechcpp_stt.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class SpeechCppSTTServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_tokkio_speechcpp_stt()
        FakeOpenAISTTService.next_text = ""
        FakeOpenAISTTService.super_calls = 0

    def make_service(self, **overrides):
        kwargs = {
            "base_url": "http://10.209.1.12:8024/v1",
            "model": "Qwen3-ASR-1.7B",
            "language": "ja",
            "prompt": "青嵐、SEIRAN",
            "timeout_s": 10.0,
        }
        kwargs.update(overrides)
        return self.module.SpeechCppSTTService(**kwargs)

    def test_repeats_prompt_matches_only_the_whole_prompt(self) -> None:
        repeats = self.module.repeats_prompt
        # What Qwen3-ASR wrote for all six silence and noise probes.
        self.assertTrue(repeats("青嵐、SEIRAN。", "青嵐、SEIRAN"))
        self.assertTrue(repeats("青嵐 SEIRAN", "青嵐、SEIRAN"))
        self.assertFalse(repeats("青嵐について教えてください。", "青嵐、SEIRAN"))
        self.assertFalse(repeats("青嵐。", "青嵐、SEIRAN"))
        self.assertFalse(repeats("はい。", "青嵐、SEIRAN"))
        self.assertFalse(repeats("", "青嵐、SEIRAN"))
        self.assertFalse(repeats("青嵐、SEIRAN。", ""))
        self.assertFalse(repeats("青嵐、SEIRAN。", None))

    def test_service_talks_to_the_local_server_without_a_key_or_retries(self) -> None:
        service = self.make_service()

        self.assertEqual(service.model_name, "Qwen3-ASR-1.7B")
        self.assertEqual(service.init_kwargs["base_url"], "http://10.209.1.12:8024/v1")
        self.assertEqual(service.init_kwargs["language"], Language.JA)
        self.assertEqual(service.init_kwargs["prompt"], "青嵐、SEIRAN")
        # AsyncOpenAI refuses to start without a key; speech serve ignores it.
        self.assertTrue(service.init_kwargs["api_key"])
        self.assertEqual(service._client.kwargs["base_url"], "http://10.209.1.12:8024/v1")
        self.assertEqual(service._client.kwargs["max_retries"], 0)
        self.assertEqual(service._client.kwargs["timeout"], 10.0)

    def test_empty_prompt_is_not_sent(self) -> None:
        # FastConformer models and English mode take no prompt; speech serve refuses one they do not take.
        service = self.make_service(language="en", prompt="")

        self.assertIsNone(service.init_kwargs["prompt"])
        self.assertEqual(service.init_kwargs["language"], Language.EN)

    def test_transcript_that_repeats_the_prompt_is_dropped(self) -> None:
        service = self.make_service()
        FakeOpenAISTTService.next_text = "青嵐、SEIRAN。"

        result = asyncio.run(service._transcribe(b"wav"))

        self.assertEqual(result.text, "")
        self.assertTrue(any("青嵐、SEIRAN。" in message for message in self.module.logger.infos))

    def stop_after(self, service, audio: bytes, *, emulated: bool = False) -> None:
        service._audio_buffer += audio
        asyncio.run(service._handle_user_stopped_speaking(types.SimpleNamespace(emulated=emulated)))

    def test_each_segment_is_transcribed_alone_by_default(self) -> None:
        service = self.make_service()

        self.stop_after(service, b"aa")
        self.stop_after(service, b"bb")
        service.reset_turn_audio()

        self.assertEqual(service.transcribed, [b"aa", b"bb"])

    def test_turn_audio_sends_the_whole_turn_until_it_is_reset(self) -> None:
        service = self.make_service()
        service.enable_turn_audio()

        self.stop_after(service, b"aa")
        self.stop_after(service, b"bb")
        service.reset_turn_audio()
        self.stop_after(service, b"cc")

        self.assertEqual(service.transcribed, [b"aa", b"aabb", b"cc"])

    def test_turn_audio_over_the_limit_starts_again_from_the_last_segment(self) -> None:
        service = self.make_service()
        # One second of 16 kHz 16-bit audio is 32000 bytes.
        service.enable_turn_audio(max_seconds=1.0)

        self.stop_after(service, b"a" * 20000)
        self.stop_after(service, b"b" * 20000)

        self.assertEqual(service.transcribed, [b"a" * 20000, b"b" * 20000])

    def test_emulated_stop_leaves_the_turn_audio_alone(self) -> None:
        service = self.make_service()
        service.enable_turn_audio()

        self.stop_after(service, b"aa", emulated=True)

        self.assertEqual(service.transcribed, [])
        self.assertEqual(bytes(service._audio_buffer), b"aa")

    def test_other_transcripts_pass_through(self) -> None:
        service = self.make_service()
        FakeOpenAISTTService.next_text = "青嵐について教えてください。"

        result = asyncio.run(service._transcribe(b"wav"))

        self.assertEqual(result.text, "青嵐について教えてください。")



    def test_chunking_strategy_goes_in_the_form_with_pipecats_fields(self) -> None:
        # Haas: the server's VAD drops noise, which Haas otherwise writes as 「うん。」.
        service = self.make_service(model="Haas-ja-v1.0", prompt="", chunking_strategy="auto")
        FakeOpenAISTTService.next_text = "青嵐について教えてください"

        result = asyncio.run(service._transcribe(b"wav"))

        self.assertEqual(result.text, "青嵐について教えてください")
        self.assertEqual(FakeOpenAISTTService.super_calls, 0)
        (request,) = service._client.audio.transcriptions.requests
        # extra_body, not the SDK's own argument: the controller image's SDK version is unknown.
        self.assertEqual(
            request,
            {
                "file": ("audio.wav", b"wav", "audio/wav"),
                "model": "Haas-ja-v1.0",
                "language": "ja",
                "extra_body": {"chunking_strategy": "auto"},
            },
        )

    def test_chunking_strategy_keeps_a_set_prompt(self) -> None:
        service = self.make_service(chunking_strategy="auto")
        FakeOpenAISTTService.next_text = "青嵐、SEIRAN。"

        result = asyncio.run(service._transcribe(b"wav"))

        self.assertEqual(service._client.audio.transcriptions.requests[0]["prompt"], "青嵐、SEIRAN")
        self.assertEqual(result.text, "")

    def test_without_chunking_strategy_pipecats_request_is_used(self) -> None:
        # The English mode (Qwen3-ASR) sends what it sent before.
        service = self.make_service(model="Qwen3-ASR-1.7B", language="en", prompt="")
        FakeOpenAISTTService.next_text = "Hello."

        result = asyncio.run(service._transcribe(b"wav"))

        self.assertEqual(result.text, "Hello.")
        self.assertEqual(FakeOpenAISTTService.super_calls, 1)
        self.assertEqual(service._client.audio.transcriptions.requests, [])


if __name__ == "__main__":
    unittest.main()
