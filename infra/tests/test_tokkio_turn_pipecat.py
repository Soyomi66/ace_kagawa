"""TurnGate inside a real pipecat pipeline (interruptions, context aggregators).

Skipped where pipecat is not installed. The controller runs pipecat-ai 0.0.68:
    python3 -m venv /tmp/pipecat-venv && /tmp/pipecat-venv/bin/pip install pipecat-ai==0.0.68
    /tmp/pipecat-venv/bin/python -m unittest infra.tests.test_tokkio_turn_pipecat
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

try:
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        EndFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        StartInterruptionFrame,
        TextFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_response import (
        LLMAssistantContextAggregator,
        LLMUserAggregatorParams,
        LLMUserContextAggregator,
    )
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext, OpenAILLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.transports.base_input import BaseInputTransport
    from pipecat.transports.base_transport import TransportParams

    HAS_PIPECAT = True
except ImportError:
    HAS_PIPECAT = False
    FrameProcessor = object
    BaseInputTransport = object


def load_tokkio_turn():
    spec = importlib.util.spec_from_file_location("tokkio_turn_pipecat_test", ROOT / "infra" / "tokkio" / "tokkio_turn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_module(name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "infra" / "tokkio" / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_generated_config_model(customize):
    """Execute CONFIG_PY the way the controller imports src/config.py (nvidia_pipecat stubbed)."""
    import types

    import pydantic

    class AnimationGraphConfiguration(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="allow")

    animation_graph_module = types.ModuleType("nvidia_pipecat.services.animation_graph_service")
    animation_graph_module.AnimationGraphConfiguration = AnimationGraphConfiguration
    module = types.ModuleType("generated_tokkio_config_for_turn_test")
    sys.modules.setdefault("nvidia_pipecat", types.ModuleType("nvidia_pipecat"))
    sys.modules.setdefault("nvidia_pipecat.services", types.ModuleType("nvidia_pipecat.services"))
    sys.modules["nvidia_pipecat.services.animation_graph_service"] = animation_graph_module
    sys.modules[module.__name__] = module
    exec(compile(customize.CONFIG_PY, "config.py", "exec"), module.__dict__)
    return module


VAD_STOP = 0.1
MAX_WAIT = 0.8
WEAK_WAIT = 0.45
MIN_SPEECH = 0.3
# This host's CPU is busy; timers may fire this much late.
SLACK = 0.3


class FakeSTT(FrameProcessor):
    """Like SegmentedSTTService: the transcript of a segment follows its UserStoppedSpeakingFrame."""

    def __init__(self):
        super().__init__()
        self.next_text = ""
        self.turn_resets = 0

    def reset_turn_audio(self):
        self.turn_resets += 1

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame) and self.next_text:
            text, self.next_text = self.next_text, ""
            await self.push_frame(TranscriptionFrame(text, "", "now"))


class FakeLLM(FrameProcessor):
    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay
        self.requests: list[tuple[float, list[str]]] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, OpenAILLMContextFrame):
            users = [m["content"] for m in frame.context.get_messages() if m["role"] == "user"]
            self.requests.append((time.monotonic(), users))
            await asyncio.sleep(self.delay)
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(f"reply to {users[-1]}"))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class FakeAvatar(FrameProcessor):
    """TTS and the animation service: speaks a text for speak_s, reports start and stop both ways."""

    def __init__(self, speak_s: float):
        super().__init__()
        self.speak_s = speak_s
        self.speaking = False
        self.spoken: list[str] = []
        self.interrupted_at: list[float] = []

    async def _report(self, frame_type):
        await self.push_frame(frame_type())
        await self.push_frame(frame_type(), FrameDirection.UPSTREAM)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartInterruptionFrame) and self.speaking:
            self.speaking = False
            self.interrupted_at.append(time.monotonic())
            await self._report(BotStoppedSpeakingFrame)
        if isinstance(frame, TextFrame):
            self.speaking = True
            self.spoken.append(frame.text)
            await self._report(BotStartedSpeakingFrame)
            await self.push_frame(frame, direction)
            await asyncio.sleep(self.speak_s)
            self.speaking = False
            await self._report(BotStoppedSpeakingFrame)
        else:
            await self.push_frame(frame, direction)


class FakeInput(BaseInputTransport):
    def __init__(self):
        super().__init__(TransportParams())


@unittest.skipUnless(HAS_PIPECAT, "pipecat is not installed")
class TurnGatePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.turn = load_tokkio_turn()

    async def start(self, *, text=True, gated=False, llm_delay=0.05, speak_s=0.2):
        self.input = FakeInput()
        self.stt = FakeSTT()
        self.context = OpenAILLMContext([{"role": "system", "content": "s"}])
        user = LLMUserContextAggregator(self.context, params=LLMUserAggregatorParams(aggregation_timeout=0.02))
        assistant = LLMAssistantContextAggregator(self.context)
        self.llm = FakeLLM(llm_delay)
        self.avatar = FakeAvatar(speak_s)
        turn_taking = SimpleNamespace(
            turn_end_mode="text" if text else "vad",
            barge_in_mode="gated" if gated else "immediate",
            vad_stop_secs_in_use=VAD_STOP,
            turn_end_retranscribe=False,
            turn_end_max_wait_s=MAX_WAIT,
            turn_end_weak_wait_s=WEAK_WAIT,
            turn_end_merge_window_s=2.0,
            turn_end_hold_phrases=["そうですね"],
            # build_turn_gate adds vad_stop_secs - 0.2 to it.
            barge_in_min_speech_s=MIN_SPEECH + 0.2 - VAD_STOP,
            barge_in_backchannels=list(self.turn.DEFAULT_BACKCHANNELS),
        )
        transport = SimpleNamespace(input=lambda: self.input)
        gate = self.turn.build_turn_gate(
            turn_taking, transport=transport, stt=self.stt, context=self.context, user_aggregator=user
        )
        processors = [self.input, self.stt, gate, user, self.llm, self.avatar, assistant]
        self.task = PipelineTask(
            Pipeline([p for p in processors if p is not None]),
            params=PipelineParams(allow_interruptions=True),
            cancel_on_idle_timeout=False,
        )
        self.runner_task = asyncio.create_task(PipelineRunner(handle_sigint=False).run(self.task))
        await asyncio.sleep(0.1)
        self.t0 = time.monotonic()

    async def finish(self):
        await self.task.queue_frame(EndFrame())
        await asyncio.wait_for(self.runner_task, timeout=5)

    async def say(self, text: str, duration: float = 0.05):
        """A speech segment as the VAD reports it: started, then stopped with its transcript."""
        await self.input._handle_user_interruption(UserStartedSpeakingFrame())
        await asyncio.sleep(duration)
        self.stt.next_text = text
        self.stopped_at = time.monotonic()
        await self.input._handle_user_interruption(UserStoppedSpeakingFrame())

    def user_messages(self):
        return [m["content"] for m in self.context.get_messages() if m["role"] == "user"]

    async def test_finished_sentence_goes_to_the_llm_at_once(self):
        await self.start()
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.5)
        await self.finish()
        self.assertEqual([users for _, users in self.llm.requests], [["学費はいくらですか"]])
        self.assertLess(self.llm.requests[0][0] - self.stopped_at, SLACK)
        self.assertEqual(self.avatar.spoken, ["reply to 学費はいくらですか"])

    async def test_unfinished_sentence_waits_for_the_silence_limit(self):
        await self.start()
        await self.say("東京工科大学の")
        await asyncio.sleep(MAX_WAIT + 0.5)
        await self.finish()
        self.assertEqual([users for _, users in self.llm.requests], [["東京工科大学の"]])
        waited = self.llm.requests[0][0] - self.stopped_at
        self.assertGreater(waited, MAX_WAIT - VAD_STOP - 0.02)
        self.assertLess(waited, MAX_WAIT - VAD_STOP + SLACK)

    async def test_plain_form_waits_the_shorter_time(self):
        await self.start()
        await self.say("AIを使った")
        await asyncio.sleep(MAX_WAIT + 0.3)
        await self.finish()
        waited = self.llm.requests[0][0] - self.stopped_at
        self.assertGreater(waited, WEAK_WAIT - VAD_STOP - 0.02)
        self.assertLess(waited, MAX_WAIT - VAD_STOP)

    async def test_held_text_is_joined_with_what_follows(self):
        await self.start()
        await self.say("東京工科大学の")
        await asyncio.sleep(0.1)
        await self.say("行き方を教えてください")
        await asyncio.sleep(0.6)
        await self.finish()
        self.assertEqual([users for _, users in self.llm.requests], [["東京工科大学の 行き方を教えてください"]])
        self.assertEqual(len(self.avatar.spoken), 1)

    async def test_reply_is_dropped_and_the_turn_merged_when_the_visitor_goes_on(self):
        await self.start(llm_delay=0.4)
        await self.say("学費を教えてください")
        await asyncio.sleep(0.15)
        await self.say("入学金もお願いします")
        await asyncio.sleep(1.0)
        await self.finish()
        self.assertEqual(
            [users for _, users in self.llm.requests],
            [["学費を教えてください"], ["学費を教えてください 入学金もお願いします"]],
        )
        self.assertEqual(self.avatar.spoken, ["reply to 学費を教えてください 入学金もお願いします"])
        self.assertEqual(self.user_messages(), ["学費を教えてください 入学金もお願いします"])

    async def test_short_reply_waits_and_is_joined_with_what_follows(self):
        await self.start()
        await self.say("はい")
        await asyncio.sleep(0.05)
        await self.say("お願いします")
        await asyncio.sleep(0.6)
        await self.finish()
        self.assertEqual([users for _, users in self.llm.requests], [["はい お願いします"]])

    async def test_noise_before_the_reply_does_not_lose_the_turn(self):
        await self.start(llm_delay=0.4)
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.15)
        await self.say("")
        await asyncio.sleep(MAX_WAIT + 1.0)
        await self.finish()
        self.assertEqual(self.avatar.spoken, ["reply to 学費はいくらですか"])
        self.assertEqual(self.user_messages(), ["学費はいくらですか"])

    async def test_speech_after_the_reply_started_is_a_new_turn(self):
        await self.start(speak_s=0.3)
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.2)
        self.assertTrue(self.avatar.speaking)
        await self.say("入学金はいくらですか")
        await asyncio.sleep(0.8)
        await self.finish()
        self.assertEqual(self.user_messages(), ["学費はいくらですか", "入学金はいくらですか"])
        self.assertEqual(len(self.avatar.interrupted_at), 1)
        self.assertGreaterEqual(self.stt.turn_resets, 1)

    async def test_backchannel_does_not_stop_the_avatar(self):
        await self.start(gated=True, speak_s=0.8)
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.2)
        self.assertTrue(self.avatar.speaking)
        await self.say("うん", duration=0.1)
        await asyncio.sleep(1.0)
        await self.finish()
        self.assertEqual(self.avatar.interrupted_at, [])
        self.assertEqual(len(self.llm.requests), 1)
        self.assertEqual(self.user_messages(), ["学費はいくらですか"])

    async def test_noise_does_not_stop_the_avatar(self):
        await self.start(gated=True, speak_s=0.8)
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.2)
        await self.say("", duration=0.1)
        await asyncio.sleep(1.0)
        await self.finish()
        self.assertEqual(self.avatar.interrupted_at, [])
        self.assertEqual(len(self.llm.requests), 1)

    async def test_short_real_speech_stops_the_avatar_on_its_transcript(self):
        await self.start(gated=True, speak_s=2.0)
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.2)
        await self.say("ちょっと待ってください", duration=0.1)
        await asyncio.sleep(0.6)
        await self.finish()
        self.assertEqual(len(self.avatar.interrupted_at), 1)
        self.assertLess(self.avatar.interrupted_at[0] - self.stopped_at, SLACK)
        self.assertEqual([users[-1] for _, users in self.llm.requests], ["学費はいくらですか", "ちょっと待ってください"])

    async def test_long_speech_stops_the_avatar_before_it_ends(self):
        await self.start(gated=True, speak_s=2.0)
        await self.say("学費はいくらですか")
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await self.say("すみません、入学金はいくらですか", duration=MIN_SPEECH + 0.6)
        await asyncio.sleep(0.6)
        await self.finish()
        self.assertEqual(len(self.avatar.interrupted_at), 1)
        stopped_after = self.avatar.interrupted_at[0] - started
        self.assertGreater(stopped_after, MIN_SPEECH - 0.02)
        self.assertLess(stopped_after, MIN_SPEECH + SLACK)
        self.assertEqual(self.llm.requests[-1][1][-1], "すみません、入学金はいくらですか")

    async def test_gated_barge_in_without_text_turn_end_passes_transcripts_through(self):
        await self.start(text=False, gated=True)
        await self.say("東京工科大学の")
        await asyncio.sleep(0.3)
        await self.finish()
        self.assertEqual([users for _, users in self.llm.requests], [["東京工科大学の"]])
        self.assertLess(self.llm.requests[0][0] - self.stopped_at, SLACK)

    async def test_generated_config_and_speechcpp_stt_send_the_whole_turn(self):
        """config.yaml -> config.py -> build_turn_gate with the real SpeechCppSTTService."""
        import yaml
        from pipecat.frames.frames import InputAudioRawFrame

        customize = load_module("customize_for_pipecat_test", "customize_tokkio_japanese.py")
        stt_module = load_module("speechcpp_stt_for_pipecat_test", "tokkio_speechcpp_stt.py")
        config_module = load_generated_config_model(customize)
        text = customize.build_config_yaml(turn_end_mode="text", barge_in_mode="gated")
        turn_taking = config_module.Config(**yaml.safe_load(text)).TurnTaking
        self.assertEqual(turn_taking.vad_stop_secs_in_use, 0.2)

        requests: list[int] = []
        replies = ["東京工科大学の", "東京工科大学の行き方を教えてください", "入学金はいくらですか"]

        class ScriptedSTT(stt_module.SpeechCppSTTService):
            async def _transcribe(self, audio: bytes):
                requests.append(len(audio))
                return SimpleNamespace(text=replies[len(requests) - 1])

        self.input = FakeInput()
        stt = ScriptedSTT(base_url="http://127.0.0.1:9/v1", model="Haas-ja-v1.0", sample_rate=16000)
        self.context = OpenAILLMContext([{"role": "system", "content": "s"}])
        user = LLMUserContextAggregator(self.context, params=LLMUserAggregatorParams(aggregation_timeout=0.02))
        self.llm = FakeLLM(0.05)
        self.avatar = FakeAvatar(0.2)
        gate = self.turn.build_turn_gate(
            turn_taking,
            transport=SimpleNamespace(input=lambda: self.input),
            stt=stt,
            context=self.context,
            user_aggregator=user,
        )
        self.assertTrue(gate._whole_turn_transcripts)
        self.task = PipelineTask(
            Pipeline([self.input, stt, gate, user, self.llm, self.avatar, LLMAssistantContextAggregator(self.context)]),
            params=PipelineParams(allow_interruptions=True),
            cancel_on_idle_timeout=False,
        )
        self.runner_task = asyncio.create_task(PipelineRunner(handle_sigint=False).run(self.task))
        await asyncio.sleep(0.1)

        async def speak(seconds: float):
            await self.input._handle_user_interruption(UserStartedSpeakingFrame())
            audio = InputAudioRawFrame(audio=b"\0" * int(32000 * seconds), sample_rate=16000, num_channels=1)
            await self.input.push_frame(audio)
            await self.input._handle_user_interruption(UserStoppedSpeakingFrame())

        # Two segments without a transcript (noise) must not stay in front of the turn.
        replies[:0] = ["", ""]
        await speak(0.5)
        await asyncio.sleep(0.1)
        await speak(0.5)
        await asyncio.sleep(0.1)
        await speak(0.5)
        await asyncio.sleep(0.1)
        await speak(0.5)
        await asyncio.sleep(1.2)
        await speak(0.5)
        await asyncio.sleep(1.0)
        await self.finish()

        del requests[:2]
        self.assertLess(requests[0], 20000)
        # The second request carries both segments; after the reply the turn starts again.
        self.assertGreater(requests[1], requests[0] + 15000)
        self.assertLess(requests[2], requests[0] + 1000)
        self.assertEqual(self.user_messages(), ["東京工科大学の行き方を教えてください", "入学金はいくらですか"])
        self.assertEqual(len(self.avatar.spoken), 2)

    async def test_no_gate_when_both_switches_are_off(self):
        turn_taking = SimpleNamespace(turn_end_mode="vad", barge_in_mode="immediate")
        self.assertIsNone(
            self.turn.build_turn_gate(turn_taking, transport=None, stt=None, context=None, user_aggregator=None)
        )


if __name__ == "__main__":
    unittest.main()
