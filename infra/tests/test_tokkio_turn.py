"""tokkio_turn.py: the text rules, and the turn gate on stand-ins for pipecat's classes.

test_tokkio_turn_pipecat.py runs the gate in a real pipecat pipeline where pipecat is installed.
"""

from __future__ import annotations

import asyncio
import enum
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class Frame:
    def __init__(self, emulated: bool = False) -> None:
        self.emulated = emulated


FRAME_NAMES = (
    "BotInterruptionFrame",
    "BotStartedSpeakingFrame",
    "BotStoppedSpeakingFrame",
    "CancelFrame",
    "EndFrame",
    "UserStartedSpeakingFrame",
    "UserStoppedSpeakingFrame",
)


class TranscriptionFrame(Frame):
    def __init__(self, text, user_id="", timestamp="", language=None) -> None:
        super().__init__()
        self.text = text
        self.user_id = user_id
        self.timestamp = timestamp
        self.language = language


class FrameDirection(enum.Enum):
    DOWNSTREAM = 1
    UPSTREAM = 2


class FakeFrameProcessor:
    def __init__(self, **kwargs) -> None:
        self.pushed: list[tuple[object, FrameDirection]] = []

    async def process_frame(self, frame, direction) -> None:
        pass

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM) -> None:
        self.pushed.append((frame, direction))


def load_tokkio_turn():
    frames_module = types.ModuleType("pipecat.frames.frames")
    for name in FRAME_NAMES:
        setattr(frames_module, name, type(name, (Frame,), {}))
    frames_module.TranscriptionFrame = TranscriptionFrame
    processor_module = types.ModuleType("pipecat.processors.frame_processor")
    processor_module.FrameDirection = FrameDirection
    processor_module.FrameProcessor = FakeFrameProcessor
    loguru_module = types.ModuleType("loguru")
    loguru_module.logger = SimpleNamespace(
        info=lambda *a, **k: None, debug=lambda *a, **k: None, error=lambda *a, **k: None
    )
    stubs = {
        "loguru": loguru_module,
        "pipecat": types.ModuleType("pipecat"),
        "pipecat.frames": types.ModuleType("pipecat.frames"),
        "pipecat.frames.frames": frames_module,
        "pipecat.processors": types.ModuleType("pipecat.processors"),
        "pipecat.processors.frame_processor": processor_module,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("tokkio_turn_under_test", ROOT / "infra" / "tokkio" / "tokkio_turn.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.frames = frames_module
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class FakeContext:
    def __init__(self, messages) -> None:
        self.messages = list(messages)

    def get_messages(self):
        return self.messages

    def set_messages(self, messages) -> None:
        self.messages = list(messages)


class TurnEndRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.turn = load_tokkio_turn()

    def assert_endings(self, expected: str, texts) -> None:
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(self.turn.classify_turn_end(text), expected)

    def test_polite_sentence_ends_are_complete(self) -> None:
        self.assert_endings(
            "complete",
            [
                "オープンキャンパスはいつですか?",
                "青嵐について教えてください",
                "青嵐について教えて下さい。",
                "研究室を見学したいです",
                "わかりました。",
                "ありがとうございます",
                "学費は高いですよね",
                "はい、そうです",
                "ハイ、お願いシマス",
            ],
        )

    def test_the_asr_full_stop_does_not_close_an_open_phrase(self) -> None:
        # Qwen3-ASR ends every transcript with 。
        self.assert_endings(
            "open",
            [
                "東京工科大学の。",
                "東京工科大学の",
                "行き方なんですけど、",
                "行き方なんですけど",
                "えっと",
                "あの。",
                "学費は",
                "八王子みなみ野駅から",
                "そうですね、",
                "",
                "。",
            ],
        )

    def test_plain_forms_and_short_replies_may_go_on(self) -> None:
        self.assert_endings(
            "weak",
            [
                "AIを使った",
                "研究室を見学したい",
                "予定していた。",
                "はい",
                "ハイ",
                "いいえ。",
                "なるほど",
                "ありがとう",
                # The 2026-11-02 rule read the last は and も as particles.
                "こんにちは",
                "こんばんは。",
                "どうも",
            ],
        )

    def test_hold_phrases_wait_for_the_silence_limit(self) -> None:
        self.assert_endings("open", ["そうですね", "そうですね。", "すみません", "あ、すみません"])
        self.assertEqual(self.turn.classify_turn_end("そうですね", hold_phrases=()), "complete")
        self.assertEqual(self.turn.classify_turn_end("はい", hold_phrases=("はい",)), "open")

    def test_backchannels(self) -> None:
        for text in ["うん", "うん。", "うん、うん", "はい", "ハイ", "へー", "へえ。", "ふーん", "なるほど", "そうなんですね", "はい、はい。"]:
            with self.subTest(text=text):
                self.assertTrue(self.turn.is_backchannel(text))
        for text in ["", "。", "はい、お願いします", "ちょっと待ってください", "ストップ", "違います", "うん、うん、うん、うん", "学費は"]:
            with self.subTest(text=text):
                self.assertFalse(self.turn.is_backchannel(text))
        self.assertTrue(self.turn.is_backchannel("すごい", backchannels=("すごい",)))


class TurnGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.turn = load_tokkio_turn()
        self.frames = self.turn.frames
        self.now = 100.0
        self.context = FakeContext([{"role": "system", "content": "s"}])
        self.aggregator = SimpleNamespace(resets=0)
        self.aggregator.reset = lambda: setattr(self.aggregator, "resets", self.aggregator.resets + 1)
        self.stt = SimpleNamespace(resets=0)
        self.stt.reset_turn_audio = lambda: setattr(self.stt, "resets", self.stt.resets + 1)

    def make_gate(self, **overrides):
        kwargs = {
            "text_turn_end": True,
            "vad_stop_secs": 0.2,
            "max_wait_s": 0.3,
            "weak_wait_s": 0.25,
            "barge_in_min_speech_s": 0.05,
            "context": self.context,
            "user_aggregator": self.aggregator,
            "stt": self.stt,
            "clock": lambda: self.now,
        }
        kwargs.update(overrides)
        return self.turn.TurnGate(**kwargs)

    async def send(self, gate, name: str) -> None:
        await gate.process_frame(getattr(self.frames, name)(), FrameDirection.DOWNSTREAM)

    async def transcript(self, gate, text: str) -> None:
        await gate.process_frame(TranscriptionFrame(text, "u", "t"), FrameDirection.DOWNSTREAM)

    async def say(self, gate, text: str) -> None:
        await self.send(gate, "UserStartedSpeakingFrame")
        await self.send(gate, "UserStoppedSpeakingFrame")
        if text:
            await self.transcript(gate, text)

    def released(self, gate) -> list[str]:
        return [frame.text for frame, _ in gate.pushed if isinstance(frame, TranscriptionFrame)]

    def interruptions(self, gate) -> int:
        return sum(
            isinstance(frame, self.frames.BotInterruptionFrame) and direction == FrameDirection.UPSTREAM
            for frame, direction in gate.pushed
        )

    async def test_complete_text_is_released_with_the_frame_fields(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "学費はいくらですか")
        frame = next(frame for frame, _ in gate.pushed if isinstance(frame, TranscriptionFrame))
        self.assertEqual((frame.text, frame.user_id, frame.timestamp), ("学費はいくらですか", "u", "t"))

    async def test_open_text_is_released_when_the_silence_reaches_the_limit(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "東京工科大学の")
        self.assertEqual(self.released(gate), [])
        await asyncio.sleep(0.05)
        self.assertEqual(self.released(gate), [])
        await asyncio.sleep(0.1)
        self.assertEqual(self.released(gate), ["東京工科大学の"])

    async def test_new_speech_cancels_the_wait_and_segments_are_joined(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "東京工科大学の")
        await self.say(gate, "行き方を教えてください")
        await asyncio.sleep(0.15)
        self.assertEqual(self.released(gate), ["東京工科大学の 行き方を教えてください"])

    async def test_whole_turn_transcripts_replace_the_held_text(self) -> None:
        gate = self.make_gate(whole_turn_transcripts=True)
        await self.say(gate, "東京工科大学の")
        await self.say(gate, "東京工科大学の行き方を教えてください")
        self.assertEqual(self.released(gate), ["東京工科大学の行き方を教えてください"])

    async def test_held_text_goes_out_when_the_next_segment_has_no_transcript(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "東京工科大学の")
        await self.say(gate, "")
        await asyncio.sleep(0.15)
        self.assertEqual(self.released(gate), ["東京工科大学の"])

    async def test_new_turn_starts_with_fresh_turn_audio(self) -> None:
        gate = self.make_gate()
        # Noise: segments without a transcript would otherwise stay in the STT's turn audio.
        await self.say(gate, "")
        await self.say(gate, "")
        self.assertEqual(self.stt.resets, 2)
        await self.say(gate, "東京工科大学の")
        self.assertEqual(self.stt.resets, 3)
        # The turn goes on: its audio is kept.
        await self.say(gate, "行き方を教えてください")
        self.assertEqual(self.stt.resets, 3)

    async def test_turn_is_taken_back_with_what_the_cancelled_reply_left_in_the_context(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "学費を教えてください")
        # The user aggregator adds the turn; the interrupted reply leaves its first clause.
        self.context.messages += [{"role": "user", "content": "学費を教えてください"}]
        await self.send(gate, "UserStartedSpeakingFrame")
        self.assertEqual(self.aggregator.resets, 1)
        self.assertEqual(len(self.context.messages), 1)
        self.context.messages += [{"role": "assistant", "content": "学費は、"}]
        await self.send(gate, "UserStoppedSpeakingFrame")
        await self.transcript(gate, "入学金もお願いします")
        self.assertEqual(self.released(gate), ["学費を教えてください", "学費を教えてください 入学金もお願いします"])
        self.assertEqual(self.context.messages, [{"role": "system", "content": "s"}])

    async def test_context_is_left_alone_when_the_mark_is_not_the_released_turn(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "学費を教えてください")
        self.context.messages += [{"role": "assistant", "content": "x"}, {"role": "user", "content": "学費を教えてください"}]
        await self.send(gate, "UserStartedSpeakingFrame")
        self.assertEqual(len(self.context.messages), 3)

    async def test_turn_is_final_once_the_avatar_speaks(self) -> None:
        gate = self.make_gate()
        await self.say(gate, "学費を教えてください")
        resets = self.stt.resets
        await self.send(gate, "BotStartedSpeakingFrame")
        self.assertEqual(self.stt.resets, resets + 1)
        await self.send(gate, "BotStoppedSpeakingFrame")
        self.now += 1.0
        await self.say(gate, "入学金もお願いします")
        self.assertEqual(self.released(gate), ["学費を教えてください", "入学金もお願いします"])
        self.assertEqual(self.aggregator.resets, 0)

    async def test_turn_without_a_voice_is_not_merged_after_the_window(self) -> None:
        gate = self.make_gate(merge_window_s=2.0)
        await self.say(gate, "学費を教えてください")
        self.now += 2.5
        await self.say(gate, "入学金もお願いします")
        self.assertEqual(self.released(gate)[-1], "入学金もお願いします")
        self.assertEqual(self.stt.resets, 2)

    async def test_without_text_turn_end_transcripts_pass_unchanged(self) -> None:
        gate = self.make_gate(text_turn_end=False, gated_barge_in=True)
        await self.say(gate, "東京工科大学の")
        self.assertEqual(self.released(gate), ["東京工科大学の"])

    async def test_interruption_is_held_back_only_while_the_avatar_speaks(self) -> None:
        gate = self.make_gate(gated_barge_in=True, barge_in_hangover_s=0.3)
        self.assertFalse(gate.defer_interruption())
        await self.send(gate, "BotStartedSpeakingFrame")
        self.assertTrue(gate.defer_interruption())
        await self.send(gate, "BotStoppedSpeakingFrame")
        # A stop and a start follow each other between two sentences of one reply.
        self.now += 0.2
        self.assertTrue(gate.defer_interruption())
        self.now += 0.2
        self.assertFalse(gate.defer_interruption())
        self.assertFalse(self.make_gate(gated_barge_in=False).defer_interruption())

    async def barge(self, gate, text: str, speech_s: float = 0.0) -> None:
        await self.send(gate, "BotStartedSpeakingFrame")
        self.assertTrue(gate.defer_interruption())
        await self.send(gate, "UserStartedSpeakingFrame")
        await asyncio.sleep(speech_s)
        await self.send(gate, "UserStoppedSpeakingFrame")
        if text:
            await self.transcript(gate, text)
        await asyncio.sleep(0.02)

    async def test_backchannel_is_dropped(self) -> None:
        gate = self.make_gate(gated_barge_in=True)
        await self.barge(gate, "うん。")
        await asyncio.sleep(0.1)
        self.assertEqual(self.interruptions(gate), 0)
        self.assertEqual(self.released(gate), [])
        # Once when the speech started, once when its transcript was dropped.
        self.assertEqual(self.stt.resets, 2)

    async def test_real_speech_interrupts_before_its_text_goes_on(self) -> None:
        gate = self.make_gate(gated_barge_in=True)
        await self.barge(gate, "ちょっと待ってください")
        kinds = [type(frame).__name__ for frame, _ in gate.pushed]
        self.assertEqual(self.interruptions(gate), 1)
        self.assertLess(kinds.index("BotInterruptionFrame"), kinds.index("TranscriptionFrame"))
        self.assertEqual(self.released(gate), ["ちょっと待ってください"])

    async def test_speech_that_lasts_interrupts_without_waiting_for_its_end(self) -> None:
        gate = self.make_gate(gated_barge_in=True)
        await self.send(gate, "BotStartedSpeakingFrame")
        gate.defer_interruption()
        await self.send(gate, "UserStartedSpeakingFrame")
        await asyncio.sleep(0.1)
        self.assertEqual(self.interruptions(gate), 1)
        await self.send(gate, "BotStoppedSpeakingFrame")
        self.now += 1.0
        await self.send(gate, "UserStoppedSpeakingFrame")
        await self.transcript(gate, "入学金はいくらですか")
        self.assertEqual(self.interruptions(gate), 1)
        self.assertEqual(self.released(gate), ["入学金はいくらですか"])

    async def test_short_word_after_the_avatar_finished_is_a_turn(self) -> None:
        gate = self.make_gate(gated_barge_in=True, text_turn_end=False)
        await self.send(gate, "BotStartedSpeakingFrame")
        gate.defer_interruption()
        await self.send(gate, "UserStartedSpeakingFrame")
        await self.send(gate, "BotStoppedSpeakingFrame")
        self.now += 1.0
        await self.send(gate, "UserStoppedSpeakingFrame")
        await self.transcript(gate, "はい")
        self.assertEqual(self.released(gate), ["はい"])
        self.assertEqual(self.interruptions(gate), 0)


class TurnGateWiringTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.turn = load_tokkio_turn()
        self.frames = self.turn.frames

    def turn_taking(self, **overrides):
        values = {
            "turn_end_mode": "text",
            "barge_in_mode": "gated",
            "vad_stop_secs_in_use": 0.2,
            "turn_end_retranscribe": True,
            "turn_end_max_wait_s": 0.8,
            "turn_end_weak_wait_s": 0.6,
            "turn_end_merge_window_s": 2.0,
            "turn_end_hold_phrases": ["そうですね"],
            "barge_in_min_speech_s": 0.6,
            "barge_in_backchannels": ["うん"],
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def build(self, turn_taking, stt=None):
        self.input = SimpleNamespace(pushed=[], original=[])

        async def original(frame):
            self.input.original.append(frame)

        async def push_frame(frame):
            self.input.pushed.append(frame)

        self.input._handle_user_interruption = original
        self.input.push_frame = push_frame
        transport = SimpleNamespace(input=lambda: self.input)
        return self.turn.build_turn_gate(
            turn_taking, transport=transport, stt=stt, context=None, user_aggregator=None
        )

    async def test_both_off_builds_nothing_and_leaves_the_transport_alone(self) -> None:
        self.assertIsNone(self.build(self.turn_taking(turn_end_mode="vad", barge_in_mode="immediate")))
        frame = self.frames.UserStartedSpeakingFrame()
        await self.input._handle_user_interruption(frame)
        self.assertEqual(self.input.original, [frame])

    async def test_text_turn_end_alone_does_not_hook_the_transport(self) -> None:
        gate = self.build(self.turn_taking(barge_in_mode="immediate"))
        await gate.process_frame(self.frames.BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        frame = self.frames.UserStartedSpeakingFrame()
        await self.input._handle_user_interruption(frame)
        self.assertEqual(self.input.original, [frame])

    async def test_hook_holds_back_the_interruption_while_the_avatar_speaks(self) -> None:
        gate = self.build(self.turn_taking())
        start, stop = self.frames.UserStartedSpeakingFrame(), self.frames.UserStoppedSpeakingFrame()
        await self.input._handle_user_interruption(start)
        self.assertEqual(self.input.original, [start])

        await gate.process_frame(self.frames.BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.input._handle_user_interruption(start)
        await self.input._handle_user_interruption(stop)
        emulated = self.frames.UserStartedSpeakingFrame(emulated=True)
        await self.input._handle_user_interruption(emulated)
        self.assertEqual(self.input.pushed, [start])
        self.assertEqual(self.input.original, [start, stop, emulated])

    async def test_whole_turn_audio_is_enabled_only_with_text_turn_end(self) -> None:
        def make_stt():
            stt = SimpleNamespace(enabled=0)
            stt.enable_turn_audio = lambda: setattr(stt, "enabled", stt.enabled + 1)
            return stt

        stt = make_stt()
        self.assertTrue(self.build(self.turn_taking(), stt)._whole_turn_transcripts)
        self.assertEqual(stt.enabled, 1)
        for overrides in ({"turn_end_retranscribe": False}, {"turn_end_mode": "vad"}):
            stt = make_stt()
            self.assertFalse(self.build(self.turn_taking(**overrides), stt)._whole_turn_transcripts)
            self.assertEqual(stt.enabled, 0)
        # RivaASRService has no turn audio.
        self.assertFalse(self.build(self.turn_taking(), SimpleNamespace())._whole_turn_transcripts)

    async def test_min_speech_grows_with_the_vad_stop(self) -> None:
        self.assertAlmostEqual(self.build(self.turn_taking())._barge_in_min_speech_s, 0.6)
        gate = self.build(self.turn_taking(turn_end_mode="vad", vad_stop_secs_in_use=0.6))
        self.assertAlmostEqual(gate._barge_in_min_speech_s, 1.0)


class TurnDefaultsTests(unittest.TestCase):
    def test_customizer_lists_are_the_modules_defaults(self) -> None:
        import yaml

        spec = importlib.util.spec_from_file_location(
            "customize_for_turn_test", ROOT / "infra" / "tokkio" / "customize_tokkio_japanese.py"
        )
        customize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(customize)
        turn = load_tokkio_turn()
        turn_taking = yaml.safe_load(customize.build_config_yaml())["TurnTaking"]

        self.assertEqual(tuple(turn_taking["barge_in_backchannels"]), turn.DEFAULT_BACKCHANNELS)
        self.assertEqual(tuple(turn_taking["turn_end_hold_phrases"]), turn.DEFAULT_HOLD_PHRASES)


if __name__ == "__main__":
    unittest.main()
