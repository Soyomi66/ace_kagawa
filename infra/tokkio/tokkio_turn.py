"""Turn taking for the Tokkio pipeline: end of turn from the transcript, and gated barge-in.

Two behaviours, each behind its own TurnTaking switch (config.yaml). With both off, bot.py does
not build the gate and the pipeline is the upstream one.

``turn_end_mode: "text"`` (plan A, docs/audits/turn-end-text-2026-11-02.md)
    The VAD stops after a short silence (0.2 s) and the gate, placed right after the STT, reads
    the end of the transcript: a finished sentence goes to the LLM at once, a plain-form ending
    waits a little longer, an unfinished one waits until the silence reaches the limit. When the
    visitor speaks again before the avatar's voice starts, pipecat's interruption drops the reply
    and the gate takes the turn back: the released text leaves the LLM context and the next
    transcript is merged with it.

``barge_in_mode: "gated"``
    While the avatar speaks, the start of speech no longer interrupts it at once. The gate
    interrupts when the speech lasts ``barge_in_min_speech_s``, or when the utterance has ended
    and its transcript is not a backchannel (うん, はい, ...). Backchannels and empty transcripts
    (noise) are dropped and the avatar keeps speaking.

The text rules need no pipecat, so the offline evaluation and the unit tests import them alone.
"""

from __future__ import annotations

import asyncio
import time
import unicodedata

from loguru import logger

try:
    from pipecat.frames.frames import (
        BotInterruptionFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        CancelFrame,
        EndFrame,
        TranscriptionFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
except ImportError:  # offline use of the text rules
    FrameProcessor = object


COMPLETE = "complete"
WEAK = "weak"
OPEN = "open"

_TRAILING_MARKS = "。.!?、, 　ー〜~…"
# Endings that leave the sentence open: conjunctive and case particles, te-form.
_OPEN_ENDINGS = (
    "けど", "けれど", "けれども", "ですが", "ますが", "だが", "ので", "のに", "から", "し",
    "て", "で", "って", "たら", "なら", "れば", "ながら", "ため", "など", "とか", "より", "まで",
    "の", "を", "に", "は", "が", "へ", "と", "も", "や", "について", "として", "という", "っていう",
)
_FILLERS = (
    "えっと", "えーと", "えーっと", "ええと", "あの", "あのー", "えー", "えーっ", "その", "まあ",
    "うーん", "んー", "なんか", "ちょっと",
)
# Polite sentence ends and sentence-final particles: the visitor has finished.
_POLITE_ENDINGS = (
    "です", "ます", "ました", "でした", "ません", "ませんでした", "でしょう", "ましょう", "ください",
    "下さい", "か", "ね", "よ", "よね", "かな", "かね",
)
# Plain-form ends. They also end a clause that modifies the next noun
# (「AIを使った｜非破壊検査の…」), which no word list can tell apart, so they wait a little.
_PLAIN_ENDINGS = ("だ", "だった", "ない", "なかった", "たい", "た", "る", "う", "い")
# Short replies and greetings. Often the whole turn, and as often the start of one
# (「はい｜お願いします」, 11 of the 14 cuts in the 2026-11-02 evaluation), so they wait like plain forms.
_SHORT_REPLIES = (
    "はい", "いいえ", "うん", "ええ", "いや", "いえ", "そう", "そうそう", "なるほど", "こんにちは", "こんばんは",
    "おはようございます", "ありがとう", "どうも", "大丈夫", "了解", "ok", "おっけー", "結構",
    "以上", "わかった",
)
# Said to get attention or to gain time: more follows (「すみません｜もう一度…」, 「そうですね、えっと…」).
# 「そうですね」 is also a reply; as a reply it waits for the silence limit. Set in config.yaml.
DEFAULT_HOLD_PHRASES = ("そうですね", "すみません", "すいません")
DEFAULT_BACKCHANNELS = (
    "うん", "うんうん", "はい", "はいはい", "ええ", "へえ", "ほう", "ほお", "ふうん", "ふむ", "ああ", "おお",
    "なるほど", "なるほどね", "そうですね", "そうですか", "そうなんですね", "そうなんですか", "そうなんだ",
    "そっか", "たしかに", "確かに",
)


def _katakana_to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def normalize_turn_text(text: str) -> str:
    """NFKC, lower case, katakana as hiragana (Haas writes 「ハイ」), no outer spaces."""
    return _katakana_to_hiragana(unicodedata.normalize("NFKC", text).strip().lower())


def _ends_with_short_reply(core: str) -> bool:
    return core in _SHORT_REPLIES or any(
        core.endswith(reply) and len(core) <= len(reply) + 4 for reply in _SHORT_REPLIES
    )


def classify_turn_end(text: str, hold_phrases=DEFAULT_HOLD_PHRASES) -> str:
    """How the transcript so far ends: COMPLETE, WEAK (may well go on) or OPEN (more to come).

    Punctuation at the end is ignored except a trailing comma: Qwen3-ASR ends every transcript
    with 。, so the ASR's own full stop says nothing.
    """
    normalized = normalize_turn_text(text)
    if not normalized or normalized.endswith(("、", ",")):
        return OPEN
    core = normalized.rstrip(_TRAILING_MARKS)
    if not core:
        return OPEN
    if any(core.endswith(normalize_turn_text(phrase)) for phrase in hold_phrases if phrase):
        return OPEN
    # Before the particles: 「こんにちは」 ends in は and 「どうも」 in も.
    if _ends_with_short_reply(core):
        return WEAK
    if core.endswith(_FILLERS) or core.endswith(_OPEN_ENDINGS):
        return OPEN
    if core.endswith(_POLITE_ENDINGS):
        return COMPLETE
    if core.endswith(_PLAIN_ENDINGS):
        return WEAK
    return OPEN


def _backchannel_key(text: str) -> str:
    # 「へー」「ふーん」「うーん」: a long-vowel mark repeats the vowel before it.
    vowels = {"あ": "あ", "か": "あ", "は": "あ", "な": "あ", "い": "い", "う": "う", "ふ": "う", "え": "え", "へ": "え",
              "お": "お", "ほ": "お", "そ": "お"}
    out = []
    for char in normalize_turn_text(text):
        if char in "ー〜~" and out:
            out.append(vowels.get(out[-1], ""))
        elif unicodedata.category(char)[0] in "LN":
            out.append(char)
        else:
            out.append(" ")
    return "".join(out)


def is_backchannel(text: str, backchannels=DEFAULT_BACKCHANNELS) -> bool:
    """True when every word of the transcript is a backchannel (「うん、うん。」「はい」)."""
    words = _backchannel_key(text).split()
    if not words or len(words) > 3:
        return False
    known = {_backchannel_key(word).strip() for word in backchannels}
    return all(word in known for word in words)


class TurnGate(FrameProcessor):
    """Holds the STT's transcripts until the visitor's turn has ended; decides on barge-in.

    Sits between the STT and the user context aggregator. Text it holds lives in attributes, not
    in frames, because an interruption empties every processor's queues.
    """

    def __init__(
        self,
        *,
        text_turn_end: bool = False,
        gated_barge_in: bool = False,
        vad_stop_secs: float = 0.2,
        max_wait_s: float = 0.8,
        weak_wait_s: float = 0.45,
        merge_window_s: float = 2.0,
        hold_phrases=DEFAULT_HOLD_PHRASES,
        whole_turn_transcripts: bool = False,
        barge_in_min_speech_s: float = 0.6,
        barge_in_hangover_s: float = 0.3,
        backchannels=DEFAULT_BACKCHANNELS,
        context=None,
        user_aggregator=None,
        stt=None,
        clock=time.monotonic,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._text_turn_end = text_turn_end
        self._gated_barge_in = gated_barge_in
        self._vad_stop_secs = vad_stop_secs
        self._max_wait_s = max_wait_s
        self._weak_wait_s = weak_wait_s
        self._merge_window_s = merge_window_s
        self._hold_phrases = tuple(hold_phrases)
        self._whole_turn_transcripts = whole_turn_transcripts
        self._barge_in_min_speech_s = barge_in_min_speech_s
        self._barge_in_hangover_s = barge_in_hangover_s
        self._backchannels = tuple(backchannels)
        self._context = context
        self._user_aggregator = user_aggregator
        self._stt = stt
        self._now = clock

        self._user_speaking = False
        self._bot_speaking = False
        self._bot_stopped_at = float("-inf")
        self._user_stopped_at = 0.0
        # Text of the turn so far that the LLM has not been given.
        self._held = ""
        self._frame_template = None
        self._release_task = None
        # The turn the LLM was given last, until the avatar's voice starts: (text, time, context length before it).
        self._released = None
        # Context length to cut back to when the turn that was taken back goes out again.
        self._takeback_mark = None
        # Barge-in while the avatar speaks: None, "deferred" (speech going on) or "awaiting" (its transcript).
        self._barge = None
        self._barge_task = None
        self._tasks = set()

    # ---- called by the input transport's hook -------------------------------------------

    def defer_interruption(self) -> bool:
        """Whether the speech that just started must not interrupt the avatar yet."""
        if not self._gated_barge_in or not self._bot_busy():
            self._barge = None
            return False
        self._barge = "deferred"
        return True

    # ---- frames ---------------------------------------------------------------------------

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._handle_transcription(frame)
            return
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self._released is not None:
                # The reply is audible: the turn is over and cannot be merged any more.
                self._finish_turn()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = self._now()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._cancel_timers()
        await self.push_frame(frame, direction)

    async def _handle_user_started(self) -> None:
        self._user_speaking = True
        await self._cancel_release_task()
        if not self._held and self._released is None:
            # A new turn. Segments that gave no transcript (noise, an STT error) or that came
            # while the avatar spoke must not stay in front of it.
            self._reset_stt_turn()
        if self._barge == "deferred":
            await self._cancel_barge_task()
            self._barge_task = self._start_task(self._barge_timer())
            return
        self._barge = None
        if self._released is None:
            return
        text, released_at, mark = self._released
        self._released = None
        if self._now() - released_at > self._merge_window_s:
            # No voice came for that turn (an operator cue, an error). Start a new turn.
            self._reset_stt_turn()
            return
        # The visitor went on before the avatar's voice: pipecat's interruption drops the
        # reply, and the next transcript continues this turn.
        logger.info(f"Turn taken back, the visitor went on: [{text}]")
        self._held = text
        if self._user_aggregator is not None:
            self._user_aggregator.reset()
        self._take_back_from_context(text, mark)

    async def _handle_user_stopped(self) -> None:
        self._user_speaking = False
        self._user_stopped_at = self._now()
        if self._barge == "deferred":
            await self._cancel_barge_task()
            self._barge = "awaiting"
        elif self._held:
            # No transcript may follow (noise transcribes to nothing): the held text still goes out.
            self._schedule_release(self._max_wait_s)

    async def _handle_transcription(self, frame) -> None:
        text = frame.text.strip()
        if not text:
            return
        if self._barge == "awaiting":
            self._barge = None
            if self._bot_busy():
                if is_backchannel(text, self._backchannels):
                    logger.info(f"Backchannel while the avatar speaks, not a turn: [{text}]")
                    self._reset_stt_turn()
                    return
                logger.info(f"Barge-in on transcript: [{text}]")
                # Not from here: the interruption cancels the task this frame is processed in.
                self._start_task(self._interrupt_then_accept(frame, text))
                return
        await self._accept(frame, text)

    async def _accept(self, frame, text: str) -> None:
        if not self._text_turn_end:
            await self.push_frame(frame)
            return
        if self._held and not self._whole_turn_transcripts:
            text = f"{self._held} {text}"
        self._held = text
        self._frame_template = frame
        if self._user_speaking:
            return
        ending = classify_turn_end(text, self._hold_phrases)
        if ending == COMPLETE:
            await self._cancel_release_task()
            await self._release(f"rule, {ending}")
        else:
            self._schedule_release(self._weak_wait_s if ending == WEAK else self._max_wait_s, ending)

    # ---- release ----------------------------------------------------------------------------

    def _schedule_release(self, silence_s: float, ending: str = "no transcript") -> None:
        """Release the held text once the silence since the VAD stop reaches silence_s."""
        delay = max(0.0, silence_s - self._vad_stop_secs - (self._now() - self._user_stopped_at))
        if self._release_task is not None:
            self._release_task.cancel()
        self._release_task = self._start_task(self._release_later(delay, ending))

    async def _release_later(self, delay: float, ending: str) -> None:
        await asyncio.sleep(delay)
        self._release_task = None
        if self._held and not self._user_speaking:
            await self._release(f"waited, {ending}")

    async def _release(self, reason: str) -> None:
        text, self._held = self._held, ""
        self._truncate_context()
        self._takeback_mark = None
        mark = len(self._context.get_messages()) if self._context is not None else None
        self._released = (text, self._now(), mark)
        waited = self._now() - self._user_stopped_at + self._vad_stop_secs
        logger.info(f"Turn end ({reason}) {waited:.2f} s after the speech: [{text}]")
        template = self._frame_template
        await self.push_frame(
            TranscriptionFrame(text, template.user_id, template.timestamp, getattr(template, "language", None))
        )

    def _take_back_from_context(self, text: str, mark) -> None:
        """Drop the released turn from the context; remember where it was."""
        self._takeback_mark = None
        if mark is None or self._context is None:
            return
        messages = self._context.get_messages()
        if len(messages) == mark:
            # The user aggregator had not added it yet.
            self._takeback_mark = mark
        elif messages[mark].get("role") == "user" and messages[mark].get("content") == text:
            self._context.set_messages(list(messages[:mark]))
            self._takeback_mark = mark

    def _truncate_context(self) -> None:
        """Drop what the cancelled reply left behind (the interruption makes the assistant
        aggregator store the part of the reply it had received)."""
        mark = self._takeback_mark
        if mark is None or self._context is None:
            return
        messages = self._context.get_messages()
        if len(messages) > mark:
            self._context.set_messages(list(messages[:mark]))

    def _finish_turn(self) -> None:
        self._released = None
        self._reset_stt_turn()

    def _reset_stt_turn(self) -> None:
        reset = getattr(self._stt, "reset_turn_audio", None)
        if reset is not None:
            reset()

    # ---- barge-in ---------------------------------------------------------------------------

    def _bot_busy(self) -> bool:
        # The animation service reports a stop and a start between the sentences of one reply.
        return self._bot_speaking or self._now() - self._bot_stopped_at < self._barge_in_hangover_s

    async def _barge_timer(self) -> None:
        await asyncio.sleep(self._barge_in_min_speech_s)
        self._barge_task = None
        if self._barge == "deferred":
            self._barge = None
            logger.info(f"Barge-in after {self._barge_in_min_speech_s:.2f} s of speech")
            await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

    async def _interrupt_then_accept(self, frame, text: str) -> None:
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)
        await self._accept(frame, text)

    # ---- tasks ------------------------------------------------------------------------------

    def _start_task(self, coroutine):
        # Plain asyncio tasks: pipecat's task manager keeps a finished task registered until
        # someone waits for it, and these timers end on their own.
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(f"{self} turn gate task failed: {task.exception()!r}")

    async def _cancel_release_task(self) -> None:
        task, self._release_task = self._release_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _cancel_barge_task(self) -> None:
        task, self._barge_task = self._barge_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _cancel_timers(self) -> None:
        self._release_task = self._barge_task = None
        for task in list(self._tasks):
            if task is not asyncio.current_task():
                task.cancel()


def install_barge_in_hook(input_transport, gate: TurnGate) -> None:
    """Let the gate hold back the interruption the input transport sends at the start of speech.

    pipecat 0.0.68's BaseInputTransport._handle_user_interruption pushes UserStartedSpeakingFrame
    and then interrupts the pipeline. While the avatar speaks, the gate gets the frame alone and
    interrupts later with a BotInterruptionFrame, or not at all.
    """
    original = input_transport._handle_user_interruption

    async def handle_user_interruption(frame):
        if isinstance(frame, UserStartedSpeakingFrame) and not frame.emulated and gate.defer_interruption():
            logger.debug("User started speaking while the avatar speaks; interruption held back")
            await input_transport.push_frame(frame)
            return
        await original(frame)

    input_transport._handle_user_interruption = handle_user_interruption


def build_turn_gate(turn_taking, *, transport, stt, context, user_aggregator):
    """The gate for config.TurnTaking, or None when both behaviours are off."""
    text_turn_end = turn_taking.turn_end_mode == "text"
    gated_barge_in = turn_taking.barge_in_mode == "gated"
    if not text_turn_end and not gated_barge_in:
        return None
    vad_stop_secs = turn_taking.vad_stop_secs_in_use
    whole_turn = bool(text_turn_end and turn_taking.turn_end_retranscribe and hasattr(stt, "enable_turn_audio"))
    if whole_turn:
        stt.enable_turn_audio()
    gate = TurnGate(
        text_turn_end=text_turn_end,
        gated_barge_in=gated_barge_in,
        vad_stop_secs=vad_stop_secs,
        max_wait_s=turn_taking.turn_end_max_wait_s,
        weak_wait_s=turn_taking.turn_end_weak_wait_s,
        merge_window_s=turn_taking.turn_end_merge_window_s,
        hold_phrases=turn_taking.turn_end_hold_phrases,
        whole_turn_transcripts=whole_turn,
        # The VAD reports the end of speech vad_stop_secs late, so a short word still counts as
        # speech for that long. The limit was chosen for a 0.2 s stop.
        barge_in_min_speech_s=turn_taking.barge_in_min_speech_s + max(0.0, vad_stop_secs - 0.2),
        backchannels=turn_taking.barge_in_backchannels,
        context=context,
        user_aggregator=user_aggregator,
        stt=stt,
    )
    if gated_barge_in:
        install_barge_in_hook(transport.input(), gate)
    logger.info(
        f"Turn gate: turn_end_mode={turn_taking.turn_end_mode} barge_in_mode={turn_taking.barge_in_mode} "
        f"vad_stop_secs={vad_stop_secs} whole_turn_transcripts={whole_turn}"
    )
    return gate
