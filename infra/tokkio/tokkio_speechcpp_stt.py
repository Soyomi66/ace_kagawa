"""speech.cpp's `speech serve` as Tokkio's speech-to-text service.

speech serve speaks OpenAI's transcription API, so this is pipecat's
OpenAISTTService (a SegmentedSTTService: one request per utterance the
transport's Silero VAD delimits) with four changes for the local server:

- no API key, a short timeout and no retries: a request takes about 0.1 s, and
  the SDK's defaults (600 s, 2 retries) would stall a turn on a server hiccup;
- the model is the server's id (``Haas-ja-v1.0``, ``Qwen3-ASR-1.7B``), not the
  catalog name;
- ``chunking_strategy`` (``auto`` for Haas): the server's own Silero VAD keeps
  only the regions where someone speaks, so noise the transport's VAD passes
  comes back empty. Without it Haas writes 「うん。」 for noise
  (docs/audits/speech-cpp-asr-haas-2026-11-02.md). The server needs a VAD model
  for it, or it answers every request 400;
- a transcript that only repeats the prompt is dropped. Qwen3-ASR writes the
  prompt for silence and noise (all six probes in
  docs/audits/speech-cpp-asr-2026-10-17.md), and the VAD passes it noise.

With the text-based end of turn (tokkio_turn.py) the VAD ends a segment at every short
pause. enable_turn_audio() then makes each request carry the whole turn so far, the
condition docs/audits/turn-end-text-2026-11-02.md evaluated, until the turn gate calls
reset_turn_audio().
"""

from __future__ import annotations

import unicodedata

from loguru import logger
from openai import AsyncOpenAI
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transcriptions.language import Language


def normalize_transcript(text: str) -> str:
    """NFKC, lower case, and no punctuation, symbols or spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] not in "PSZC")


def repeats_prompt(text: str, prompt: str | None) -> bool:
    if not prompt:
        return False
    normalized = normalize_transcript(text)
    return bool(normalized) and normalized == normalize_transcript(prompt)


class _DroppedTranscription:
    # BaseWhisperSTTService.run_stt pushes no frame for an empty text.
    text = ""


class SpeechCppSTTService(OpenAISTTService):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        language: str = "ja",
        prompt: str = "",
        chunking_strategy: str = "",
        timeout_s: float = 10.0,
        **kwargs,
    ):
        # Read by _create_client, which OpenAISTTService.__init__ calls.
        self._timeout_s = timeout_s
        self._chunking_strategy = chunking_strategy
        # None: one request per VAD segment. A bytearray: the audio of the turn's earlier segments.
        self._turn_audio: bytearray | None = None
        self._turn_audio_max_s = 30.0
        super().__init__(
            model=model,
            api_key="unused",
            base_url=base_url,
            language=Language(language),
            # A model that takes no prompt (FastConformer) refuses the field.
            prompt=prompt or None,
            **kwargs,
        )

    def _create_client(self, api_key, base_url):
        return AsyncOpenAI(api_key=api_key or "unused", base_url=base_url, timeout=self._timeout_s, max_retries=0)

    def enable_turn_audio(self, max_seconds: float = 30.0) -> None:
        self._turn_audio = bytearray()
        self._turn_audio_max_s = max_seconds

    def reset_turn_audio(self) -> None:
        if self._turn_audio is not None:
            self._turn_audio = bytearray()

    async def _handle_user_stopped_speaking(self, frame):
        if self._turn_audio is not None and not frame.emulated:
            # SegmentedSTTService transcribes _audio_buffer and clears it. The buffer starts
            # up to one second before the segment, so short pauses stay in the turn's audio.
            self._turn_audio += self._audio_buffer
            if len(self._turn_audio) > int(self._turn_audio_max_s * self.sample_rate) * 2:
                logger.info("Turn audio is over the limit; transcribing the last segment alone")
                self._turn_audio = bytearray(self._audio_buffer)
            self._audio_buffer = bytearray(self._turn_audio)
        await super()._handle_user_stopped_speaking(frame)

    async def _request(self, audio: bytes):
        if not self._chunking_strategy:
            return await super()._transcribe(audio)
        # pipecat 0.0.68's OpenAISTTService._transcribe, plus chunking_strategy. It goes
        # in extra_body, which every 1.x SDK merges into the form, whatever version the
        # controller image has.
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self.model_name,
            "language": self._language,
            "extra_body": {"chunking_strategy": self._chunking_strategy},
        }
        if self._prompt is not None:
            kwargs["prompt"] = self._prompt
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return await self._client.audio.transcriptions.create(**kwargs)

    async def _transcribe(self, audio: bytes):
        response = await self._request(audio)
        if repeats_prompt(response.text, self._prompt):
            logger.info(f"Dropped a transcript that only repeats the STT prompt: [{response.text}]")
            return _DroppedTranscription()
        return response
