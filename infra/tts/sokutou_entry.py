"""Start sokutou's voice server (engine.serve) on the text exactly as irodori_server.py sends it.

sokutou's server.py adds 「。」 to every text that does not end in its TERMINATORS. irodori_server.py
has already normalized the text and split the reply at 「、」 and at 64 characters, and speech.cpp
read those parts unchanged. With the added 「。」 a clause such as 😊私は複合材料、 becomes
「私は複合材料、。」 and ends in a stray 「を」 (5 of 30 clips, 0 with speech.cpp;
docs/audits/tts-sokutou-2026-11-11.md). So this keeps sokutou's whitespace folding and drops the
added 「。」, which gives sokutou the same text speech.cpp got for every kind of ending.

run_sokutou_tts.sh mounts this file read-only into the container and runs it with the image's
python; PYTHONPATH there already holds server.py (/app) and engine/ (/opt/sokutou).
"""

import runpy

import server


def _fold_whitespace(text: str) -> str:
    return " ".join(text.split())


server.normalize_text = _fold_whitespace
runpy.run_module("engine.serve", run_name="__main__", alter_sys=True)
