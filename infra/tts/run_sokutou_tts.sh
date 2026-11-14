#!/usr/bin/env bash
# sokutou's voice server (github.com/kosuketut/sokutou) in docker, for the sokutou backend of
# irodori_server.py. ace-sokutou-tts.service sets the SOKUTOU_TTS_* variables.
set -euo pipefail

: "${SOKUTOU_TTS_IMAGE:?}" "${SOKUTOU_TTS_GPU_UUID:?}" "${SOKUTOU_TTS_NUM_STEPS:?}"
: "${SOKUTOU_TTS_VOICE:?}" "${SOKUTOU_TTS_REFERENCE_WAV:?}" "${SOKUTOU_TTS_HF_HOME:?}"
NAME="${SOKUTOU_TTS_CONTAINER:-ace-sokutou-tts}"
PORT="${SOKUTOU_TTS_PORT:-8026}"
ENTRY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sokutou_entry.py"

if [[ ! -f "${SOKUTOU_TTS_REFERENCE_WAV}" ]]; then
  echo "reference wav not found: ${SOKUTOU_TTS_REFERENCE_WAV} (install_irodori_tts_service.sh makes it)" >&2
  exit 1
fi
if [[ ! -d "${SOKUTOU_TTS_HF_HOME}/hub" ]]; then
  echo "Hugging Face cache not found: ${SOKUTOU_TTS_HF_HOME}/hub" >&2
  exit 1
fi

# A container left by a killed run would hold the name and the port.
docker rm -f "${NAME}" >/dev/null 2>&1 || true

# The cache is read-only and offline, so starting never downloads. The voice is the service's
# reference wav itself. 48 kHz out gives irodori_server.py's output stage the same input as
# speech.cpp, and EMPTY_CACHE_EVERY=0 is the setting the 2026-11-11 benchmark ran.
exec docker run --rm --name "${NAME}" \
  --gpus "device=${SOKUTOU_TTS_GPU_UUID}" \
  -p "127.0.0.1:${PORT}:8088" \
  -v "${SOKUTOU_TTS_HF_HOME}:/root/.cache/huggingface:ro" \
  -v "${SOKUTOU_TTS_REFERENCE_WAV}:/voices/${SOKUTOU_TTS_VOICE}/ref.wav:ro" \
  -v "${ENTRY}:/opt/ace/sokutou_entry.py:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e "IRODORI_NUM_STEPS=${SOKUTOU_TTS_NUM_STEPS}" \
  -e EMPTY_CACHE_EVERY=0 \
  -e OUTPUT_SAMPLE_RATE=48000 \
  "${SOKUTOU_TTS_IMAGE}" /opt/irodori/.venv/bin/python /opt/ace/sokutou_entry.py
