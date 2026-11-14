#!/usr/bin/env bash
# speech.cpp's speech serve for Tokkio's SpeechCppSTTService: Haas for Japanese
# (ace-speechcpp-asr.service), Qwen3-ASR for the English mode (ace-speechcpp-asr-en.service).
# The unit sets the SPEECHCPP_ASR_* variables.
set -euo pipefail

: "${SPEECHCPP_ASR_BIN:?}" "${SPEECHCPP_ASR_MODEL:?}" "${SPEECHCPP_ASR_DEVICE:?}"
HOST="${SPEECHCPP_ASR_HOST:-127.0.0.1}"
PORT="${SPEECHCPP_ASR_PORT:-8024}"
# The default, every core, contends with the renderer and triton on this host:
# 22 s of audio took 20-30 s instead of 0.07 s (docs/audits/speech-cpp-asr-2026-10-17.md).
THREADS="${SPEECHCPP_ASR_THREADS:-4}"
READY_ATTEMPTS="${SPEECHCPP_ASR_READY_ATTEMPTS:-120}"
export SPEECH_NO_UPDATE_CHECK=1

models=("${SPEECHCPP_ASR_MODEL}")
# Silero VAD beside the recognition model serves chunking_strategy=auto, which the
# Japanese config sends: without it Haas writes 「うん。」 for noise. A server
# without the VAD model answers every such request 400, so a missing file stops here.
if [[ -n "${SPEECHCPP_ASR_VAD_MODEL:-}" ]]; then
  if [[ ! -f "${SPEECHCPP_ASR_VAD_MODEL}" ]]; then
    echo "speech.cpp VAD model not found: ${SPEECHCPP_ASR_VAD_MODEL}" >&2
    exit 1
  fi
  models+=("${SPEECHCPP_ASR_VAD_MODEL}")
fi
# Server-default custom vocabulary (Haas only), "written=reading,reading"; a request without keywords uses it.
extra=()
if [[ -n "${SPEECHCPP_ASR_KEYWORDS:-}" ]]; then
  extra+=(--keywords "${SPEECHCPP_ASR_KEYWORDS}")
fi

"${SPEECHCPP_ASR_BIN}" serve "${models[@]}" \
  --device "${SPEECHCPP_ASR_DEVICE}" --threads "${THREADS}" --host "${HOST}" --port "${PORT}" "${extra[@]}" &
pid=$!
trap 'kill "${pid}" 2>/dev/null || true; wait "${pid}" 2>/dev/null || true; exit 143' TERM INT

# Vulkan numbers this host's GPUs the other way round from nvidia-smi, so the
# device name alone does not say which GPU the model went to.
if [[ -n "${SPEECHCPP_ASR_GPU_UUID:-}" ]]; then
  probe_host="${HOST}"
  [[ "${probe_host}" == "0.0.0.0" ]] && probe_host="127.0.0.1"
  ready=false
  for _ in $(seq "${READY_ATTEMPTS}"); do
    kill -0 "${pid}" 2>/dev/null || break
    if curl -fsS "http://${probe_host}:${PORT}/health" >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done
  if [[ "${ready}" != true ]]; then
    echo "speech serve (pid ${pid}) did not become ready on port ${PORT}" >&2
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    exit 1
  fi
  actual="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
    | awk -F', *' -v p="${pid}" '$1 == p { print $2 }')"
  if [[ "${actual}" != "${SPEECHCPP_ASR_GPU_UUID}" ]]; then
    echo "speech serve (pid ${pid}) is on GPU '${actual:-none}', not ${SPEECHCPP_ASR_GPU_UUID}; stopping it" >&2
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    exit 1
  fi
  echo "speech serve (pid ${pid}) is on ${actual}"
fi

wait "${pid}"
