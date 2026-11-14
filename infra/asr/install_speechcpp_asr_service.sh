#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The Japanese unit (Haas) and the English mode's unit (Qwen3-ASR); manage_tokkio.sh starts the one config.yaml uses.
read -r -a SERVICE_NAMES <<<"${SPEECHCPP_ASR_SERVICE_NAMES:-ace-speechcpp-asr.service ace-speechcpp-asr-en.service}"
RESTART=false

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

case "${1:-}" in
  "")
    ;;
  --restart)
    RESTART=true
    ;;
  *)
    die "usage: $0 [--restart]  (--restart applies the units to running services and drops a transcription in progress)"
    ;;
esac

# Paths come from the unit itself, so the files checked here are the ones the service reads.
unit_env() {
  sed -n -E "s/^Environment=\"?$2=([^\"]*)\"?\$/\1/p" "$1" | tail -n 1
}

check_unit() {
  local src="$1" bin model vad wrapper
  [[ -f "${src}" ]] || die "service unit not found: ${src}"
  bin="$(unit_env "${src}" SPEECHCPP_ASR_BIN)"
  model="$(unit_env "${src}" SPEECHCPP_ASR_MODEL)"
  vad="$(unit_env "${src}" SPEECHCPP_ASR_VAD_MODEL)"
  wrapper="$(sed -n -E 's/^ExecStart=([^[:space:]]+).*/\1/p' "${src}" | head -n 1)"
  [[ -n "${bin}" && -n "${model}" ]] || die "${src} must set SPEECHCPP_ASR_BIN and SPEECHCPP_ASR_MODEL"
  [[ -x "${bin}" ]] || die "speech.cpp binary not found: ${bin}"
  [[ -f "${model}" ]] || die "speech.cpp model not found: ${model} (fetch it with speech pull, SPEECH_MODEL_DIR=/data/ACE/speech-cpp/models)"
  [[ -z "${vad}" || -f "${vad}" ]] || die "speech.cpp VAD model not found: ${vad} (fetch it with: speech pull silero-vad)"
  [[ -x "${wrapper}" ]] || die "start script not found or not executable: ${wrapper}"
}

for name in "${SERVICE_NAMES[@]}"; do
  check_unit "${SCRIPT_DIR}/${name}"
done

for name in "${SERVICE_NAMES[@]}"; do
  sudo install -m 0644 "${SCRIPT_DIR}/${name}" "/etc/systemd/system/${name}"
  echo "[INFO] Installed /etc/systemd/system/${name}"
done
sudo systemctl daemon-reload

# manage_tokkio.sh starts and stops the units with Tokkio when config.yaml selects SpeechCppSTTService.
for name in "${SERVICE_NAMES[@]}"; do
  if [[ "${RESTART}" == "true" ]]; then
    sudo systemctl try-restart "${name}"
    echo "[INFO] Restarted ${name} if it was running"
  elif systemctl is-active --quiet "${name}"; then
    echo "[WARN] ${name} is still running the previous unit. Restart to apply: sudo systemctl restart ${name}" >&2
  fi
done
