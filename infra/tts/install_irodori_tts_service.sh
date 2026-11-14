#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="${IRODORI_TTS_SERVICE_NAME:-ace-irodori-tts.service}"
SERVICE_SRC="${SCRIPT_DIR}/${SERVICE_NAME}"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}"
# The sokutou backend's container unit, installed beside the service when the unit selects it.
SOKUTOU_SERVICE_NAME="ace-sokutou-tts.service"
SOKUTOU_SERVICE_SRC="${SCRIPT_DIR}/${SOKUTOU_SERVICE_NAME}"
SOKUTOU_SERVICE_DST="/etc/systemd/system/${SOKUTOU_SERVICE_NAME}"
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
    die "usage: $0 [--restart]  (--restart applies the unit to a running service and cuts off speech in progress)"
    ;;
esac

[[ -f "${SERVICE_SRC}" ]] || die "service unit not found: ${SERVICE_SRC}"

# Paths come from the unit itself, so the files prepared here are the ones the service reads.
unit_env() {
  sed -n -E "s/^Environment=\"?$1=([^\"]*)\"?\$/\1/p" "${2:-${SERVICE_SRC}}" | tail -n 1
}

DATA_ROOT="$(unit_env IRODORI_TTS_DATA_ROOT)"
REFERENCE_SRC="$(unit_env IRODORI_TTS_REFERENCE_SOURCE)"
REFERENCE_WAV="$(unit_env IRODORI_TTS_REFERENCE_WAV)"
SERVICE_PYTHON="$(sed -n -E 's/^ExecStart=([^[:space:]]+).*/\1/p' "${SERVICE_SRC}" | head -n 1)"

[[ -n "${DATA_ROOT}" && -n "${REFERENCE_SRC}" && -n "${REFERENCE_WAV}" ]] \
  || die "${SERVICE_SRC} must set IRODORI_TTS_DATA_ROOT, IRODORI_TTS_REFERENCE_SOURCE and IRODORI_TTS_REFERENCE_WAV"
[[ -x "${SERVICE_PYTHON}" ]] || die "Irodori-TTS venv python not found: ${SERVICE_PYTHON}"
[[ -f "${REFERENCE_SRC}" ]] || die "reference audio not found: ${REFERENCE_SRC}"
if [[ ! -f "${REFERENCE_WAV}" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
  die "ffmpeg is required to convert ${REFERENCE_SRC} to ${REFERENCE_WAV}"
fi

BACKEND="$(unit_env IRODORI_TTS_BACKEND)"
if [[ "${BACKEND}" == "speechcpp" ]]; then
  SPEECHCPP_BIN="$(unit_env IRODORI_TTS_SPEECHCPP_BIN)"
  SPEECHCPP_MODEL="$(unit_env IRODORI_TTS_SPEECHCPP_MODEL)"
  SPEECHCPP_VOICE="$(unit_env IRODORI_TTS_SPEECHCPP_VOICE)"
  [[ -n "${SPEECHCPP_BIN}" && -n "${SPEECHCPP_MODEL}" && -n "${SPEECHCPP_VOICE}" ]] \
    || die "${SERVICE_SRC} must set IRODORI_TTS_SPEECHCPP_BIN, _MODEL and _VOICE for the speechcpp backend"
  [[ -x "${SPEECHCPP_BIN}" ]] || die "speech.cpp binary not found: ${SPEECHCPP_BIN}"
  [[ -f "${SPEECHCPP_MODEL}" ]] || die "speech.cpp model not found: ${SPEECHCPP_MODEL}"
elif [[ "${BACKEND}" == "sokutou" ]]; then
  [[ -f "${SOKUTOU_SERVICE_SRC}" ]] || die "sokutou unit not found: ${SOKUTOU_SERVICE_SRC}"
  SOKUTOU_IMAGE="$(unit_env SOKUTOU_TTS_IMAGE "${SOKUTOU_SERVICE_SRC}")"
  [[ -n "${SOKUTOU_IMAGE}" ]] || die "${SOKUTOU_SERVICE_SRC} must set SOKUTOU_TTS_IMAGE"
  # The container and the client must agree, or the service refuses the engine at its first request.
  [[ "$(unit_env SOKUTOU_TTS_NUM_STEPS "${SOKUTOU_SERVICE_SRC}")" == "$(unit_env IRODORI_TTS_NUM_STEPS)" ]] \
    || die "SOKUTOU_TTS_NUM_STEPS in ${SOKUTOU_SERVICE_SRC} must equal IRODORI_TTS_NUM_STEPS in ${SERVICE_SRC}"
  [[ "$(unit_env SOKUTOU_TTS_REFERENCE_WAV "${SOKUTOU_SERVICE_SRC}")" == "${REFERENCE_WAV}" ]] \
    || die "SOKUTOU_TTS_REFERENCE_WAV in ${SOKUTOU_SERVICE_SRC} must be ${REFERENCE_WAV}"
  command -v docker >/dev/null 2>&1 || die "docker is required for the sokutou backend"
  docker image inspect "${SOKUTOU_IMAGE}" >/dev/null 2>&1 \
    || die "docker image not found: ${SOKUTOU_IMAGE}; build it with the command in ${SCRIPT_DIR}/README.md"
fi

sudo install -d -o "${USER}" -g "${USER}" "${DATA_ROOT}" "$(dirname "${REFERENCE_WAV}")"

if [[ ! -f "${REFERENCE_WAV}" ]]; then
  ffmpeg -hide_banner -loglevel error -y \
    -i "${REFERENCE_SRC}" \
    -ac 1 \
    -ar 48000 \
    "${REFERENCE_WAV}"
fi

# The service never builds its voice file: on the CPU it takes ~100 s, and a GPU-built
# voice cloned less closely (2026-10-16 benchmark).
if [[ "${BACKEND}" == "speechcpp" && ! -f "${SPEECHCPP_VOICE}" ]]; then
  mkdir -p "$(dirname "${SPEECHCPP_VOICE}")"
  echo "[INFO] Building ${SPEECHCPP_VOICE} from ${REFERENCE_WAV} on the CPU (about 100 s)"
  "${SPEECHCPP_BIN}" voice "${SPEECHCPP_MODEL}" "${REFERENCE_WAV}" "${SPEECHCPP_VOICE}" --device cpu
fi

if [[ "${BACKEND}" == "sokutou" ]]; then
  sudo install -m 0644 "${SOKUTOU_SERVICE_SRC}" "${SOKUTOU_SERVICE_DST}"
fi
sudo install -m 0644 "${SERVICE_SRC}" "${SERVICE_DST}"
sudo systemctl daemon-reload
# The service wants the container unit only with the sokutou backend (its WantedBy=), so the backend
# line in the service unit is the one switch between engines.
if [[ "${BACKEND}" == "sokutou" ]]; then
  sudo systemctl enable "${SOKUTOU_SERVICE_NAME}"
elif systemctl is-enabled --quiet "${SOKUTOU_SERVICE_NAME}" 2>/dev/null; then
  sudo systemctl disable "${SOKUTOU_SERVICE_NAME}"
fi

if [[ "${BACKEND}" == "sokutou" ]]; then
  echo "[INFO] Installed ${SOKUTOU_SERVICE_DST}"
fi
echo "[INFO] Installed ${SERVICE_DST}"
echo "[INFO] Reference wav: ${REFERENCE_WAV}"
# A running service keeps its old environment until restarted. Restarting cuts off
# avatar speech in progress, so it only happens on request.
if [[ "${RESTART}" == "true" ]]; then
  if [[ "${BACKEND}" == "sokutou" ]]; then
    # Restarting the service leaves the container running, so a changed container unit needs its own restart.
    # A service that was running starts the container if it was not (switching from another backend).
    sudo systemctl try-restart "${SOKUTOU_SERVICE_NAME}"
  elif systemctl is-active --quiet "${SOKUTOU_SERVICE_NAME}"; then
    # Another backend no longer wants the container, which would keep its GPU memory.
    sudo systemctl stop "${SOKUTOU_SERVICE_NAME}"
    echo "[INFO] Stopped ${SOKUTOU_SERVICE_NAME}, which the ${BACKEND:-python} backend does not use"
  fi
  sudo systemctl try-restart "${SERVICE_NAME}"
  echo "[INFO] Restarted ${SERVICE_NAME} if it was running"
elif systemctl is-active --quiet "${SERVICE_NAME}"; then
  echo "[WARN] ${SERVICE_NAME} is still running the previous unit. Restart to apply: sudo systemctl restart ${SERVICE_NAME}" >&2
  if [[ "${BACKEND}" == "sokutou" ]] && systemctl is-active --quiet "${SOKUTOU_SERVICE_NAME}"; then
    echo "[WARN] ${SOKUTOU_SERVICE_NAME} keeps its unit until: sudo systemctl restart ${SOKUTOU_SERVICE_NAME}" >&2
  fi
else
  echo "[INFO] Start with: sudo systemctl start ${SERVICE_NAME}"
fi
if [[ "${BACKEND}" != "sokutou" && "${RESTART}" != "true" ]] && systemctl is-active --quiet "${SOKUTOU_SERVICE_NAME}"; then
  echo "[WARN] ${SOKUTOU_SERVICE_NAME} still holds GPU memory for the sokutou backend: sudo systemctl stop ${SOKUTOU_SERVICE_NAME}" >&2
fi
