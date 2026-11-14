from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASR_DIR = ROOT / "infra" / "asr"
UNIT = ASR_DIR / "ace-speechcpp-asr.service"
EN_UNIT = ASR_DIR / "ace-speechcpp-asr-en.service"
WRAPPER = ASR_DIR / "run_speechcpp_asr.sh"
INSTALLER = ASR_DIR / "install_speechcpp_asr_service.sh"
BASH = shutil.which("bash") or "/bin/bash"


def unit_env(unit: Path = UNIT) -> dict[str, str]:
    values = {}
    for line in unit.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^Environment="?([A-Z0-9_]+)=(.*?)"?$', line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class SpeechCppAsrUnitTests(unittest.TestCase):
    def test_units_start_the_wrapper_and_set_what_it_reads(self) -> None:
        wrapper = WRAPPER.read_text(encoding="utf-8")
        for unit in (UNIT, EN_UNIT):
            with self.subTest(unit=unit.name):
                self.assertIn(f"ExecStart={WRAPPER}\n", unit.read_text(encoding="utf-8"))
                # The wrapper's SIGTERM trap exits 143; without this a stop leaves the unit "failed".
                self.assertIn("SuccessExitStatus=143\n", unit.read_text(encoding="utf-8"))
                env = unit_env(unit)
                for key in env:
                    self.assertIn(key, wrapper, key)
                self.assertEqual(env["SPEECHCPP_ASR_THREADS"], "4")
                self.assertEqual(env["SPEECHCPP_ASR_DEVICE"], "Vulkan1")
                # Haas needs 0.9.4 or later.
                self.assertIn("/runtime/v0.10.1/", env["SPEECHCPP_ASR_BIN"])

    def test_japanese_unit_serves_haas_with_vad_and_the_seiran_keyword(self) -> None:
        env = unit_env(UNIT)

        self.assertTrue(env["SPEECHCPP_ASR_MODEL"].endswith("/Haas-616M-ja-v1.0-Q8_0.gguf"))
        # chunking_strategy=auto, which the Japanese config sends, needs it (a 400 without).
        self.assertTrue(env["SPEECHCPP_ASR_VAD_MODEL"].endswith("/silero-vad-309K-v6.2-F32.gguf"))
        self.assertEqual(env["SPEECHCPP_ASR_KEYWORDS"], "青嵐=せいらん,セイラン")

    def test_english_unit_serves_qwen_without_vad_or_keywords(self) -> None:
        env = unit_env(EN_UNIT)

        self.assertTrue(env["SPEECHCPP_ASR_MODEL"].endswith("/Qwen3-ASR-1.7B-Q8_0.gguf"))
        self.assertNotIn("SPEECHCPP_ASR_VAD_MODEL", env)
        self.assertNotIn("SPEECHCPP_ASR_KEYWORDS", env)

    def test_unit_ports_match_the_controller_urls_and_manage_health_urls(self) -> None:
        manage = (ROOT / "infra" / "tokkio" / "manage_tokkio.sh").read_text(encoding="utf-8")
        customize = (ROOT / "infra" / "tokkio" / "customize_tokkio_japanese.py").read_text(encoding="utf-8")
        ja_port = unit_env(UNIT)["SPEECHCPP_ASR_PORT"]
        en_port = unit_env(EN_UNIT)["SPEECHCPP_ASR_PORT"]

        self.assertNotEqual(ja_port, en_port)
        self.assertIn(f'DEFAULT_SPEECHCPP_ASR_HEALTH_URL="http://127.0.0.1:{ja_port}/health"', manage)
        self.assertIn(f'DEFAULT_SPEECHCPP_ASR_EN_HEALTH_URL="http://127.0.0.1:{en_port}/health"', manage)
        self.assertIn(f'DEFAULT_SPEECHCPP_ASR_EN_SERVICE="{EN_UNIT.name}"', manage)
        self.assertEqual(customize.count(f'    base_url: "http://10.209.1.12:{ja_port}/v1"\n'), 2)
        self.assertEqual(customize.count(f'    base_url: "http://10.209.1.12:{en_port}/v1"\n'), 1)


class SpeechCppAsrWrapperTests(unittest.TestCase):
    """Runs the wrapper against fake speech, curl and nvidia-smi; nothing touches the GPU."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.log = self.root / "speech.log"
        self.pid_file = self.root / "speech.pid"
        write_executable(
            self.root / "speech",
            '#!/usr/bin/env bash\necho "$$" > "${SHIM_PID_FILE}"\necho "speech $*" >> "${SHIM_LOG}"\n'
            "trap 'exit 0' TERM\nwhile true; do sleep 0.1; done\n",
        )
        write_executable(
            self.bin_dir / "curl",
            '#!/usr/bin/env bash\n[[ -s "${SHIM_PID_FILE}" ]] || exit 7\nexit "${SHIM_CURL_RC:-0}"\n',
        )
        write_executable(
            self.bin_dir / "nvidia-smi",
            '#!/usr/bin/env bash\necho "1, GPU-someone-else"\necho "$(cat "${SHIM_PID_FILE}"), ${SHIM_GPU_UUID}"\n',
        )

    def start(self, **env_overrides: str) -> subprocess.Popen[str]:
        env = {key: value for key, value in os.environ.items() if not key.startswith(("SPEECHCPP_", "SHIM_"))}
        env.update(
            PATH=f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}",
            SHIM_LOG=str(self.log),
            SHIM_PID_FILE=str(self.pid_file),
            SHIM_GPU_UUID="GPU-expected",
            SPEECHCPP_ASR_BIN=str(self.root / "speech"),
            SPEECHCPP_ASR_MODEL="/models/Qwen3-ASR-1.7B-Q8_0.gguf",
            SPEECHCPP_ASR_DEVICE="Vulkan1",
            SPEECHCPP_ASR_GPU_UUID="GPU-expected",
            SPEECHCPP_ASR_HOST="0.0.0.0",
            SPEECHCPP_ASR_PORT="8024",
            SPEECHCPP_ASR_READY_ATTEMPTS="20",
        )
        env.update(env_overrides)
        return subprocess.Popen([BASH, str(WRAPPER)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_for_output(self, proc: subprocess.Popen[str], timeout: float = 10.0) -> tuple[str, str]:
        try:
            return proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise

    def speech_is_running(self) -> bool:
        pid = int(self.pid_file.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def test_stops_the_server_on_another_gpu(self) -> None:
        proc = self.start(SHIM_GPU_UUID="GPU-other")

        out, err = self.wait_for_output(proc)

        self.assertEqual(proc.returncode, 1, out + err)
        self.assertIn("is on GPU 'GPU-other', not GPU-expected", err)
        time.sleep(0.3)
        self.assertFalse(self.speech_is_running())

    def test_keeps_serving_on_the_expected_gpu_and_stops_with_systemd(self) -> None:
        proc = self.start()
        deadline = time.monotonic() + 10
        while not self.log.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)

        self.assertIsNone(proc.poll(), "the wrapper must stay up while the server runs")
        self.assertIn(
            "speech serve /models/Qwen3-ASR-1.7B-Q8_0.gguf --device Vulkan1 --threads 4 --host 0.0.0.0 --port 8024",
            self.log.read_text(),
        )
        proc.send_signal(signal.SIGTERM)
        out, err = self.wait_for_output(proc)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("is on GPU-expected", out)
        time.sleep(0.3)
        self.assertFalse(self.speech_is_running())

    def test_passes_the_vad_model_and_keywords_when_the_unit_sets_them(self) -> None:
        vad = self.root / "silero-vad.gguf"
        vad.write_bytes(b"")
        proc = self.start(
            SPEECHCPP_ASR_MODEL="/models/Haas-616M-ja-v1.0-Q8_0.gguf",
            SPEECHCPP_ASR_VAD_MODEL=str(vad),
            SPEECHCPP_ASR_KEYWORDS="青嵐=せいらん,セイラン",
        )
        deadline = time.monotonic() + 10
        while not self.log.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)

        self.assertIn(
            f"speech serve /models/Haas-616M-ja-v1.0-Q8_0.gguf {vad} --device Vulkan1 --threads 4 "
            "--host 0.0.0.0 --port 8024 --keywords 青嵐=せいらん,セイラン",
            self.log.read_text(encoding="utf-8"),
        )
        proc.send_signal(signal.SIGTERM)
        self.wait_for_output(proc)

    def test_refuses_to_start_when_the_vad_model_is_missing(self) -> None:
        # Without it every chunking_strategy request is a 400: Tokkio would hear nothing.
        proc = self.start(SPEECHCPP_ASR_VAD_MODEL=str(self.root / "missing.gguf"))

        out, err = self.wait_for_output(proc)

        self.assertEqual(proc.returncode, 1, out + err)
        self.assertIn("speech.cpp VAD model not found", err)
        self.assertFalse(self.pid_file.exists())

    def test_gives_up_when_the_server_never_becomes_ready(self) -> None:
        write_executable(self.bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")
        proc = self.start(SHIM_CURL_RC="7", SPEECHCPP_ASR_READY_ATTEMPTS="3")

        out, err = self.wait_for_output(proc)

        self.assertEqual(proc.returncode, 1, out + err)
        self.assertIn("did not become ready on port 8024", err)


class SpeechCppAsrInstallerTests(unittest.TestCase):
    """Runs the installer on copies of both units whose files point into a temporary directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.asr_dir = self.root / "asr"
        self.asr_dir.mkdir()
        shutil.copy2(INSTALLER, self.asr_dir / INSTALLER.name)
        self.fake_bin = self.root / "speech"
        write_executable(self.fake_bin, "#!/usr/bin/env bash\n")
        for name in ("model.gguf", "vad.gguf"):
            (self.root / name).write_bytes(b"")
        shims = self.root / "bin"
        shims.mkdir()
        self.log = self.root / "calls.log"
        write_executable(shims / "sudo", f'#!/usr/bin/env bash\necho "sudo $*" >> "{self.log}"\n')
        write_executable(shims / "systemctl", "#!/usr/bin/env bash\nexit 3\n")
        self.path = f"{shims}{os.pathsep}{os.environ.get('PATH', '')}"

    def write_unit(self, unit: Path, **paths: str) -> None:
        text = unit.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^Environment=SPEECHCPP_ASR_BIN=.*$", f"Environment=SPEECHCPP_ASR_BIN={self.fake_bin}", text)
        for key, value in {"MODEL": f"{self.root}/model.gguf", "VAD_MODEL": f"{self.root}/vad.gguf", **paths}.items():
            text = re.sub(rf"(?m)^Environment=SPEECHCPP_ASR_{key}=.*$", f"Environment=SPEECHCPP_ASR_{key}={value}", text)
        (self.asr_dir / unit.name).write_text(text, encoding="utf-8")

    def install(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [BASH, str(self.asr_dir / INSTALLER.name)],
            env={**os.environ, "PATH": self.path},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_installer_installs_both_units(self) -> None:
        self.write_unit(UNIT)
        self.write_unit(EN_UNIT)

        result = self.install()

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.log.read_text(encoding="utf-8")
        for unit in (UNIT, EN_UNIT):
            self.assertIn(f"/etc/systemd/system/{unit.name}", calls)
        self.assertIn("sudo systemctl daemon-reload", calls)

    def test_installer_stops_before_sudo_when_the_model_is_missing(self) -> None:
        self.write_unit(UNIT, MODEL=f"{self.root}/missing.gguf")
        self.write_unit(EN_UNIT)

        result = self.install()

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("speech.cpp model not found", result.stderr)
        self.assertFalse(self.log.exists())

    def test_installer_stops_before_sudo_when_the_vad_model_is_missing(self) -> None:
        self.write_unit(UNIT, VAD_MODEL=f"{self.root}/missing-vad.gguf")
        self.write_unit(EN_UNIT)

        result = self.install()

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("speech.cpp VAD model not found", result.stderr)
        self.assertFalse(self.log.exists())

    def test_installer_stops_before_sudo_when_the_english_unit_is_missing(self) -> None:
        self.write_unit(UNIT)

        result = self.install()

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("service unit not found", result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
