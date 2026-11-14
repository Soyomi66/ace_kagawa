# speech.cpp ASR service (kagawa)

Tokkio の `SpeechCppSTTService`（`infra/tokkio/tokkio_speechcpp_stt.py`）から呼ばれる host-side の音声認識です。speech.cpp v0.10.1 の `speech serve` で認識モデルを常駐させ、OpenAI 互換の `POST /v1/audio/transcriptions` を受けます。`GET /health` は `{"status":"ok"}` を返します。

| unit | モデル | port | 使う場面 |
| --- | --- | --- | --- |
| `ace-speechcpp-asr.service` | Haas-ja-v1.0（Q8_0）＋Silero VAD＋登録語「青嵐」 | 8024 | 日本語（既定、2026-11-02 から） |
| `ace-speechcpp-asr-en.service` | Qwen3-ASR-1.7B（Q8_0） | 8025 | 英語モード（`TOKKIO_TTS_PROCESSOR=QwenTTSService`）。Haas は日本語専用で `language=en` が 400 になるため |

`speech serve` は認識モデルを 1 つしか載せられないので、言語ごとに unit を分けています。`manage_tokkio.sh` は生成済み config.yaml の `SpeechCppSTTService.language` を見て、どちらか一方だけを起動します。

選んだ理由と比較結果は `docs/audits/speech-cpp-asr-haas-2026-11-02.md`（Haas）と `docs/audits/speech-cpp-asr-2026-10-17.md`（Qwen3-ASR と Riva）にあります。

## 導入

```bash
./install_speechcpp_asr_service.sh            # 2 つの unit を /etc/systemd/system に置く（sudo）
sudo systemctl start ace-speechcpp-asr.service
curl -s http://127.0.0.1:8024/health
```

installer は、unit に書かれたバイナリ・モデル・VAD モデル・起動スクリプトがそろっていなければ、sudo の前に止まります。動いている unit に設定を反映するときは `--restart` を付けます（認識中の要求は落ちます）。

Tokkio 側は `infra/tokkio/customize_tokkio_japanese.py` の `Pipeline.stt_processor` を `SpeechCppSTTService` にし、`prepare_tokkio_workspace.py` と `manage_tokkio.sh sync-controller` で反映します（`infra/tokkio/README.md` の「音声認識」）。以後は `manage_tokkio.sh start` / `stop` が unit も起動・停止します。

## 設定（`ace-speechcpp-asr.service`）

| 変数 | 値 | 備考 |
| --- | --- | --- |
| `SPEECHCPP_ASR_BIN` | `/data/ACE/speech-cpp/runtime/v0.10.1/release/speech` | 公式 Linux x64 Vulkan リリース。Haas は 0.9.4 以上、サーバー既定の `--keywords` は 0.10.0 以上が必要。本番 TTS（v0.8.7）とは別 |
| `SPEECHCPP_ASR_MODEL` | `…/Haas-616M-ja-v1.0-Q8_0.gguf` | `sakasegawa/Haas-ja-v1.0-GGUF` revision `6b965d5`。サーバー id は `Haas-ja-v1.0` |
| `SPEECHCPP_ASR_VAD_MODEL` | `…/silero-vad-309K-v6.2-F32.gguf` | Tokkio が送る `chunking_strategy=auto` に必要。ないと要求が毎回 400 になるので、ファイルがなければ起動しない |
| `SPEECHCPP_ASR_KEYWORDS` | `青嵐=せいらん,セイラン` | サーバー既定の登録語（`表記=読み,読み`）。要求に `keywords` がなくても効く |
| `SPEECHCPP_ASR_DEVICE` | `Vulkan1` | このマシンでは nvidia-smi の GPU 0。VRAM 約 0.7 GiB |
| `SPEECHCPP_ASR_GPU_UUID` | `GPU-d4c17c05-…` | 起動後に nvidia-smi で照合し、違う GPU なら止める |
| `SPEECHCPP_ASR_THREADS` | 4 | 既定（全コア）だと CPU を奪い合い、22 秒の音声に 20〜30 秒かかった |
| `SPEECHCPP_ASR_HOST` / `_PORT` | `0.0.0.0` / `8024` | controller の pod から `http://10.209.1.12:8024/v1` で呼ぶ。認証はない |

`ace-speechcpp-asr-en.service` は同じ変数で、モデルが `Qwen3-ASR-1.7B-Q8_0.gguf`（VRAM 約 2.3 GiB）、port が 8025、VAD と登録語はなしです。

言語・プロンプト・`chunking_strategy` はサーバーに固定せず、Tokkio が要求ごとに送ります（日本語は `ja`・プロンプトなし・`auto`、英語モードは `en`・プロンプトなし・なし）。

## Qwen3-ASR に戻す

1. `ace-speechcpp-asr.service` の `SPEECHCPP_ASR_MODEL` を `ace-speechcpp-asr-en.service` と同じ Qwen3-ASR のパスにし、`SPEECHCPP_ASR_VAD_MODEL` と `SPEECHCPP_ASR_KEYWORDS` の行を消す
2. `./install_speechcpp_asr_service.sh --restart`
3. Tokkio 側の設定を戻す（`infra/tokkio/README.md` の「音声認識」の最後の項目）
