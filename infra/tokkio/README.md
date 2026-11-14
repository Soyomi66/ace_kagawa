# Tokkio 5.0 Helpers

このディレクトリは `Tokkio 5.0` を controller 側から起動するための補助物です。Tokkio 本体はここには含めず、NVIDIA 公式の `NVIDIA/ACE` repo を利用します。

## What This Adds

- `prepare_tokkio_workspace.py`
  - `.env` を読んで `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace` 配下を初期化
  - one-click script 用の `my-config.env` を生成
  - `TOKKIO_OPENAI_API_KEY` が空でも、Tokkio 5.0 の過剰な secret validation を避けるために placeholder を補完
  - `TOKKIO_LLM_BASE_URL` / `TOKKIO_LLM_MODEL` を日本語向け `llm-rag` config に反映
  - `TOKKIO_RAG_ENABLED=true` の場合は `TOKKIO_RAG_MODE=auto|always|off`、`TOKKIO_RAG_PROVIDER=local|nvidia`、RAG collection/local DB 設定を `llm-rag` config に反映
  - `NVIDIA-ACE` clone が存在すれば、日本語・標準語向け `llm-rag` パッチを自動適用
- `deploy_tokkio.sh`
  - `envbuild.sh` の `init-config`, `install`, `info`, `uninstall` を叩くラッパ
- `manage_tokkio.sh`
  - install 済み Tokkio 環境の `start`, `stop`, `status`, `restart`, `reapply`, `restart-controller`, `logs` をまとめる運用ラッパ
- `check_tokkio_endpoints.py`
  - UI/API/Grafana と `kubectl get pods -n app` の簡易確認
  - `https://...:30888` が plain HTTP を返す環境では自動で HTTP fallback して protocol mismatch を表示
- `check_tokkio_ngc_access.py`
  - 生成済み `my-config.env` を使って Tokkio chart repo へのアクセス可否を先に確認
  - `Audio2Face-3D` イメージの NVCR 取得権限も確認し、NGC ブラウザ上での license 未承認を早期に検出

## Role Split

- `deploy_tokkio.sh`
  - `init-config`, `install`, `info`, `uninstall` のための controller/deploy ラッパ
- `manage_tokkio.sh`
  - install 済み環境の日常運用で使う起動停止・状態確認ラッパ

初回構築やフル再構築は `deploy_tokkio.sh` と runbook を使い、日々の起動停止や再確認は `manage_tokkio.sh` を使います。

## Quick Start

### Initial Deployment

```bash
cp .env.example .env
python3 prepare_tokkio_workspace.py --env-file .env
```

生成後、`/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env` ができます。

`TOKKIO_OPENAI_API_KEY` を空欄にした場合、生成される `my-config.env` には placeholder 値が入ります。Tokkio 5.0 chart の default `llm_processor` は `NvidiaLLMService` なので、NVIDIA LLM 経路を使う限りこの placeholder は実行時に参照されません。`OpenAILLMService` を使う場合は実際の OpenAI key に差し替えてください。

Tokkio の LLM は既定で hosted NVIDIA NIM の OpenAI-compatible endpoint を使います。`.env` の `TOKKIO_LLM_BASE_URL` は `https://integrate.api.nvidia.com/v1`、`TOKKIO_LLM_MODEL` は `nvidia/nemotron-3-ultra-550b-a55b` です。`TOKKIO_LLM_API_KEY` を空にすると、`TOKKIO_NVIDIA_API_KEY` が Tokkio controller の `NVIDIA_LLM_API_KEY` として使われます。

音声対話では長い内部推論を待たないよう、Nemotron 3 Ultra に `enable_thinking=false`、`temperature=0` を指定します。複雑な推論を優先する別用途では、用途ごとのクライアントから reasoning を明示的に有効化してください。

Hosted model の採用前には、`https://build.nvidia.com/<provider>/<model>` に終了予定の表示がないことを確認してください。`/v1/models` は現時点の提供有無だけを示し、将来の終了予定までは返しません。現在の既定モデルは 2026-07-17 時点で Free Endpoint が利用可能です。以前の `stockmark/stockmark-2-100b-instruct` は hosted Free Endpoint が Deprecated のため、既定値には戻しません。

Hosted NIM を使う場合、`manage_tokkio.sh start|stop|restart` で別途 LLM コンテナを起動・停止する必要はありません。

```bash
TOKKIO_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
TOKKIO_LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
TOKKIO_LLM_API_KEY=
```

RAG を使う場合、既定の `TOKKIO_RAG_PROVIDER=local` では `data/rag/corpus` から作った SQLite index を controller に同期し、検索結果だけを `NvidiaLLMService` 相当の NIM 直呼びへ渡します。`TOKKIO_RAG_MODE=auto` では通常会話は直接 NIM で応答し、資料・論文・詳細質問だけ RAG に回します。`always` にすると全発話を RAG に通し、`off` にすると RAG を使いません。NVIDIA RAG Blueprint を使う場合は `TOKKIO_RAG_PROVIDER=nvidia` にします。

```bash
TOKKIO_RAG_ENABLED=true
TOKKIO_RAG_MODE=auto
TOKKIO_RAG_PROVIDER=local
TOKKIO_RAG_SERVER_URL=http://10.209.1.12:8081/v1
TOKKIO_RAG_COLLECTION_NAME=ace_kagawa
TOKKIO_RAG_MAX_TOKENS=64
TOKKIO_RAG_VDB_TOP_K=12
TOKKIO_RAG_RERANKER_TOP_K=5
TOKKIO_RAG_MULTIMODAL_RERANKER_TOP_K=10
TOKKIO_RAG_ENABLE_RERANKER=true
TOKKIO_RAG_ROUTE_KEYWORDS=論文,文献,出典,根拠,資料,ドキュメント,引用,詳細,詳しく,経歴,業績,研究業績,研究内容,プロジェクト,発表,受賞,特許,EBC,CMC,SiC/SiC,非破壊評価
TOKKIO_RAG_FALLBACK_TO_LLM_ON_ERROR=false
TOKKIO_RAG_HEALTH_URL=http://127.0.0.1:8081/v1/health?check_dependencies=true
TOKKIO_LOCAL_RAG_DB=data/rag/local/local_rag.sqlite
TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH=/code/configs/local_rag.sqlite
TOKKIO_LOCAL_RAG_TOP_K=3
TOKKIO_LOCAL_RAG_MAX_CONTEXT_CHARS=2800
```

RAG対象の知識質問で検索結果が得られない場合やRAG処理が失敗した場合は、直接LLMへフォールバックせず、参照情報を確認できない旨を応答します。例外は、`研究`・`受賞` などの一般語だけで香川先生ドメインに振り分けられ（`香川`・`学長`・`あなた`・`先生` を含まない）、質問の話題語がコーパスにない場合です（例: `量子コンピュータの研究について教えてください`）。これは RAG 対象外としてルーティング段階で通常の LLM に回し、「話題が資料にない・自分の研究と関係づけない・大学固有の数値は推測しない」という注記を付けます。判定の詳細は `infra/rag/README.md` の Routing, Confidence, And Citations を参照してください。

local RAG index の作成は repo root から次を使います。

```bash
python3 infra/rag/build_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite
```

NVIDIA RAG Blueprint を使う場合だけ、repo root から次を使います。

```bash
infra/rag/manage_rag.sh start
infra/rag/ingest_local_corpus.sh --collection ace_kagawa
```

LLM endpoint の疎通確認は repo root から次を使います。

```bash
export NVIDIA_API_KEY=<your-nvidia-nim-api-key>
python3 infra/llm/check_llm_endpoint.py \
  --base-url https://integrate.api.nvidia.com/v1 \
  --model nvidia/nemotron-3-ultra-550b-a55b
```

次に NVIDIA 公式 repo を用意します。

```bash
git clone https://github.com/NVIDIA/ACE.git /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE
```

公式の `config-template.yml` を controller 側へコピーします。

```bash
./deploy_tokkio.sh init-config --env-file .env
```

`/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml` を編集したら、install を実行します。

```bash
python3 check_tokkio_ngc_access.py --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
./deploy_tokkio.sh install --env-file .env
```

`check_tokkio_ngc_access.py` では Helm repo だけでなく `nvcr.io/nim/nvidia/audio2face-3d` の pull 権限も見ます。`412 Precondition Failed` や `Please accept license on the browser` が返る場合は、NGC/NVIDIA のブラウザ画面で `Audio2Face-3D` NIM の利用規約承認がまだ済んでいません。

デプロイ後に install 出力の endpoint を使って確認します。

```bash
python3 check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://<app-ip>:30111 \
  --api-url http://<app-ip>:30888 \
  --grafana-url http://<app-ip>:32300 \
  --tts-base-url http://<app-ip>:8021
```

`--tts-base-url` を付けると、TTS の `/healthz` と短い文（既定「確認しています」）の `/v1/audio/speech` ストリームを確認します。HTTP 200、空でない偶数長の PCM、`X-Sample-Rate-Hz: 16000` を満たさなければ失敗です。`--tts-health-only` で合成を省き、`/healthz` だけを見ます。どれかの確認に失敗すると終了コードは 1 です。

`manage_tokkio.sh` の `status`・`start`・`restart`・`reapply` も、日本語モードでは同じ TTS 確認を行います。合成まで確かめたくない場合は `.env` に `TOKKIO_TTS_SYNTHESIS_CHECK_ENABLED=false` を書きます。ElevenLabs gateway（`:8022`）へ向けている場合は、実行のたびに課金対象の合成リクエストが 1 回発生するため、`false` を推奨します。

`start`・`restart`・`reapply` では、復元した Pod（`tokkio-ingress`、`riva`、`triton` など）が controller より遅れて起動します。そのため最終確認の前に、UI・API・Grafana の確認だけを `TOKKIO_ENDPOINT_WAIT_ATTEMPTS` 回（既定 36）、`TOKKIO_ENDPOINT_WAIT_SECONDS` 秒（既定 5）間隔で繰り返します。待機中は TTS を確認せず、待機後の最終確認で合成を 1 回だけ行います。各 URL の HTTP タイムアウトは 10 秒なので、接続を拒否せずに黙る host では待機が最長で約 20 分になります。`status` は待たずに 1 回だけ確認します。

`start`・`restart` は TTS や RAG の起動確認に失敗しても先に app workload を戻し、最後に終了コード 1 を返します。`reapply` も最終確認に失敗すると終了コード 1 を返します。原因は `[WARN]` 行と、endpoint 確認の JSON で `"ok": false` になっている項目で確認します。

### Day-2 Operations

既存インストール済み環境の起動停止は次を使います。

```bash
./manage_tokkio.sh start --env-file .env
./manage_tokkio.sh status --env-file .env
./manage_tokkio.sh stop --env-file .env
```

`start` は `containerd`, `kubelet`, `nginx`, `coturn` を起動し、endpoint が応答するまで待ってから `kubectl get pods -n app` と endpoint 確認まで流します。`stop` は非破壊停止のみで、`deploy_tokkio.sh uninstall` は呼びません。

現在の `stop` は host 側サービス停止の前に `app` namespace の `Deployment` / `StatefulSet` を `replicas=0` へ落として GPU を使う Tokkio workload を静止します。退避した replica 数は `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/app-workload-replicas.tsv` に保存され、次の `start` で復元されます。

停止待ちでは、`Failed` / `Succeeded` の Pod（以前の ReplicaSet が残した `Error`・`ContainerStatusUnknown`・`Completed` の Pod）は数えません。これらは replica を 0 にしても消えないため、数えると毎回タイムアウトしていました。それでも動作中の Pod が残った場合、`stop` は TTS と host 側サービスの停止まで続けてから、終了コード 1 で終わります。vLLM コンテナ（`ace-vllm-llama31-swallow-8b`、Gemma 4 に切り替えた後は `ace-vllm-gemma4-12b`）は `stop` では止まりません。止めるときは `containerd` を止める前に `docker stop` してください（後に回すと docker が終了を検知できず、状態が `Up` のまま残ります）。

起動だけで戻らない場合:

```bash
./manage_tokkio.sh reapply --env-file .env
```

`ElevenLabs` など secret を更新して controller に再読込させたい場合:

```bash
./manage_tokkio.sh restart-controller --env-file .env
./manage_tokkio.sh logs controller --env-file .env
```

### ローカル LLM を Gemma 4 12B（FP8）に切り替える

2026-10-28 の比較（`docs/audits/llm-candidates-2026-10-28.md`）で Swallow-8B の後継に選んだ構成です。port 8011 と GPU0 の枠（`--gpu-memory-utilization 0.55`、`--max-model-len 16384`）は Swallow と同じです。

モデル名は `gemma-4-12b-it` です。次の 3 か所がそろっている必要があります。

- `infra/llm/docker-compose.gemma4-12b.yml` の `--served-model-name`
- `.env` の `TOKKIO_LLM_MODEL`
- `tokkio_llm.py` の `nvidia_model_input_params`。名前に `gemma-4` を含むと、比較で使った生成設定（温度 0.1、top_p 0.95、top_k 64、最大 512 トークン、思考オフ）を送ります。

前提:

- GPU0 に約 27 GB の空きが要ります。空きが `0.55 × 48 GB` に足りないと vLLM は起動しません。`nvidia-smi` で確かめてください。
- bf16 のままでは枠に入らないため（KV キャッシュが 0.36 GiB しか残らない）、読み込み時に FP8 へ量子化します（`--quantization fp8`）。
- vLLM 0.9.2 は Gemma 4 を読めません。比較で使った版と同じ `vllm/vllm-openai:v0.32.0-cu129` を使います。ただしこのイメージの torchcodec は CUDA 13 用で、`libnvrtc.so.13` が無いため起動に失敗します。そこで torchcodec を外した派生イメージ（`infra/llm/Dockerfile.vllm-cu129`、`ace/vllm-openai:v0.32.0-cu129-no-torchcodec`）をローカルでビルドして使います。
- `docker compose up` / `docker start` には containerd が要ります。Tokkio の `stop` 後は止まっています（`systemctl is-active containerd`）。

切り替え（repo root で実行）:

```bash
docker pull vllm/vllm-openai:v0.32.0-cu129
docker compose -f infra/llm/docker-compose.gemma4-12b.yml build   # 派生イメージ（数秒）
docker stop ace-vllm-llama31-swallow-8b        # 戻すときに使うので削除しない
docker compose -f infra/llm/docker-compose.gemma4-12b.yml up -d
docker logs -f ace-vllm-gemma4-12b             # "Application startup complete" まで。初回は数分
python3 infra/llm/check_llm_endpoint.py --base-url http://127.0.0.1:8011/v1 --model gemma-4-12b-it
```

続けて `.env` の `TOKKIO_LLM_MODEL` を `gemma-4-12b-it` に変えます（`TOKKIO_LLM_BASE_URL` はそのまま）。Tokkio が止まっていれば `./manage_tokkio.sh start --env-file .env`、動いていれば `./manage_tokkio.sh restart-controller --env-file .env` を実行します。どちらも `prepare_tokkio_workspace.py` で config を作り直し、`tokkio_llm.py` と `config.yaml` を controller に同期します。

Tokkio なしのローカル経路（`infra/local_avatar/`）も同じ LLM を使います。`infra/local_avatar/profiles/kagawa.json` の `llm_model` も `gemma-4-12b-it` に変え、起動時の案内用に `LOCAL_AVATAR_LLM_CONTAINER=ace-vllm-gemma4-12b` を設定してください。

戻し方:

```bash
docker stop ace-vllm-gemma4-12b
docker start ace-vllm-llama31-swallow-8b
```

`.env` の `TOKKIO_LLM_MODEL`（ローカル経路を使うなら `kagawa.json` の `llm_model` も）を `llama-3.1-swallow-8b-instruct-v0.5` に戻し、同じく `start` か `restart-controller` を実行します。

## Important Notes

- quickstart の既定構成では `TTS` に `ElevenLabs` を使います
- controller と application を別マシンに分けるのが公式前提ですが、同一ワークステーションでも `SSH` と `passwordless sudo` の条件を満たせば運用できます
- Tokkio の作業領域は `infra/tokkio/workspace` 配下に置きます。このディレクトリは Git 管理外です
- この環境では `UI` は `https://<app-ip>:30111`、`Grafana` は `http://<app-ip>:32300`、`API` は `http://<app-ip>:30888` で応答しました。install 出力が `https` を出しても、そのままでは false negative になる場合があります

## Japanese Customization

`Tokkio 5.0` の `ace-controller` は、公式 image を起動しつつ `/code` を `app-storage-volume` にマウントし、`ACE Configurator` がその volume を `HotReload` します。この repo には `NVIDIA-ACE` clone の `llm-rag` ソースへ日本語標準語向けの `ASR / LLM / TTS` 設定を当てる補助スクリプト [`customize_tokkio_japanese.py`](/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/customize_tokkio_japanese.py) を追加してあり、`prepare_tokkio_workspace.py` からも自動実行されます。

```bash
python3 infra/tokkio/customize_tokkio_japanese.py \
  --ace-repo-dir /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE \
  --llm-base-url https://integrate.api.nvidia.com/v1 \
  --llm-model nvidia/nemotron-3-ultra-550b-a55b
```

通常は個別実行は不要で、`./manage_tokkio.sh start --env-file .env` または `./manage_tokkio.sh reapply --env-file .env` の前に `prepare_tokkio_workspace.py` が自動で同じパッチを当てます。

### 応答開始までの時間（話し終わり判定・最初の句）

話し終わりから最初の声までを縮めるため、`customize_tokkio_japanese.py` の `CONFIG_YAML` で次の 2 点を設定しています。値を変えたら `prepare_tokkio_workspace.py` と `manage_tokkio.sh reapply` で反映します。

- `TurnTaking`: ユーザーの発話を LLM へ送るタイミング。Silero VAD が `vad_stop_secs` 秒の無音を検出し、かつ Riva の確定結果（`asr_stop_history_ms` ミリ秒の無音＋認識処理の後に届く）がそろった時点で送ります。確定結果が VAD より後に届くと、届いてからさらに `aggregation_timeout_s` 秒待ちます。そのため Riva の値は VAD の値より十分小さくしてください。既定値は 0.6 秒 / 500 ms / 0.05 秒です（pipecat の初期値は 0.8 秒 / 500 ms / 0.5 秒）。集約待ちが 0.05 秒なのは、既定の `SpeechCppSTTService` が発話ごとに 1 回、VAD の停止後に結果を返すためです（下の「音声認識」）。VAD 0.6 秒は 2026-10-17 から試行中です。`RivaASRService` に戻すときは VAD も 0.8 秒に戻します。短くするほど、言いよどみの途中で送られやすくなります。
  - Riva を 250 ms にすると、短い間で発話が細切れの確定結果になり、切れ目の文字が落ちます（2026-10-17 の実マイク会話で「こんにちは」→「お」「にちは」、合成音声の質問 160 本で CER 22.7%。500 ms では 5.3%）。500 ms のまま VAD を 0.5 秒にすると、約半数の発話で確定結果が VAD より後に届き、途中までの文が LLM へ送られるので、VAD は 0.8 秒にします。詳細は `docs/audits/speech-cpp-asr-2026-10-17.md`。
- `IrodoriTTSService.start_on_first_clause: true`: 回答の最初の読点までを、文末を待たずに TTS へ送ります。読点の前が英数字・かな漢字で 4 文字未満（「はい、」など）なら切りません。2 文目以降は従来どおり文単位です。英語モード（`QwenTTSService`）には適用しません。

`TurnTaking` の節がない古い config では pipecat の初期値で動き、`start_on_first_clause` がなければ従来どおり文末まで待ちます。

### 会話の切り替え（言い終わり判定・割り込み・話し方）

次の 3 つは `.env` で切り替えます。`TOKKIO_TURN_END` と `TOKKIO_BARGE_IN` の既定は off で、off のときの動きは上の「応答開始までの時間」のとおりです。`TOKKIO_SPOKEN_STYLE` の既定は on です（2026-11-13 に人物設定のプロンプトを会話向けに書き直したときに変えました）。**どれも実マイクでは確かめていません。** 仕組みとオフラインの評価は `docs/audits/turn-taking-2026-11-13.md` にあります。

| `.env` | 値（先頭が既定） | on にしたときの動き |
| --- | --- | --- |
| `TOKKIO_TURN_END` | `vad` / `text` | `text`: VAD は 0.2 秒で止まり、文字起こしの文末で言い終わりを判定します。丁寧な文末（です、ます、か、ください）はすぐ、普通形と短い返事（〜た、はい）は無音 0.6 秒、続きがある形（の、けど、えっと）は無音 0.8 秒で LLM に渡します。アバターの声が出る前に来訪者が話を続けたら、応答を取り消して 1 つの発話にまとめます |
| `TOKKIO_BARGE_IN` | `immediate` / `gated` | `gated`: アバターが話している間は、相づち（うん、はい）と雑音では止まりません。発話が 0.6 秒続くか、短い発話の文字起こしが相づちでなければ止まります。`text` と一緒に使う前提です |
| `TOKKIO_SPOKEN_STYLE` | `on` / `off` | `on`: 人物設定のプロンプトに、話し言葉の文末（「〜ですね」「〜ですよ」）で答える指示を足します。`off`: 「です・ます」だけで答えます |

- 反映するには、`.env` を書き換えてから `./manage_tokkio.sh sync-controller --env-file .env` を実行します。戻すときは値を既定に戻して同じコマンドを実行します。
- 細かい値（待ち時間、相づちの一覧、「そうですね」の扱い）は、生成された `config.yaml` の `TurnTaking` にあります。変えるときは `customize_tokkio_japanese.py` の `CONFIG_YAML` を書き換えます。
- 英語モード（`TOKKIO_TTS_PROCESSOR=QwenTTSService`）では、3 つとも off として設定を作ります。

### 資料にない質問への答え方

人物設定のプロンプト（`customize_tokkio_japanese.py` の `CONFIG_YAML`）と RAG ルーター（`tokkio_rag.py`）は、質問を次の 3 種類に分けて答えます。

| 質問の種類 | 答え方 |
| --- | --- |
| 雑談、一般的な知識、考えや助言 | 資料がなくても、学長・研究者としての考えで答えます |
| 東京工科大学の数値・日程・制度・人名、経歴や業績の細部 | 固定プロフィールか RAG の参考情報にあるときだけ答えます。ないときは、正確なところは手元にないと伝え、大学の窓口や公式サイトを案内します |
| 固定プロフィールにない私生活や体験（好きな食べ物、趣味、家族、研究者になった理由など） | 本人の事実として作りません。AI の自分はそこまで本人から聞いていない、と伝えます |

- RAG の検索結果が質問に合わなかったとき、以前は固定の断り文を返していました。今は LLM に渡し、作ってはいけない範囲を書いた「回答メモ」（`local_rag.py` の `ungrounded_prompt_note`）を付けます。索引の読み込みに失敗したときだけ、固定の文を返します。
- AI かどうかを聞かれたら、本人の分身として学生が開発した AI アバターだと答えます。
- 本番の LLM（Gemma 4 26B）に本番と同じルーターを通して送った結果は `/data/ACE/llm-eval/2026-11-13-conversation/` にあります。実マイクでは確かめていません。
- controller のログでは、`Turn end (`（言い終わりの判定と秒数）、`Turn taken back`（取り消しと結合）、`Backchannel while the avatar speaks`（捨てた相づち）、`Barge-in`（割り込み）の行で動きを追えます。

### 音声認識（Riva / speech.cpp）

`CONFIG_YAML` の `Pipeline.stt_processor` で音声認識を選びます。既定は 2026-10-17 から `SpeechCppSTTService` です。

- `RivaASRService`: クラスタ内の Riva（`conformer-unified-ja-JP` ストリーミング）。途中経過も出ます。`RivaASRService` 節と `TurnTaking.asr_stop_history_ms` を使います。
- `SpeechCppSTTService`: ホストの speech.cpp（`infra/asr/`）。VAD が区切った発話ごとに 1 回、`SpeechCppSTTService.base_url` の OpenAI 互換 API へ送ります（pipecat の `OpenAISTTService`）。途中経過は出ません。2026-11-02 から日本語は Haas-ja-v1.0（port 8024）、英語モードは Qwen3-ASR-1.7B（port 8025）です。Haas は合成音声の質問 160 本で CER 2.72%（Qwen3-ASR は 3.19%、差は有意でない）、FLEURS で 4.47%（同 5.33%、有意に良い）、「青嵐」は 16 回中 14 回（同 16 回）。詳細は `docs/audits/speech-cpp-asr-haas-2026-11-02.md`（Qwen3-ASR を選んだ 10-17 の比較は `docs/audits/speech-cpp-asr-2026-10-17.md`）。

`SpeechCppSTTService` の注意点:

- `model` はサーバーが返す id（`Haas-ja-v1.0`、英語モードは `Qwen3-ASR-1.7B`）です。違うと 404 になります。
- Haas は `prompt` を受け付けません（送ると 400）。「青嵐」はサーバー側の登録語（`ace-speechcpp-asr.service` の `SPEECHCPP_ASR_KEYWORDS`）で書かせます。登録語を増やすと、遠くの話し声に登録語を書き込むことがあるので、数語にとどめます。
- `chunking_strategy: "auto"` は必須です。サーバーの Silero VAD が話している区間だけを認識し、雑音は空文字になります（何も送られません）。外すと Haas は雑音を「うん。」と書き、LLM に渡ります。サーバーに VAD モデルがないと毎回 400 になるので、起動スクリプトは VAD モデルのファイルがなければ起動しません。
- 英語モード（`QwenTTSService`）では `base_url` が port 8025、`model: "Qwen3-ASR-1.7B"`、`language: "en"`、`prompt: ""`、`chunking_strategy: ""` に置き換わります。Haas は日本語専用（`language=en` は 400）のためです。プロンプトだけの文字起こしを捨てる処理（`tokkio_speechcpp_stt.py`）は、`prompt` を設定したときだけ働きます。
- 結果は VAD の停止後に届くので、`TurnTaking.aggregation_timeout_s` がそのまま上乗せされます。発話ごとに 1 回しか届かないため 0.05 秒にしています。話し終わりから LLM までは約 VAD 停止（0.6 秒）＋認識（約 0.05 秒）＋0.05 秒です。
- `manage_tokkio.sh` の `start` / `restart` は、生成済み config.yaml が `SpeechCppSTTService` を選んでいるときだけ ASR の unit を起動します。`SpeechCppSTTService.language` が `en` なら `ace-speechcpp-asr-en.service`（`http://127.0.0.1:8025/health` を待つ）、それ以外は `ace-speechcpp-asr.service`（`http://127.0.0.1:8024/health`）で、もう一方が動いていれば止めます。`stop` は両方を止めます。Riva はそのまま残るので、`stt_processor` を戻して `sync-controller` すれば元に戻せます。
- Qwen3-ASR に戻すには、`CONFIG_YAML` と `SPEECHCPP_STT_JA_YAML` を `model: "Qwen3-ASR-1.7B"`・`prompt: "青嵐、SEIRAN"`・`chunking_strategy: ""` にし、`ace-speechcpp-asr.service` のモデルを Qwen3-ASR に替えて VAD と登録語の行を消します（`infra/asr/README.md`）。

### オープンキャンパス挨拶

マイクまたはテキスト入力から「オープンキャンパスのご挨拶をお願いします」「オープニングをお願いします」「台本を読んでください」のいずれかを伝えると、AI学長が八王子キャンパス向けの挨拶を文章単位で読み上げます。この定型発話は LLM や RAG を呼ばないため、原稿が言い換えられることはありません。

冒頭の「皆さん、こんにちは。」だけは、`/data/ACE/irodori/fixed-phrases/open_campus_greeting_16k_mono.pcm` が存在する場合に事前生成済み音声を返します。ファイルがない場合は通常のIrodori生成へフォールバックします。

詳細な手順と前提は [`docs/operations/tokkio-japanese-customization.md`](/home/kyano/workspace/ACE/ace_kagawa/docs/operations/tokkio-japanese-customization.md) を参照してください。

### 英語シンポジウム挨拶

英語音声へ切り替える場合は `.env` で `TOKKIO_TTS_PROCESSOR=QwenTTSService` を指定し、`TOKKIO_QWEN_TTS_ENABLED` は `true` にするか空にします（空なら `true` 扱い。`QwenTTSService` なのに明示的に `false` だと `start`・`restart` はエラーで止まります）。互換性のためselector名はQwenのままですが、実backendは `en-tts/indextts_tts_server.py` のIndexTTS-2.5です。香川先生の英語参照音声を使い、サーバー側で22.05 kHzの生成音声をAudio2Faceの16 kHz契約へ帯域制限つきでresampleして返します。日本語モードに戻したときに `ace-indextts-tts.service` が動いていると、GPU1 を使い続けるため `status` が警告します。

```bash
en-tts/install_indextts_tts_service.sh
infra/tokkio/manage_tokkio.sh reapply --env-file infra/tokkio/.env
```

英語モードではASRも `en-US` に切り替わり、自動の挨拶・退出音声は無効になります。WebUIで `Start` と発話または入力すると固定英語原稿を7文に分けてIndexTTSへ渡し、読み上げ中の `Stop` は即時中断して返答音声を出しません。`Please read the symposium opening.` も利用できます。事前生成WAVの直接再生は使いません。
