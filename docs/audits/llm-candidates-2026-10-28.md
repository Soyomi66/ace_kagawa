# 現行 Swallow-8B と候補 LLM の比較（2026-10-28）

前提の調査: `docs/audits/llm-lfm3-2026-10-28.md`（LFM3 の配信経路、ライセンス、最初の 10 問比較）。

## 結論

- **Swallow から替えるなら Gemma 4 12B を勧める。**
  - 本番相当の総合は 3.97（Swallow 3.10）で、使えない応答は 63 問中 4（Swallow 14）。参考情報への忠実さは全候補で最も高い（4.63）。
  - 知識も JMMLU 0.73（Swallow 0.60）と高い。
  - 応答完了までの中央値は 0.85 秒で、Swallow の 0.99 秒より速い。答えが短いため。
  - ライセンスは Apache 2.0。
  - 本番と同じ枠（`--gpu-memory-utilization 0.55`、`--max-model-len 16384`）では、bf16 は KV キャッシュが足りず起動しない。vLLM の FP8 量子化（`--quantization fp8`）なら起動し、KV は 91,331 トークン（16,384 トークンの会話 5.6 本分）取れる。
  - FP8 にしても知識と読解はほぼ変わらない（JMMLU 0.726 → 0.716、JCQA 0.946 → 0.942、JSQuAD 同等）。応答完了は 0.56 秒とむしろ速くなる。ただし 100 字を超える応答が 126 件中 5 → 13 に増えた。`max_tokens` で切ると文の途中の断片が読み上げられるので、長さはシステムプロンプトで抑える（`max_tokens` は比較と同じ 512 で、暴走を止めるためだけに使う）。
- **品質の最高点は Qwen Flash（Strata）だが、今の構成には入らない。**
  - 本番相当 4.10、ELYZA 4.32、JMMLU 0.81 で、どれも最高。
  - ただし 48 GB の GPU を 1 枚まるごと使い、Tokkio の renderer と同居できない。
  - ライセンスは Qwen Community License 1.0 で、エンジンも Strata 専用。
  - 「Strata なら速い」は生成速度（141 tok/s）については正しい。ただ最初の文字までは 0.90 秒で最も遅く、短い応答では利点が出ない。共通のシステムプロンプトも毎回読み直しているため（ログで再利用 0）。
- **Qwen 27B は Gemma 12B と互角だが、本番には入りにくい。** 知識は 0.76 と強い。一方で、FP8 でも重みが 29 GB あって GPU0 の枠に入らず、最初の文字までも 0.59 秒と遅い。
- **GPU に余裕がないときの次善は Gemma E4B。** 総合 3.78 で、応答完了 0.39 秒と最速。ただし知識は Swallow 並みで、使えない応答が 9 ある。
- **LFM3 と Nemotron JP は勧めない。** どちらも本番相当で Swallow を下回った。冗長で、作り話もする。
- どの候補も本番の vLLM 0.9.2 では動かない。入れ替えるには、vLLM の更新と、Tokkio が渡す生成パラメータの修正（思考オフ、温度、`max_tokens`）が要る。

## 対象

| 略称 | モデル | 大きさ | ライセンス | 動かし方 |
|---|---|---|---|---|
| Swallow | `tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5`（本番） | 8B | Llama 3.1 Community License + Gemma Terms of Use | transformers 5.21.0 と vLLM 0.32.0 の両方 |
| LFM3 | `LiquidAI/LFM3-1.7B` | 1.7B | LFM Open License v1.0 | transformers 5.21.0（vLLM 非対応） |
| Gemma 12B | `google/gemma-4-12B-it` | 12B | Apache 2.0 | vLLM 0.32.0 |
| Gemma E4B | `google/gemma-4-E4B-it` | 実効 4B | Apache 2.0 | vLLM 0.32.0 |
| Nemotron JP | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` | 9B | NVIDIA Nemotron Open Model License | vLLM 0.32.0（`mamba_ssm_cache_dtype=float32`） |
| Qwen 27B | `Qwen/Qwen3.8-27B-FP8` | 27B（FP8） | Apache 2.0 | vLLM 0.32.0（`max_num_seqs=128`） |
| Qwen Flash（Strata） | Qwen3.8-Flash-Next IQ3_S（Strata 版 GGUF） | 総 125B・活性 6B（3 bit） | Qwen Community License 1.0（`qwen-community-1.0`） | Strata 0.1.52、CUDA 13 エンジン、GPU1 1 枚 + CPU メモリ、`--tune quick`（結果は標準設定のまま） |

- Strata は Qwen3.8-27B に対応していないため、Strata で動く Qwen として Qwen3.8-Flash-Next を比べた。
- GPU は RTX 6000 Ada（48 GB）。NVIDIA ドライバーは途中で 570.86.15 から 580.196.07 に上げた（Strata の CUDA エンジンが 575.51 以上を要求するため）。生成結果への影響はないものとして扱う。
- 思考（reasoning）は全モデルで切った（`enable_thinking: false`）。LFM3 だけ思考ありも試した。

## 調べ方

評価一式は `/data/ACE/llm-eval/`。スクリプトは `scripts/`、生成結果は `results/*.jsonl`、採点は `judge/`。4 つの層で比べた。

| 層 | データ | 件数 | 生成設定 | 見るもの |
|---|---|---|---|---|
| 1. 本番相当 | `infra/rag/evaluation_cases.json` を本番の振り分けで再現 + 追加ケース | 63 問 × 2 回 | 下記 | 日本語の自然さ、人物像、長さ、参考情報への忠実さ、読み上げ適性 |
| 2. 日本語の指示追従 | ELYZA-tasks-100 | 100 問 | greedy、最大 1024 トークン | 5 段階評価（ELYZA の基準） |
| 3. 読解 | JSQuAD validation（答えのある問題から無作為） | 300 問 | greedy、最大 48 トークン | 完全一致、文字単位 F1、正解を含む割合 |
| 4. 知識・常識 | JMMLU（科目ごと 20 問）、JCommonsenseQA validation | 1,120 問、1,119 問 | 下記 | 正答率 |

- 層 1 のサンプリング: Swallow は `temperature 0.6, top_p 0.9`（本番の現状）と `temperature 0.1, top_p 0.9`（10-17 監査の修正案）。Gemma は `0.1, top_p 0.95, top_k 64`、Qwen は `0.1, top_p 0.8, top_k 20`、LFM3 は `0.1, top_k 50, repetition_penalty 1.1`、Nemotron はカード推奨の greedy（1 回）。
- 層 1 の振り分けは `tokkio_rag.py` の `_rag_route` と `_stream_local_rag_response` を `local_rag.py` の同じ関数で再現した。本番で LLM を通らない 17 件（名前・年齢の定型応答 2、参考情報からの直接応答 8、根拠不足で定型の断り 7）は除いた。
- 層 4 は 3 通りで測った。
  - 対数尤度: 最初のトークンで選択肢記号の確率を比べる（vLLM は上位 20 候補から読む）。
  - 「正解は」前置き: 応答の頭に「正解は」を置いてから同じく比べる。
  - 生成: 最大 8 トークン生成させ、最初に出た記号を答えとみなす。Strata は確率を返さないので、この方式だけで比べる。
- 「記号だけで答えて」という指示を守るかを JCQA 先頭 100 問で数えた（最初の 2 文字に記号があるか）。
- 層 1 と層 2 は Claude のサブエージェントが盲検で採点した。応答のラベルは問題ごとに無作為に入れ替え、対応表（`judge/SECRET_key.json`）は採点後に照合した。
- 遅延は、1 モデルずつ同じ GPU1 に OpenAI 互換サーバーを立て、本番相当 63 問の 3 問おき（21 問）を 1 件ずつストリーミングで送って測った。
- 本番の vLLM・Tokkio・`.env` には触れていない。

## 結果

### まとめ

| | Swallow（現行） | Gemma 12B | Gemma E4B | Qwen 27B | Qwen Flash（Strata） | Nemotron JP | LFM3 |
|---|---|---|---|---|---|---|---|
| 本番相当 総合（盲検、1〜5） | 3.10 | 3.97 | 3.78 | 4.03 | **4.10** | 2.86 | 1.73 |
| 本番相当 平均順位（8 構成中） | 5.37 | 3.16 | 3.52 | 3.14 | **2.73** | 5.62 | 7.06 |
| 本番相当 使えない応答（総合 2 以下、63 問中） | 14 | **4** | 9 | **4** | 6 | 24 | 54 |
| ELYZA 点（盲検、1〜5） | 3.30 | 4.26 | 3.96 | 4.12 | **4.32** | 3.35 | 2.35 |
| ELYZA 日本語の自然さ（1〜5） | 4.37 | 4.64 | 4.57 | 4.67 | **4.68** | 3.97 | 2.94 |
| JMMLU（対数尤度） | 0.600 | 0.726 | 0.580 | 0.755 | **0.810** | 0.603 | 0.396 |
| JCommonsenseQA（対数尤度） | 0.927 | 0.946 | 0.928 | 0.949 | **0.971** | 0.890 | 0.706 |
| JSQuAD 完全一致 / 正解を含む | 0.853 / 0.910 | 0.833 / **0.943** | 0.840 / 0.927 | **0.870** / 0.937 | 0.860 / 0.930 | 0.793 / 0.917 | 0.580 / 0.893 |
| 記号だけで答えた数（100 問） | 100 | 100 | 100 | 100 | 100 | 100 | 31 |
| 本番相当の字数（中央値） | 78 | 43 | 41.5 | 40 | 47.5 | 117 | 149.5 |

- Swallow の本番相当は本番の現状（`temperature 0.6`）の値。`temperature 0.1` にしても総合 3.10 で変わらず、使えない応答はむしろ 18 に増えた。温度を下げるだけでは作り話は減らない。
- Swallow は transformers と vLLM でほぼ同じ結果だった（JMMLU 0.600 / 0.586、JCQA 0.927 / 0.924、本番相当の字数 78 / 75.5）。エンジンの違いによる差は小さい。
- 上位 3 つ（Qwen Flash、Qwen 27B、Gemma 12B）の差は、採点のぶれと同じくらい小さい。Swallow との差（本番相当で約 0.9 点、ELYZA で約 1 点）ははっきりしている。

### 層 1: 本番相当（63 問、盲検）

観点別の平均（1〜5）:

| 構成 | 自然さ | 人物像 | 長さ・形式 | 忠実さ | 読み上げ | 総合 |
|---|---|---|---|---|---|---|
| Qwen Flash（Strata） | 4.57 | 4.59 | **4.83** | 4.49 | 4.68 | **4.10** |
| Qwen 27B | 4.60 | 4.59 | 4.67 | 4.48 | 4.70 | 4.03 |
| Gemma 12B | **4.63** | **4.60** | 4.73 | **4.63** | 4.70 | 3.97 |
| Gemma E4B | 4.59 | 4.57 | 4.73 | 4.38 | **4.76** | 3.78 |
| Swallow（t0.6、現行） | 4.27 | 4.37 | 3.27 | 4.10 | 4.06 | 3.10 |
| Swallow（t0.1） | 4.32 | 4.30 | 3.27 | 3.89 | 3.98 | 3.10 |
| Nemotron JP | 4.16 | 4.25 | 2.71 | 3.63 | 3.62 | 2.86 |
| LFM3 | 2.90 | 2.41 | 1.92 | 3.24 | 2.89 | 1.73 |

- Swallow の一番の弱点は長さと形式（3.27）。指示は通常 40〜60 字だが、Swallow は中央値 78 字で、60 字超えが 126 件中 78 件。候補の上位 4 つは 12〜24 件。
- 参考情報への忠実さは Gemma 12B が最も高い（4.63）。
- 作り話の典型（Swallow の「使えない」判定から抜き取り）: 「今日の天気は晴れ、最高 25 度」、「2024 年のオープンキャンパスは 8 月 3〜5 日」、好物を「きのこ料理」と創作、英語で返答、「私の倫理規定に反します」と AI 的な語り口。
- 上位の候補にも作り話はある。Qwen Flash と Qwen 27B は、八王子キャンパスの行き方を参考情報なしで答えて路線を誤った。Gemma 12B は学費表の別学部の列を読み違えた。
- カテゴリ別（各 2〜4 問なので傾向だけ）: 確認できない個人的事実を聞く「unknown」で、Swallow と Nemotron は 1.0（作り話）、Qwen Flash は 5.0、Qwen 27B は 4.0、Gemma は 3.0。不適切な依頼への応じ方「inappropriate」は、Swallow 1.67〜2.0、Gemma・Qwen 27B 4.3〜4.7。

### 層 2: ELYZA-tasks-100（盲検）

- 点数順に Qwen Flash 4.32、Gemma 12B 4.26、Qwen 27B 4.12、Gemma E4B 3.96、Nemotron 3.35、Swallow 3.30、LFM3 2.35。2 点以下は Swallow 25 問、Gemma 12B 6 問、Qwen Flash 5 問。
- 応答の長さ（中央値）は Swallow 296 字、ほかは 480〜690 字。ELYZA の基準は「説明や配慮があり役に立つ」を 5 点とするので長い応答が有利になりうる。採点者には「長さだけで決めない」と指示したが、この偏りは残っている可能性がある。

### 層 3・4: 読解・知識

- 知識（JMMLU）は Qwen Flash 0.81 > Qwen 27B 0.76 > Gemma 12B 0.73 > Swallow・Nemotron・Gemma E4B（0.58〜0.60）> LFM3 0.40。
- 3 通りの測り方（対数尤度、「正解は」前置き、生成）の差は、vLLM で動かしたモデルでは 0.01〜0.04 に収まった（例: Gemma 12B の JMMLU 0.726 / 0.724 / 0.724）。Strata は対数尤度だけで測った（API で応答の頭を固定できないので前置き式は省き、生成式は時間の都合で打ち切った）。
- 読解（JSQuAD）は LFM3 以外は互角（正解を含む割合 0.91〜0.94）。

### 遅延

GPU1 に 1 モデルずつサーバーを立て、本番相当 21 問を 1 件ずつストリーミングで送った（`results/latency/`）。プロンプトは中央値 1,130〜1,430 トークン。vLLM は共通のシステムプロンプトをキャッシュから再利用する（prefix caching、既定で有効）。

| モデル | 最初の文字まで（中央値 / 最大） | 応答完了まで（中央値 / 最大） | 生成 tok/s | 出力トークン（中央値） |
|---|---|---|---|---|
| Swallow（vLLM） | 0.07 / 0.33 秒 | 0.99 / 3.51 秒 | 52 | 49 |
| Gemma 12B | 0.12 / 0.47 秒 | 0.85 / 1.72 秒 | 33 | 25 |
| Gemma 12B（FP8、本番の枠） | 0.07 / 0.24 秒 | 0.56 / 2.53 秒 | 51 | 25 |
| Gemma E4B | 0.05 / 0.17 秒 | 0.39 / 0.52 秒 | 78 | 23 |
| Qwen 27B（FP8） | 0.59 / 1.37 秒 | 1.43 / 2.96 秒 | 28 | 24 |
| Qwen Flash（Strata） | 0.90 / 1.34 秒 | 1.01 / 1.64 秒 | 141 | 30 |
| Nemotron JP | 0.29 / 0.62 秒 | 2.34 / 5.90 秒 | 48 | 100 |
| LFM3（transformers） | 0.05 / 1.06 秒 | 1.68 / 3.37 秒 | 47 | 79 |

- Strata は生成こそ最速（141 tok/s、MTP の下書きつき）だが、最初の文字までは最も遅い（0.90 秒）。エンジンのログでは、質問が変わるたびに再利用 0 トークン（`0 reused`）で、共通のシステムプロンプトも含めてプロンプト全体（600〜2,200 トークン）を約 1,100〜1,250 tok/s で読み直していた。同じプロンプトを続けて送ったときだけ再利用され、0.28 秒になった。答えが 1 文で終わるアバターの用途では、完了までの時間は Swallow と同じくらい。
- Gemma 12B は生成が遅い（33 tok/s）が、答えが短いので完了までは Swallow より速い。FP8 では重みの読み出しが半分になり、51 tok/s まで上がった。FP8 の行だけは本番と同じ枠（0.55、16,384）で測った。最大 2.53 秒は、入試の質問に 116 トークンで答えた 1 件。
- 測定中、GPU0 では別の作業（ComfyUI）が動いていた。GPU は分けたが、CPU と PCIe は共有している。

## 本番に入れる場合の作業

本番の LLM は GPU0 で vLLM 0.9.2 のコンテナ `ace-vllm-llama31-swallow-8b` として動き、`--gpu-memory-utilization 0.55`（約 26 GB）、`--max-model-len 16384`。GPU0 には A2F・Riva・TTS も載り、GPU1 は Unreal renderer が使う。

| 候補 | 重みの大きさ | GPU0 の 26 GB 枠 | vLLM 0.9.2 | 備考 |
|---|---|---|---|---|
| Gemma 12B | bf16 で約 23 GB、FP8 で 12.9 GiB | bf16 は起動しない（KV 0.36 GiB、16,384 トークンに 1.82 GiB 必要）。FP8 は起動し、KV 10.1 GiB（91,331 トークン） | 非対応 | Apache 2.0 |
| Gemma E4B | 約 15 GB（音声・画像の部品込み） | 入る | 非対応 | Apache 2.0 |
| Qwen 27B | FP8 で約 29 GB | 入らない。4 bit 量子化か枠の拡大が要る（未検証） | 非対応 | Apache 2.0。`max_num_seqs` を下げる必要あり |
| Qwen Flash（Strata） | GGUF 84 GB | 入らない。48 GB の GPU 1 枚をほぼ使い切り、残りを CPU メモリに置く | 対象外（Strata 専用） | Qwen Community License 1.0。今の 2 枚構成では Tokkio と同居できない |

- どの候補も、本番の vLLM 0.9.2 では動かない。vLLM を新しくする必要がある（今回は 0.32.0 で確認）。ドライバーは 580.196.07 になったので、CUDA 13 版のイメージも使える。
- vLLM 0.32.0 では、システムの nvcc（12.4）で FlashInfer のサンプラーがビルドに失敗した。`VLLM_USE_FLASHINFER_SAMPLER=0` で回避した。コンテナなら同梱の CUDA を使うので起きない見込み（未検証）。
- Tokkio 側は `.env` のモデル名を変えるだけでは足りない。`infra/tokkio/tokkio_llm.py` の `nvidia_model_input_params` は、思考つき Nemotron 以外には空の dict を返し、温度も `max_tokens` も `enable_thinking` も渡さない（Swallow も同じ。10-17 監査の指摘 1）。Qwen3.8 は思考が既定でオンなので（`chat_template.jinja` 46 行目、未指定なら思考あり）、`chat_template_kwargs: {enable_thinking: false}` を渡さないと答えの前に長い思考を出す。Gemma 4 は既定でオフ（12B・E4B とも `chat_template.jinja` 186 行目 `default(false)`）なので、渡さなくても思考は出ない。ただし温度などの生成パラメータは、どのモデルでも渡す処理が要る。
- 2026-10-28 に Gemma 4 の準備を済ませた。`nvidia_model_input_params` は名前に `gemma-4` を含むモデルに比較と同じ設定（温度 0.1、top_p 0.95、top_k 64、最大 512 トークン、思考オフ）を送る。起動定義は `infra/llm/docker-compose.gemma4-12b.yml`（vLLM v0.32.0-cu129、FP8、port 8011、GPU0 の 0.55 枠）、切り替えと戻し方は `infra/tokkio/README.md` に書いた。本番の切り替えはまだしていない。本番の `NvidiaLLMRAGRouterService` は `bot.py` から `params` を受け取らないので、この設定がそのまま効く。RAG サーバー用の `max_tokens: 64` は LLM の要求には入らない（`test_rag_router_built_like_bot_py_sends_the_gemma4_sampling`）。
- 公式イメージ `vllm/vllm-openai:v0.32.0-cu129` は、そのままでは Gemma 4 を起動できなかった。torch は CUDA 12.9 用なのに torchcodec 0.17.0 だけが CUDA 13 用で、`libnvrtc.so.13` が無いと止まる（評価用 venv と同じ問題）。torchcodec を外した派生イメージ（`infra/llm/Dockerfile.vllm-cu129`）を compose からビルドする。GPU1・port 8018 で compose ファイルどおりに試験起動し、185 秒で起動、KV 91,331 トークン（venv と同じ）を確認した。Tokkio と同じ生成設定（`nvidia_model_input_params` の戻り値）で本番相当 11 問を送り、すべて `finish=stop`、2 問目以降の最初の文字まで 0.05〜0.13 秒。起動直後の 1 問目だけ 1.45 秒かかった。

## 未確認・限界

- 盲検の採点者は Claude（サブエージェント）で、人の採点ではない。抜き取った 9 件のうち 1 件（片柳研究所の所長の件）は、参考情報にある記載を「根拠なし」として厳しすぎた。ほかは妥当だった。
- ELYZA は長い応答が有利になりうる（Swallow の応答は他の約半分の長さ）。
- 本番相当は 63 問で、カテゴリごとには 2〜4 問しかない。カテゴリ別の差は傾向として読む。
- Swallow と LFM3 は transformers、他は vLLM・Strata で生成した。Swallow を両方で動かした差は小さかった。
- 本番の Tokkio を通した会話（音声認識の誤り、会話の続き、TTS の句切り）では試していない。
- Gemma 12B の FP8 は、自動指標と応答の長さ・遅延だけ測った。本番相当の盲検の採点はしていない。Qwen 27B の 4 bit 化は測っていない。
- 本番の枠での起動確認は GPU1 で行った（GPU0 と同じ 48 GB のカード）。本番の GPU0 は A2F・Riva・TTS と同居するので、空きメモリの状態は同じではない。
- Qwen Community License 1.0 の商用条件は原文を読んでいない。
- Strata は既定設定で測った。`--tune quick` は「標準設定が最速」と返したが、プロンプトの先頭を再利用する設定があるかは調べていない。あれば最初の文字までの 0.90 秒は縮む可能性がある。
