# Haas-ja-v1.0（日本語 ASR）を speech.cpp で評価

実施日: 2026-11-02 / 対象: `ace_kagawa`

**Haas は TTS ではなく音声認識（ASR）のモデル。** Third Intelligence の `third-intelligence/Haas-ja-v1.0`（616M、FastConformer＋TDT、予測ネットワークに Kimi Delta Attention、Apache-2.0、2026-10-30 更新）。speech.cpp 作者による GGUF（`sakasegawa/Haas-ja-v1.0-GGUF`）が出ており、speech.cpp 0.9.4 以降で動く。そこで、本番の ASR（speech.cpp の Qwen3-ASR-1.7B＋prompt「青嵐、SEIRAN」）と同じ方法で比べた。

本番の unit・Tokkio・`.env` は変更していない。Tokkio と本番 ASR は計測時点で停止していた。GPU 0 は ComfyUI（PID 3481673、42 GB、こちらの起動ではない）が使っていたので、すべて空いている GPU 1 で測った。

## 結論

**Haas は「サーバー側 VAD＋登録語「青嵐」」で使えば、現行と同等以上の精度で、速く、軽い。ただし切り替えにはコード変更が要り、体感の速さはほぼ変わらない。**

| 項目 | 現行 Qwen3-ASR-1.7B＋prompt | Haas Q8_0＋VAD＋「青嵐」 |
| --- | --- | --- |
| 来訪者の質問 160 本（合成音声）CER | 3.19% | 2.72%（差 −0.48 pt、95%CI −1.41〜+0.45、差は有意でない） |
| 「青嵐」を正しく書いた回数 | 16/16 | 14/16 |
| FLEURS 650 本（実音声）CER | 5.33%（prompt なし） | 4.47%（差 −0.85 pt、95%CI −1.32〜−0.38、有意に良い） |
| 雑音・無音 39 本で LLM に渡る文字 | 0 本（prompt だけの文字起こしを捨てる本番の処理で） | 0 本（サーバーの VAD が空文字にする） |
| 認識時間 p50（来訪者） | 75 ms | 46 ms |
| VRAM | 2.3 GB | 0.7 GB |

- **話し終わりから LLM までは約 0.03 秒しか縮まない。** VAD 停止 0.6 秒＋集約 0.05 秒が大半で、認識時間の差（約 30 ms）は小さい。
- **語彙を多く登録しても壊れにくいのが Haas の強み。** 評価文の固有名詞 18 語を登録すると CER 1.21%、固有名詞 98.8%（Qwen は 18 語の prompt で一覧を書き出し、CER 10.8% に悪化した。10/17 の報告）。ただし、遠くの話し声（−40 dB）12 本中 2 本に登録語（「香川豊」「医療保健学部」）を書き込んだ。登録語は数語にとどめる。
- **そのままでは置き換えられない。** 次の 3 つが要る（4 節）。
  1. Tokkio の `SpeechCppSTTService` から `chunking_strategy=auto` を送る（サーバー既定にする起動オプションがない）。送らないと、雑音 39 本中 21 本に「すぅ」「ぐぅ」「グル」などを書き、本番の雑音フィルター（prompt と同じ文字起こしを捨てる）は効かない。
  2. 英語モードの扱い。Haas は日本語専用で、`language=en` は 400 になる。今の英語モードは同じサーバーに `en` を送っている。
  3. speech.cpp を 0.8.10 から 0.10.1 に上げる（Haas は 0.9.4 以上、`keywords` の名前は 0.10.0 以上）。
- **実マイク・Tokkio 稼働中（GPU・CPU の競合あり）では測っていない。** 来訪者セットは合成音声、FLEURS は読み上げ。

## 1. 来訪者の質問（合成音声 160 本、文頭に 0.8 秒の無音）

10/17 と同じ音声（40 文 × 4 話者、Irodori で合成、固有名詞はかなで入力して漢字の参照文で採点）。文頭 0.8 秒の無音は、pipecat が VAD 検出前の最大 1 秒を一緒に送るのに合わせた条件。現行も同じ GPU・同じ speech.cpp 0.10.1 で測り直した（CER 3.19%・75 ms で、10/17 の 3.27%・76 ms を再現）。

| 構成 | CER | かな CER | 青嵐 漢字 / かな / 誤り | 固有名詞 | 現行との CER 差 [95%CI] | 認識 p50 / p90 | VRAM |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 現行 Qwen3-ASR-1.7B＋prompt | 3.19% | 2.80% | 16 / 0 / 0 | 82.9% | — | 75 / 102 ms | 2.3 GB |
| Haas Q8_0 | 3.96% | 3.83% | 0 / 2 / 14 | 77.4% | +0.77 [−0.19, +1.78] | 23 / 47 ms | 0.7 GB |
| Haas Q8_0＋「青嵐」 | 3.08% | 2.58% | 15 / 0 / 1 | 86.6% | −0.11 [−0.90, +0.72] | 35 / 56 ms | 0.7 GB |
| Haas Q8_0＋VAD | 3.30% | 2.66% | 2 / 0 / 14 | 79.3% | +0.11 [−0.97, +1.21] | 42 / 64 ms | 0.7 GB |
| **Haas Q8_0＋VAD＋「青嵐」** | **2.72%** | **1.86%** | 14 / 0 / 2 | 86.6% | −0.48 [−1.41, +0.45] | 46 / 69 ms | 0.7 GB |
| Haas Q8_0＋VAD＋18 語 | 1.21% | 1.28% | 14 / 0 / 2 | 98.8% | −1.98 [−3.03, −1.00] | 49 / 72 ms | 0.7 GB |
| Haas Q8_0＋18 語 | 1.36% | 1.55% | 15 / 0 / 1 | 99.4% | −1.84 [−2.83, −0.90] | 38 / 59 ms | 0.7 GB |
| Haas F16 | 3.85% | 3.80% | 0 / 2 / 14 | 77.4% | +0.66 [−0.29, +1.65] | 36 / 49 ms | 1.3 GB |
| parakeet-tdt_ctc-0.6b-ja（参考） | 4.92% | 4.97% | 0 / 1 / 15 | 75.0% | +1.73 [+0.73, +2.82] | 24 / 44 ms | 0.7 GB |

- CER・かな CER・青嵐・固有名詞の定義は 10/17 の報告と同じ（NFKC・句読点除去、かなは pyopenjtalk の読み）。95%CI は発話単位の対応ブートストラップ（10,000 回）。
- 「VAD」は、サーバーに silero-vad（v6.2）も載せ、要求に `chunking_strategy=auto` を付けた構成。発話のある区間だけを認識する。
- 「青嵐」は `keywords[]=青嵐=せいらん,セイラン`。18 語は 10/17 の P2 prompt と同じ語にかなの読みを付けたもの（`scripts/run_haas.sh` の `K2`）。**評価文の固有名詞そのものなので甘い条件。**
- 登録語なしの Haas の誤りは大半が固有名詞（「青嵐」→「青藍」「生卵」「セーラー」、「NEDO」→「ネド」、「医療保健」→「医療保険」、「炭化ケイ素」→「単かけ磯」、「基複合」→「起伏合」）。18 語を登録して悪化した発話は 0 本だった。
- 「SEIRAN」と言った文は登録語で「青嵐」と書く。採点では同じ語とみなした。
- Haas は文頭の無音で少し悪くなる（Q8_0 で無音なし 3.23% → 0.8 秒 3.96%、差 +0.73 [−0.07, +1.56]）。VAD を付けると 3.30% で、無音なしとほぼ同じになる。
- Q8_0 と F16 は同じ精度（差 −0.11 [−0.34, 0.00]）。Q8_0 は既定でデコードを CPU で行い、空いた GPU ではこれが最も速かった（GPU でデコードすると 32 ms）。

## 2. FLEURS ja_jp test（実音声 650 本）

| 構成 | CER | かな CER | Qwen との CER 差 [95%CI] | 認識 p50 / p90 | RTFx | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-1.7B（prompt なし） | 5.33% | 3.26% | — | 162 / 232 ms | 79 | 2.8 GB |
| Haas Q8_0 | 4.44% | 2.70% | −0.89 [−1.34, −0.44] | 58 / 69 ms | 231 | 0.8 GB |
| Haas F16 | 4.42% | 2.68% | −0.91 [−1.36, −0.46] | 72 / 96 ms | 181 | 1.4 GB |
| Haas Q8_0＋VAD | 4.47% | 2.87% | −0.85 [−1.32, −0.38] | 88 / 145 ms | 133 | 0.7 GB |
| parakeet-ja（参考） | 6.06% | 4.09% | +0.73 [−0.06, +1.59] | 55 / 68 ms | 238 | 0.8 GB |

- 95%CI は 2,000 回のブートストラップ。Haas の公式カードによれば FLEURS は学習に使っていない。
- Haas Q8_0 の 4.44% は GGUF カードの値（4.39%、正規化が違う）とほぼ同じ。
- FLEURS は平均 13 秒の読み上げで、来訪者の短い問いかけとは性質が違う。Qwen は 10/17 と同じく prompt なしで測った。

## 3. 雑音・無音

10/17 の 6 本に加え、非音声 39 本（白色・ピンク・褐色雑音 −45〜−25 dBFS × 0.6〜2.4 秒、息や触れた音のような短い雑音 6 本、50 Hz のハム 6 本）と、遠くの話し声 12 本（FLEURS 3 本を重ねて −40 / −30 dBFS）を作った（`scripts/make_noise51.py`、乱数固定）。

| 構成 | 非音声 39 本で文字が出た | うち登録語 | 話し声 12 本で登録語を書いた |
| --- | ---: | ---: | ---: |
| 現行 Qwen＋prompt | 39（すべて prompt そのもの。本番の処理で全部捨てる） | — | 0（4 本は prompt そのもので捨てる） |
| Qwen（prompt なし） | 39（「はい。」34、「ふ。」5） | — | — |
| Haas | 21（「すぅ」「ぐぅ」「グル」「n」「ハハハ」など） | 0 | 0 |
| Haas＋「青嵐」 | 21 | 1（褐色雑音に「青嵐。」） | 0 |
| Haas＋18 語 | 21 | 0 | 2 |
| **Haas＋VAD（「青嵐」あり・なし）** | **0** | 0 | 0 |
| Haas＋VAD＋18 語 | 0 | 0 | 2（「香川豊」「医療保健学部は原発内で…」） |

- 遠くの話し声は実際の音声なので、どの構成もその文を書く（VAD でも 12 本中 11 本）。問題は登録語が混ざること。18 語の登録では、元の文（「駆けつけの原因」「原子力省によれば」）が登録語に置き換わった。Haas のガード（登録語まるごとを含む変更だけ残す）はこれを防がない。
- 雑音のとき Qwen は prompt を書くので本番の処理で捨てられるが、Haas は prompt を受け付けない（`unsupported_parameter` の 400）。代わりに VAD で空文字にする。
- 1.0 秒の無音を curl で送ると Haas は「こんにちは。」と書いた（1.5 秒にすると空、VAD ありなら空）。VAD なしで使わない理由の一つ。

## 4. Tokkio で使うなら

> **2026-11-02 追記: 切り替えた。** 下の表のとおり変更し、英語モードは別 unit `ace-speechcpp-asr-en.service`（Qwen3-ASR-1.7B、port 8025）にした。`chunking_strategy` は `extra_body` で送る（controller の openai SDK の版が分からないため）。テスト用ポートで確かめたこと: 雑音は `chunking_strategy` ありで空文字、なしだと「うん。」になる。VAD モデルのないサーバーに送ると毎回 400 になるので、起動スクリプトは VAD モデルのファイルがなければ起動しない。本番の unit は導入して起動・停止を確かめた。Tokkio での実マイク確認はまだ。手順は `infra/asr/README.md` と `infra/tokkio/README.md` の「音声認識」。


| 場所 | 変更 |
| --- | --- |
| `infra/asr/ace-speechcpp-asr.service` | `SPEECHCPP_ASR_BIN` を v0.10.1、`SPEECHCPP_ASR_MODEL` を Haas Q8_0 のパスに。silero-vad の GGUF も載せる（`run_speechcpp_asr.sh` はモデルを 1 つしか渡さないので引数を足す）。登録語はサーバー既定 `--keywords $'青嵐=せいらん,セイラン'` で渡せる（要求に `keywords` がなくても効くことを確認済み） |
| `infra/tokkio/tokkio_speechcpp_stt.py` | 要求に `chunking_strategy="auto"` を付ける（`_transcribe` で `extra_body` などに足す）。prompt は送らない |
| `customize_tokkio_japanese.py` | `SpeechCppSTTService.model: "Haas-ja-v1.0"`、`prompt: ""`。英語モード（`language: "en"`）は Haas で 400 になるので、英語用に別のサーバー（Qwen3-ASR か parakeet-tdt-0.6b-v3、別ポート）を立てるか、英語モードだけ Qwen を残す。`speech serve` は認識モデルを 1 つしか載せられない |
| `infra/tests/test_tokkio_speechcpp_stt.py` | 上の変更に合わせる |

見積もり: 半日（コード・テスト 2〜3 時間、実マイク確認 1 時間）。英語モードを別サーバーにするなら＋1〜2 時間。

## 5. 速度の補足

- 認識時間は GPU 1 が空いた状態の値。本番は GPU 0 を vLLM・TTS・A2F と共有し、CPU も混んでいる（Unreal・triton）。Haas Q8_0 はデコードを CPU（4 スレッド）で行うので、CPU が混むと遅れうる。その場合は `--decode-on-cpu=false`（空いた状態で 32 ms）を試す。
- 長さ別（1〜28 秒、新しいサーバーで昇順・降順）では Haas 0.02〜0.12 秒、Qwen 0.03〜0.44 秒。Haas は長い音声ほど差が大きい。
- **計測の序盤（10:13〜10:18）に、1 要求だけ 2.1〜4.6 秒かかったことが 5 回あった**（Haas 3 回・Qwen 2 回、いずれも集計から除いたウォームアップ中）。その後、長さ順・要求順を変えて 3 回試したが再現しなかった。NVIDIA のシェーダーキャッシュ（`~/.cache/nvidia/GLCache/…/2529453c2b75eef7.bin`）が 10:12 に作られ 10:20 まで増えていたので、**speech.cpp を新しい版にした直後のシェーダーのコンパイルと推定する（未証明）**。本番の unit も kyano で動くので、同じキャッシュを使う。版を上げたら、会話の前に長さの違う音声を数本送っておくとよい。

## 6. 条件と限界

| 項目 | 条件 |
| --- | --- |
| GPU | NVIDIA RTX 6000 Ada、nvidia-smi の GPU 1（UUID `GPU-ed07bcd6-…`、speech.cpp の `Vulkan0`）。各サーバーの配置は nvidia-smi で確認 |
| speech.cpp | 公式 Linux x64 Vulkan リリース 0.10.1（C API 4.1）、SHA256 照合済み。`--threads 4` |
| Haas | `sakasegawa/Haas-ja-v1.0-GGUF` revision `6b965d5`（元モデル `660728d`）、Q8_0・F16 とも README の SHA256 と一致 |
| 比較対象 | 本番と同じ Qwen3-ASR-1.7B Q8_0・parakeet-ja Q8_0 のファイル |
| 要求 | openai SDK の `audio.transcriptions.create`（pipecat と同じ）。話し終わりの後に 0.5 秒の無音。各構成の前に 3 本ウォームアップ |

- 来訪者セットは合成音声。実マイク（反響・遠い話者・言いよどみ）は未確認。
- **サーバー側 VAD は雑音を落とす代わりに、小声・遠い話者の発話を丸ごと落とす（空文字になり、アバターが黙る）おそれがある。** 合成音声は普通の音量なので、この取りこぼしは測れていない。実マイクで確かめる。
- Tokkio 稼働中の負荷（GPU 0 の共有、CPU の混雑）での速度は未確認。
- 同時に複数の会話は測っていない。
- 18 語の登録は評価文から作った甘い条件。実際の語彙を決めたら別の文で確かめる。

## 7. 再現

すべて `/data/ACE/speech-cpp/20261102-haas/` にある。

| パス | 中身 |
| --- | --- |
| `scripts/run_haas.sh` | 来訪者・FLEURS・6 本の雑音（Haas 各構成と比較対象） |
| `scripts/run_haas2.sh`、`run_haas3.sh`、`run_haas4.sh` | 雑音 51 本、初回要求と英語、VAD 付きの構成 |
| `scripts/lenprobe.sh`、`stallprobe.sh` | 長さ別の時間、遅い要求の再現の試み |
| `scripts/bench_asr.py`、`score_asr.py`、`summarize.py` | 10/17 のものに `--keywords`・`--chunking` を追加 |
| `scripts/summarize_haas.py` | 表の数値と 95%CI |
| `scripts/make_noise51.py`、`noise51/` | 雑音 51 本 |
| `results/` | 発話ごとの結果、サーバーのログ、VRAM |
| `upstream-docs/` | speech.cpp 0.10.1 の `server.md`・`fastconformer.md`・`cli.md`、SHA256SUMS |

評価用 venv は `/data/ACE/venvs/asr-eval`（openai・soundfile・numpy・rapidfuzz・pyopenjtalk）。評価音声は `/data/ACE/speech-cpp/20261017-asr/` のものを使った。

## 一次資料

- [Haas-ja-v1.0 モデルカード](https://huggingface.co/third-intelligence/Haas-ja-v1.0)
- [Haas-ja-v1.0 GGUF for speech.cpp](https://huggingface.co/sakasegawa/Haas-ja-v1.0-GGUF)
- [speech.cpp v0.10.1 リリース](https://github.com/nyosegawa/speech.cpp/releases/tag/v0.10.1)、[server.md](https://github.com/nyosegawa/speech.cpp/blob/v0.10.1/docs/server.md)、[fastconformer.md](https://github.com/nyosegawa/speech.cpp/blob/v0.10.1/docs/models/fastconformer.md)
- 前回の比較: `docs/audits/speech-cpp-asr-2026-10-17.md`
