# アバターの顔と手が動かない理由（2026-11-18 調査）

対象: Tokkio 本番と local_avatar の両方で使っている CC5 パッケージ `cc5_tokkio` 2026-07-15。
稼働中の Tokkio には書き込んでいない（ログ・configmap・ファイルの読み取りだけ）。

## 結論

- 口は A2F-3D の blendshape で動く。それ以外（まばたき以外の表情、頭、体、手）は、今の構成では動かす部品がない。
- 体と手: パッケージにアニメーションのクリップも retarget の設定もない。UE 側の ACE ノードは blendshape（curve）を当てるものだけ。
  Animation Graph が関節を送っても、CC5 の骨には当たらないと読んでいる（下の「未確認」を参照）。
- 顔: Audio2Emotion（a2e）は `enabled: false`。これは NVIDIA の Tokkio 5.0 の values.yaml の既定で、こちらで切ったものではない。
  A2F の `EyeLook*` は active_poses が 0 で、視線は動かさない設定。黙っている間は A2F が動かないので、顔は止まる。
- 直すには UE プロジェクトの Animation Blueprint に待機モーション・まばたき・頭の動きを足して、パッケージを作り直すのが本筋。
  ビルド元は実行ファイルに残ったパス `C:/Users/kyano/Documents/workspace/cc5_tokkio`。

## 証拠（原文）

### パッケージのアセット（`cc5_tokkio-Linux.utoc`、724 件）

アニメーション関係で見つかったのは次だけ（`A_`・`AS_`・`AM_`・`BS_`・`RTG_`・`IK_` で始まるものは 0 件）。

```
CC5.uasset CC5_ABP_RL_Rig_Logic.uasset CC5_PA_ExtendFacialProfile.uasset CC5_PhysicsAsset.uasset
CC5_Skeleton.uasset CR_CC_HeadIK.uasset CR_CtrlValueLimit.uasset CR_FaceExpRig.uasset L_CC5_Tokkio.umap
```

目次の `Animation` ディレクトリの中身は ControlRig の形状（`ControlRig_Sphere_3mm.ubulk` など）で、動きのクリップではない。

### 実行ファイル `cc5_tokkio/Binaries/Linux/cc5_tokkio` の ACE 関連の名前

```
AnimNode_ApplyACEAnimation
ACECurveSource
UACEAudioCurveSourceComponent
C:/Users/kyano/Documents/workspace/cc5_tokkio/Plugins/NV_ACE_Reference/Source/ACERuntime/Private/ACEAudioCurveSourceComponent.cpp
ACC5TokkioAvatar
CC5CharacterMesh
EyeBlinkLeft EyeBlinkRight Eye_Blink_L Eye_Blink_R Mouth_Roll_In_Upper
```

関節を受けて骨を動かすノードの名前（`Joint`・`Bone`・`Body` を含む ACE の名前）は見つからなかった。

### Animation Graph（configmap `ia-animation-graph-microservice-params`）

```
IAAGMS_SCENE_FILENAME: Avatar_Scene.usda
REMOTE_RESOURCE_PATH: nvidia/ace/default-avatar-scene:1.1.5
```

NVIDIA の既定アバター用の場面で、CC5 用ではない。

### A2F-3D（configmap `a2f-configs-cm` の `stylization_config.yaml`）

```
a2e:
  enabled: false
...
EyeBlinkLeft: 1
EyeBlinkRight: 1
EyeLookDownLeft: 0
...
EyeLookUpRight: 0
```

`infra/tokkio/workspace/NVIDIA-ACE/workflows/tokkio/5.0.0-ga/llm-rag/tokkio-1stream-with-ui/values.yaml:52-53` も `a2e: enabled: false`。

### renderer のログ

```
[2026.11.18-07.26.37:556][164]LogACERuntime: [ACE SID 2 callback] received 83804 animation samples, 44863901 audio samples for clip on CC5TokkioAvatar ...
```

83,804 フレームは 30 fps で約 2,790 秒で、セッションの間ずっとフレームが届いている。止まって見えるのは、届いたフレームに動きがないか、UE が使わないため。

### local_avatar の中継（`infra/local_avatar/mini_animgraph.py:222-225`）

```
# blend shapes, audio, translations, rotations and scales: it
# crashes when a list is empty and drops frames whose joint values
# are empty. UE only uses blend shapes and audio from this stream,
# so one identity transform for a dummy joint satisfies it.
```

## 未確認

- CC5 アクターが関節のデータを捨てていること。実行ファイルの名前とアセット一覧からの推測で、関節を送って試してはいない。
- （同日に解決）A2F はまばたきを出さない。下の追記の 2 節を参照。
- `CC5_ABP_RL_Rig_Logic` が `EyeLook*` の curve を目の骨に変換するか。

## 直し方の候補

| 方法 | 動くもの | 必要な作業 | 両経路に効くか |
|---|---|---|---|
| UE の Animation Blueprint に待機モーション・まばたき・頭の揺れを足す | 体・手・頭・まばたき | Windows の UE でプロジェクトを編集し、Linux パッケージを作り直す。Tokkio の overrides のハッシュを更新して再配備 | 効く |
| UE に「curve → 頭の骨」の対応を足す | 頭のうなずき・傾き | 上と同じビルドに含める。以後は Linux 側から curve で頭を動かせる | 効く |
| 中継で blendshape にまばたき・眉・視線を足す | まばたき・眉・視線 | local は `mini_animgraph.py` を直す。Tokkio は UE と Animation Graph の間に中継を挟む必要があり、SDR の経路で動くかは未検証 | local は確実、Tokkio は未検証 |
| a2e を有効にする | 話している間の眉・目・口角 | a2f の pod の設定変更。今の GPU0 の空きでは起動しなかった（下の追記） | 黙っている間・頭・手には効かない |

## 追記: Audio2Emotion（a2e）を有効にした場合（同日に実測）

試験は Tokkio と別に、ローカル用の A2F-3D 1.3.16 を GPU0 で一時起動して行った（コンテナ `kagawa-a2e-test`、終了済み）。
設定は稼働中の Tokkio の `a2f-configs-cm` の `stylization_config.yaml` をそのまま使い、`a2e.enabled` だけを変えた。
音声は本番の TTS（:8021、voice `kagawa`）で作った 6 文、計 1,130 フレーム。
データとスクリプトは `/data/ACE/ace_kagawa/local_avatar/experiments/20261118_a2e/`。

### 1. a2e を有効にすると、今の GPU0 では A2F が起動しない

a2e 無効では A2F の処理部分は GPU を 2,164 MiB 使って起動した。有効にすると、初期化の途中で次のエラーが出て処理部分が止まった。

```
[A2X SDK] [ERROR] CUDA error: out of memory
[A2F SDK] [ERROR] Cannot Initialize from Json file: /tmp/tmp_a2f_new_config_file_merged.json
ERROR from element a2f-library: Runtime error: Couldn't create A2F IProcessor from config
```

起動中の GPU0 の使用量は 42,971 MiB から 48,494 MiB（総量 49,140 MiB）まで増えてから失敗した。a2e 有効の A2F は約 5.5 GB では足りない。
a2e のエンジンは `a2e_v1.0_rtx6000_fp32_bs8_v4`（383 MB）で、`trt_model_generation.a2e.maximum_shape: 128`。
Tokkio の A2F の pod を a2e 有効で作り直した場合は分からない。今の pod が使う 2,164 MiB が空くので空きは約 8.3 GB になり、必要量（5.5 GB より多い）と際どい。
Tokkio の pod は自分で TRT エンジンを作るので、試験で使ったダウンロード版のエンジンと使用量が違う可能性もある。新旧の pod が同時に動く入れ替え方だと、その間はさらに足りない。

### 2. 感情の値が表情をどう変えるか

a2e は声から感情を推定し、その値を A2F に渡す。a2e は動かせなかったので、同じ値を要求の `emotions` で手で渡した。
a2e 無効のときは、渡した時刻のフレームにだけ効き、次のフレームで元に戻る（0.1 秒ごとに渡すと 3 フレームに 1 回だけ変わる）。
`preferred_emotion_strength` を 1 にしても同じだった。ストリームの header の `emotion_post_processing_params` で変わるかは試していない。
次の表は、感情が効いたフレーム 377 個での平均。

| blendshape | なし | joy 0.5 | joy 1.0 | sadness 0.5 | amazement 0.5（a2e は出さない） |
|---|---|---|---|---|---|
| BrowOuterUp（眉の外側を上げる） | 0.05 | 0.44 | 0.83 | 0.00 | 0.45 |
| BrowInnerUp（眉の内側を上げる） | 0.01 | 0.15 | 0.29 | 0.28 | 0.05 |
| EyeWide（目を見開く） | 0.04 | 0.11 | 0.18 | 0.09 | 0.12 |
| MouthSmile（口角を上げる） | 0.00 | 0.09 | 0.22 | 0.00 | 0.05 |
| MouthStretch（口を横に引く） | 0.06 | 0.17 | 0.30 | 0.07 | 0.14 |
| MouthUpperUp（上唇を上げる） | 0.04 | 0.11 | 0.17 | 0.09 | 0.09 |

- 感情なしのとき、眉・目・頬の値はほぼ 0 のまま。画面で「顔が動かない」と見える主な原因と考えている（画面では確かめていない）。
- `EyeBlinkLeft/Right` は、どの条件でも全フレームで 0.000。設定は active_poses 1・weight_multipliers 1・weight_offsets 0・eyelid_offset 0 で、設定で消しているのではない。A2F（james_v2.3）がまばたきを出さない。
- a2e が出せる感情は anger・disgust・fear・joy・sadness の 5 つ（コンテナ内の `/tmp/tmp_a2e_new_config_file.json` の `emotion_correspondence`。neutral は感情なし）。amazement の列は参考。
- anger 1.0 では先頭のフレームで BrowDown（眉を寄せて下げる）が 0.00 → 0.84。a2e が穏やかな声を anger に分類すると、怒った顔になる。
- 感情は口の形も変える（口角・横の引き・上唇）。10-20 に調整した口の開き方に重なる。

### 未確認

- a2e が香川先生の日本語の声（😊 付きの TTS）をどの感情に分類するか。a2e を動かせなかったので測っていない。
- a2e 有効の A2F が実際に何 GB 使うか（5.5 GB より多いことだけ確認）。

## 追記 2: GPU1 で a2e を実際に動かした結果（同日）

ユーザーの了承を得て、GPU1 で a2e 有効の A2F を数分だけ動かした（aituber の :8088 は前後とも 200）。音声・設定は上と同じ。

### メモリ

a2e 有効の A2F の処理部分は 5,568 MiB を使った（GPU1 の使用量は 18,487 → 最大 24,060 MiB）。無効なら 2,164 MiB。
Tokkio の A2F の Deployment は `RollingUpdate`（maxSurge 25%、maxUnavailable 25%、replicas 1）なので、入れ替えの間は新旧の pod が並ぶ。
今の GPU0 の空きは約 6.2 GB で、新しい pod の 5.6 GB はぎりぎり入る計算だが、余裕は約 0.6 GB しかない。

### a2e の判定（`a2fSmoothedOutput` の文ごとの平均 / 最大）

| 文 | joy | sadness | disgust | anger |
|---|---|---|---|---|
| ありがとうございます。そう言っていただけると、本当にうれしいです。 | 0.12 / 0.79 | 0.34 / 0.81 | 0.22 / 0.46 | 0.12 / 0.61 |
| こんにちは、香川です。今日は来てくださって、ありがとうございます。 | 0.02 / 0.06 | 0.04 / 0.35 | **0.72 / 0.97** | 0.13 / 0.50 |
| セラミックス基複合材料は、軽くて熱に強いので、… | 0.64 / 0.98 | 0.10 / 0.44 | 0.13 / 0.55 | 0.04 / 0.37 |
| 学生のころは、研究室に泊まり込んで… | 0.26 / 0.97 | 0.04 / 0.30 | **0.51 / 0.95** | 0.14 / 0.66 |
| 申し訳ありません。その件については、… | 0.39 / 0.96 | 0.41 / 0.80 | 0.04 / 0.11 | 0.02 / 0.06 |
| えっ、そうなんですか。それは驚きました。 | 0.27 / 0.63 | 0.14 / 0.35 | 0.05 / 0.15 | 0.22 / 0.66 |

挨拶と思い出話が「嫌悪」、お礼が「悲しみ」寄りになった。香川先生の日本語の声では、a2e の判定は文の内容と合わない。

### 条件ごとの表情（全 1,130 フレームの平均）

| 条件 | 眉の外側を上げる | 眉の内側を上げる | 眉を下げる | 目を見開く | 口角を上げる | 鼻にしわ |
|---|---|---|---|---|---|---|
| a2e 無効（いま） | 0.052 | 0.013 | 0.001 | 0.044 | 0.001 | 0.006 |
| a2e 有効（そのまま） | 0.196 | 0.142 | 0.187 | 0.092 | 0.092 | 0.055 |
| a2e 有効＋初期感情 joy 0.3・優先 1.0 | 0.285 | 0.090 | 0.001 | 0.086 | 0.038 | 0.006 |
| a2e 有効＋初期感情 joy 0.3・優先 0.7 | 0.231 | 0.075 | 0.005 | 0.086 | 0.050 | 0.015 |

- `preferred_emotion_strength` が混ぜる相手は、設定の `default_beginning_emotions` だった。要求で渡した `emotions` は、a2e 有効のときは使われなかった（joy 0.4 を渡しても出力の感情は全部 0）。
- 優先 1.0 にすると、a2e の判定を使わず joy 0.3 で固定になる（全 6 文で joy = 0.30、ほかは 0）。眉を下げる・鼻のしわ（嫌悪・怒りの顔）は出ない。
- 優先 0.7 にすると、joy 0.3 を土台に a2e の判定が 3 割混ざる。挨拶の disgust は 0.72 → 0.22 に下がるが、残る。
- 優先 1.0 は「感情の推定」ではなく「話している間だけ、いつも穏やかに笑む」設定になる。a2e は値をなめらかにするためだけに動き、メモリは a2e 有効と同じだけ使う。

### 未確認

- 画面での見え方。眉の外側の平均 0.285 は、話している間ずっと眉が上がって見える可能性がある。数値だけで判断していない。
- joy の値（0.3）を変えたとき。0.3 しか試していない。

## 追記 3: a2e の判定が外れる理由（同日）

### 一次情報: NGC の Audio2Emotion モデルカード（2026-11-18 取得、原文）

出典: https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/audio2emotion_model （取得は引用の長さに上限があり、125 字を超える部分は短く切った）

- 学習データ: "Multiple datasets, including RAVDESS, CREMA-D, JL, EMO-DB, Emozionalmente, TTS GPT 4o (internal), Lindy & Rodney (internal)"
- "Quantity: 30029 samples"
- 出力: "The model can predict six emotions from speech: Anger, disgust, fear, joy, neutral, and sadness."
- v1.0: "Production version based on HuggingFace backbone `facebook/wav2vec2-base` with 80% accuracy."
- v2.0: "Fine-tuned version from `facebook/wav2vec2-large-lv60`. Higher accuracy than "base" model."
- v3.0: "Same starting checkpoint as v2.0, different training (trained on small chunks), different inference style." / "More suitable for Audio2Face purposes."
- 制限: "The model is not intended for standalone emotion recognition applications or general-purpose audio analysis."
- Bias 小カード（https://catalog.ngc.nvidia.com/orgs/nim/nvidia/models/audio2emotion_model/-/model-card）: 影響を受けうる群として "Age, Gender, Linguistic Background, Accent, Speech Patterns, and Cultural Context"
- 学習データの言語は書かれていない。build.nvidia.com の Audio2Face-3D のモデルカードは本文を取得できなかった。

補足（モデルカードの外の知識で、ここでは確かめていない）: RAVDESS・CREMA-D は英語、JL はニュージーランド英語、EMO-DB はドイツ語、Emozionalmente はイタリア語の、俳優が演じた感情音声のコーパス。
手元の A2F-3D 1.3.16 が使う a2e は `a2e_v1.0_rtx6000_fp32_bs8_v4` で、表の v1.0（wav2vec2-base）にあたる。

### 実験: 話者・言語・TTS を変えて a2e に通した

GPU1 で a2e 有効の A2F（本番と同じ設定）に 46 本を通した。数値は `a2fSmoothedOutput` の平均。データは `/data/ACE/ace_kagawa/local_avatar/experiments/20261118_a2e/why/`。

| 音声 | 本数 | joy | sadness | disgust | anger | fear | 合計 |
|---|---|---|---|---|---|---|---|
| 香川先生の TTS（本番、😊 あり） | 6 | 0.30 | 0.16 | 0.30 | 0.10 | 0.02 | 0.89 |
| 香川先生の TTS（😊 なし、sokutou 直呼び） | 6 | 0.25 | 0.10 | 0.35 | 0.11 | 0.04 | 0.86 |
| 香川先生本人の録音（日本語、8 秒×4） | 4 | 0.26 | 0.33 | 0.16 | 0.13 | 0.04 | 0.92 |
| 香川先生の声の英語クローン（Qwen3-TTS、8 秒×3） | 3 | 0.25 | 0.44 | 0.01 | 0.02 | 0.01 | 0.73 |
| 同（IndexTTS-2.5、8 秒×3） | 3 | 0.28 | 0.13 | 0.03 | 0.06 | 0.00 | 0.52 |
| VOICEVOX ふつうの読み（玄野武宏・青山龍星・もち子さん、各 2 文） | 6 | 0.11 | 0.00 | 0.04 | 0.25 | 0.47 | 0.87 |

VOICEVOX の感情スタイル（同じ平叙文 2 つ）で、平均が最大の感情がスタイルと一致したのは 9 組中 4 組（joy 1/3、sadness 1/3、anger 2/3）。
「ふつう」のスタイルでも、玄野武宏は fear 0.72、青山龍星は fear 0.69、もち子さんは anger 0.57 になった。

### 分かったこと

1. **落ち着いた日本語の声を「感情なし」と判定しない。** 日本語の音声は、ふつうの読みでも感情の合計が 0.86〜0.92 で、感情のこもった声と変わらない。同じ香川先生の声でも英語では 0.52〜0.73。neutral に振られた分が、日本語ではほとんどない。
2. **日本語の感情の向きも当たらない。** VOICEVOX の感情スタイルで 9 組中 4 組。話者ごとの癖が強く、もち子さんは喜びの声も anger、玄野武宏は平叙も悲しみも fear になった。
3. **😊 と TTS は主な原因ではない。** 😊 を外しても挨拶は disgust 0.69 のまま。本人の録音でも感情の合計は 0.92 で、sadness 0.33・disgust 0.16。
4. **disgust は日本語の香川先生の声で出る。** TTS で 0.30〜0.35、本人の録音で 0.16、英語クローンでは 0.01〜0.03。

まとめると、a2e v1.0 は日本語の話し方を学習しておらず、日本語の落ち着いた話し方を感情のある声と取り違えている、というのが実験と合う説明。
学習データの言語はモデルカードに書かれていないので、「日本語を学習していない」は名前の挙がったコーパスからの推定。

### 限界

- 1 群 2〜6 本と少なく、ばらつきの幅は出していない。
- 英語の比較は香川先生のクローン 2 種だけで、英語の感情スタイルのある音声（正解のある英語）では試していない。
- a2e の後処理（neutral を除いた後に値をどう扱うか）の式は確かめていない。

## 追記 4: A2F の別モデルと上半分の強さ（同日、GPU0）

a2e 無効のまま、本番の 6 文で比べた（全 1,130 フレーム、平均 / フレーム間の変化の平均）。

| 条件 | 眉 | 目 | 頬 | 口の開き | まばたきの最大 |
|---|---|---|---|---|---|
| james_v2.3（いま） | 0.024 / 0.0024 | 0.022 / 0.0014 | 0.003 / 0.0020 | 0.101 / 0.0247 | 0.000 |
| mark_v2.3（mark_topo1_v2.3） | 0.031 / 0.0015 | 0.000 / 0.0000 | 0.000 / 0.0000 | 0.049 / 0.0126 | 0.035 |
| claire_v2.3（claire_topo1_v2.3） | 0.011 / 0.0019 | 0.072 / 0.0071 | 0.000 / 0.0001 | 0.023 / 0.0130 | 0.014 |
| james、`upper_face_strength` 1.5 | 同上（james と完全に同じ値） | | | | 0.000 |
| james、`upper_face_strength` 2.0 | 同上 | | | | 0.000 |

- 別モデルにしても、上半分の顔はほとんど動かず、まばたきも出ない。口の開きは半分以下になる。
- `upper_face_strength` は、この設定（blendshape 出力）では値が 1 つも変わらなかった。効く範囲は確かめていない。
