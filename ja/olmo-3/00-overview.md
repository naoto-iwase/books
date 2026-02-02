# Olmo 3 全体像

> - **論文**: Olmo 3 (arXiv:2512.13961v1, December 2025)
> - **著者**: Allen Institute for AI (AI2) Olmo Team
> - **主要モデル**: Olmo 3 Base/Think/Instruct (7B, 32B)

---

## 1. Olmoプロジェクトの概要

### なぜ「真のオープンソース」なのか

<details>
<summary><strong>Open-weight vs Fully-open の定義と主要モデルの分類</strong></summary>

LLMの「オープン」には段階があり、混同されやすい。

**オープンネスの4段階**

| レベル | 公開物 | 例 |
|-------|-------|-----|
| **Closed** | APIのみ | GPT-4, Claude |
| **Open-weight** | モデル重みのみ | Llama 3, Gemma 2 |
| **Open-weight + Data** | 重み + 学習データ | DeepSeek V3（部分的） |
| **Fully-open** | 重み + データ + コード + レシピ | Olmo 3, Pythia |

**主要モデルの分類**

| モデル | 重み | データ | コード | レシピ | 評価 |
|-------|------|-------|-------|-------|------|
| **Olmo 3** | ✓ | ✓ | ✓ | ✓ | Fully-open |
| **Llama 3** | ✓ | ✗ | 一部 | ✗ | Open-weight |
| **Qwen 2.5** | ✓ | ✗ | 一部 | ✗ | Open-weight |
| **DeepSeek V3** | ✓ | 一部 | ✓ | 一部 | 準Fully-open |
| **Gemma 2** | ✓ | ✗ | ✗ | 一部 | Open-weight |
| **Mistral** | ✓ | ✗ | ✗ | ✗ | Open-weight |

**なぜFully-openが重要か**

1. **再現性**: 論文の主張を独立に検証可能
2. **介入研究**: 任意の段階からカスタマイズ可能
3. **教育**: LLM学習パイプライン全体を学べる
4. **透明性**: バイアスやデータ汚染を検証可能

**Olmo 3が公開している具体物**

- **データ**: Dolma 3 Mix, Dolmino Mix, Dolci
- **ツール**: Duplodocus, datamap-rs, OLMES, decon
- **コード**: OLMo-core, Open Instruct
- **ログ**: 全ステージの学習ログ、中間チェックポイント

**参考**: [The Foundation Model Transparency Index](https://crfm.stanford.edu/fmti/)

</details>

Olmoは他のopen-weight（重みのみ公開）モデルと異なり、**Model Flow全体**を公開している：

| 公開物 | 説明 |
|--------|------|
| **モデル重み** | 最終チェックポイントだけでなく、各ステージの中間チェックポイントも |
| **学習データ** | Dolma 3 Mix（事前学習）、Dolma 3 Dolmino Mix（midtraining）、Dolci（事後学習） |
| **学習コード** | OLMo-core（事前学習）、Open Instruct（事後学習） |
| **データ処理ツール** | datamap-rs、duplodocus（重複除去）、dolma3（レシピ） |
| **評価コード** | OLMES、decon（デコンタミネーション） |
| **学習ログ** | 全ステージの学習ログ |

これにより、研究者は任意のステージから介入・カスタマイズが可能。

### 公開されているモデルバリエーション

| モデル名 | 説明 |
|----------|------|
| **Olmo 3 Base** | 事前学習済みベースモデル（7B, 32B） |
| **Olmo 3 Think** | 思考トレースを生成するreasoningモデル |
| **Olmo 3 Instruct** | 短く直接的な応答を生成するチャットモデル |
| **Olmo 3 RL-Zero** | ベースモデルから直接RLで学習したモデル |

---

## 2. モデルアーキテクチャ

### サイズバリエーション

| パラメータ | 7B | 32B |
|------------|-----|------|
| Hidden size | - | - |
| Layers | - | - |
| Context length（事前学習） | 8,192 | 8,192 |
| Context length（拡張後） | 65,536 | 65,536 |

### アーキテクチャの設計選択

Olmo 3はOLMo 2をベースとし、以下の改良を加えたdecoder-only Transformer：

| 設計要素 | Olmo 3の選択 |
|----------|--------------|
| **位置エンコーディング** | RoPE（Rotary Position Embedding） |
| **アテンション** | Sliding Window Attention（SWA）+ Full Attention の組み合わせ |
| **SWAパターン** | 4層中3層がSWA（window size 4096）、最後の層は常にFull Attention |
| **精度** | bfloat16 |
| **Tokenizer** | cl100k（OpenAI由来） |

### Sliding Window Attention（SWA）の導入理由

> 詳細: [Sliding Window Attention 補足](./01-sliding-window-attention.md)

- 長いシーケンスでの事前学習のスケーラビリティ向上
- 推論コストの削減
- 各トークンは前の4096トークンにのみattend

### 他モデルとの違い

| 比較項目 | Olmo 3 | Llama 3 |
|----------|--------|---------|
| オープン度 | 完全オープン（データ・コード含む） | open-weight（重みのみ） |
| SWA | あり（3/4層） | なし |
| 事前学習context長 | 8,192 | 8,192 |
| 長文脈拡張 | 50-100B tokens | 800B tokens |

---

## 3. 事前学習データ（Pretraining Data）

### Dolma 3 Mix の構成

全体で約**5.93T tokens**の事前学習データ：

| ソース | Pool (9T) | Mix (6T) | 割合 |
|--------|-----------|----------|------|
| Common Crawl | 8.14T | 4.51T | 76.1% |
| olmOCR science PDFs | 972B | 805B | 13.6% |
| Stack-Edu（コード） | 137B | 409B | 6.89% |
| arXiv | 21.4B | 50.8B | 0.86% |
| FineMath 3+ | 34.1B | 152B | 2.56% |
| Wikipedia & Wikibooks | 3.69B | 2.51B | 0.04% |

### データ品質管理・フィルタリング

#### 1. テキスト抽出
- Common Crawl: Resiliparseでセマンティックテキスト抽出
- PDF: olmOCRで線形化プレーンテキストに変換

#### 2. ヒューリスティックフィルタリング
- URLフィルタリング（スパム・アダルトコンテンツ除去）
- 長さフィルタリング（短すぎ/長すぎ除去）
- 記号過多・アルファベット不足除去
- 内部重複除去
- fastText言語分類（英語のみ保持）
- Madlad400の文レベルヒューリスティック

#### 3. 重複除去（Deduplication）

> 詳細: [重複除去 (Deduplication) 補足](./02-deduplication.md)

3段階のアプローチで75%のドキュメントを削減：

| ステップ | 手法 | 削減率 |
|----------|------|--------|
| 1. Exact deduplication | ドキュメントハッシュ | 67% |
| 2. Fuzzy deduplication | MinHash | 23% |
| 3. Substring deduplication | Fuzzy suffix array | 14%（バイト） |

**Duplodocus**: 新開発のRust製大規模分散重複除去ツール

#### 4. トピック・品質分類
- **WebOrganizer**: 24トピックに分類（distilledしたfastTextモデル使用）
- **品質分類器**: OpenHermes-2.5、ELI5、UltraChat-200k、WildChat-1Mを正例としてfastText分類器を学習
- 24トピック × 20品質層 = **480のサブセット**に分割

### データミックスの決め方

> 詳細: [データミックス最適化 補足](./03-data-mixing.md)

#### Constrained Data Mixing

3段階のswarm-basedアプローチ：

1. **Swarm構築**: 30Mパラメータのproxyモデルを複数学習（各5B tokens）
2. **タスクごとの回帰**: 各mixのBPB（bits-per-byte）を予測するGLMを学習
3. **Mix最適化**: 最大アップサンプリング制約（4-7x）下で平均BPBを最小化

#### Quality-aware Upsampling

- 単純なフィルタリングではなく、高品質データを選択的にアップサンプリング
- 上位5%のデータを最大7xまで繰り返し
- 各トピックで別々のアップサンプリングカーブを適用

---

## 4. 学習手法

### 学習設定

| パラメータ | 7B | 32B |
|------------|-----|------|
| バッチサイズ | 4M tokens | 8M tokens |
| シーケンス長 | 8,192 | 8,192 |
| ピーク学習率 | 3×10⁻⁴ | 6×10⁻⁴ |
| 最終学習率 | ピークの10% | ピークの10% |
| Warm-up | 2,000 steps | 2,000 steps |
| LRスケジュール | Cosine | Cosine |
| 精度 | bfloat16 | bfloat16 |
| スループット | 7,700 tok/sec/GPU | 1,960 tok/sec/GPU |
| MFU | 約43% | 約41% |

### 分散学習の手法

- **OLMo-core**コードベースを使用
- PyTorchのtorch.compile()を活用
- カスタムカーネル: FlashAttention-2、言語モデリングヘッド（Liger kernel）
- 非同期・バッチ化されたメトリクス収集
- 非同期チェックポイント書き込み
- 長文脈拡張時: 8-way Context Parallelism（CP）

### 学習の安定化テクニック

1. **学習率スケジューリング**
   - 7B: 前半cosine→後半stretch（1エポック5.93Tトークンまで）
   - 32B: cosine、5.5Tで打ち切り

2. **Model Souping（モデル平均化）**
   - 異なるシードの複数ランを平均化
   - 32B midtraining: 2つのラン結果をマージ
   - 長文脈拡張: 最後の3チェックポイントをマージ

3. **評価中の性能推定**
   - 7B: 定期的に学習率をアニールして評価
   - 32B: 4チェックポイントの重み平均で評価（Li et al., 2025の手法）

---

## 5. 学習ステージ

### Stage 1: Pretraining（事前学習）

- **トークン数**: 7B: 5.93T, 32B: 5.5T
- **データ**: Dolma 3 Mix
- **期間**: 約47日（32B、クラッシュ含む）

### Stage 2: Midtraining

> 詳細: [Midtraining・合成データ 補足](./04-midtraining.md)

- **トークン数**: 100B
- **データ**: Dolma 3 Dolmino Mix

#### Dolma 3 Dolmino Mix の構成

| カテゴリ | 主なソース | トークン数 |
|----------|------------|------------|
| Math（合成） | TinyMATH, CraneMath, MegaMatt, Dolmino Math | 約20B |
| Code | Stack-Edu (FIM), CraneCode | 約20B |
| QA（合成） | Reddit-to-Flashcards, Wiki-to-RCQA, Nemotron | 約14B |
| Thinking traces | Meta-reasoning, OMR rewrite, QwQ traces等 | 約8B |
| Instruction | Tulu 3 SFT, Flan | 約6B |
| Web/PDF（高品質） | Common Crawl HQ, olmOCR PDFs | 約32B |

#### 新規合成データセット

| データセット | 説明 | 効果 |
|--------------|------|------|
| **CraneMath** | FineMath4+をQwen3でrewrite | MATH +18.5pt, GSM8K +27.4pt |
| **MegaMatt** | MegaMath-Web-ProをQwen3でrewrite | MATH +8.0pt, GSM8K +13.0pt |
| **TinyMATH** | MATHの各問題から100問生成 | MATH +13.2pt, GSM8K +13.9pt |
| **CraneCode** | SwallowCode手法をQwen3で再現 | HumanEval +13.5pt |
| **Meta-reasoning** | 認知能力タスク（backtracking等） | 大幅なmath/code改善 |

### Stage 3: Long-context Extension（長文脈拡張）

> 詳細: [長文脈拡張 補足](./05-long-context.md)

- **トークン数**: 7B: 50B, 32B: 100B
- **データ**: Dolma 3 Longmino Mix（34% long + 66% midtrain）
- **目標context**: 65,536 tokens

#### 拡張手法

| 要素 | 選択 |
|------|------|
| RoPE拡張 | YaRN（Full Attention層のみに適用） |
| ドキュメントパッキング | Best-fit packing |
| マスキング | Intra-document masking |
| 並列化 | 8-way Context Parallelism |

#### 合成データ拡張

- **CWE (Common Word Extraction)**: 頻出単語の出現回数を問うQA生成
- **REX (Rewriting Expressions)**: 12種類のvignette形式でリライト（対話、クイズ等）

---

## 6. 事後学習（Post-training）

### 三段階パイプライン

```
Base → SFT → DPO → RLVR → Final Model
```

### 6.1 SFT（Supervised Fine-Tuning）

#### Dolci Think SFT の構成

| カテゴリ | 主なソース | 規模 |
|----------|------------|------|
| Math | OpenThoughts3+, SYNTHETIC-2-Verified | 約85万 |
| Code | OpenThoughts3+, Dolci Think Python Algorithms | 約55万 |
| Chat & IF | WildChat, Persona IF, OpenAssistant | 約45万 |
| Safety | CoCoNot, WildGuardMix, WildJailbreak | 約9万 |
| その他 | Aya, TableGPT | 約10万 |

#### SFTの主要な知見

- **モデル生成**: QwQ-32B（Math/Code）、DeepSeek R1（Chat/Safety）で思考トレースを生成
- **フィルタリング**: 不完全なトレース、ドメイン固有エラー、過度の繰り返し等を除去
- **トピックフィルタリング**: OpenAI taxonomy使用
- **デコンタミネーション**: 8-gramマッチング（50%閾値）

### 6.2 DPO（Direct Preference Optimization）

#### Delta Learning

選好学習で重要なのは「選択と棄却の品質差（デルタ）」：

| 設定 | Chosen | Rejected |
|------|--------|----------|
| Think | Qwen 3 32B (thinking) | Qwen 3 0.6B (thinking) |
| Instruct | Qwen 3 32B | Qwen 3 0.6B |

#### 主要な知見

- **SFTでは改善しないデータでもDPOで改善可能**
  - Qwen3-32Bの出力でSFTすると性能低下
  - 同じデータでもweaker modelとペアにしてDPOすると大幅改善
- **DPOはSFTより良いRL開始点を提供**
- **データ量の最適値はタスク依存**（50-100K程度で飽和するものも）

#### Instruct向け追加改善

- **Multi-turn preferences**: 合成会話で複数ターン対応
- **Length control**: chosen/rejected間の長さ差を100トークン以下に制限
  - 冗長性を抑え、ユーザビリティ向上

### 6.3 RLVR（Reinforcement Learning with Verifiable Rewards）

> 詳細: [GRPO / OlmoRL 補足](./06-grpo-olmorl.md)

#### OlmoRL アルゴリズム

GRPOをベースに以下の改良を統合：

| 改良 | 説明 |
|------|------|
| Zero gradient filtering | 全サンプルが同一報酬のグループを除外 |
| Active sampling | 非ゼロ勾配サンプルでバッチを埋める |
| Token-level loss | サンプル単位でなくトークン単位で正規化 |
| No KL loss | KLペナルティなし |
| Clip higher | 上限クリッピングを下限より高く設定 |
| Truncated importance sampling | 重要度サンプリング比を打ち切り |
| No std normalization | advantage計算で標準偏差正規化なし |

#### Verifier（検証器）

| ドメイン | 検証方法 |
|----------|----------|
| Math | SymPy比較、正解なら報酬1 |
| Code | テストケース実行、全パスなら報酬1 |
| IF（Instruction Following） | 制約チェック関数、全満たせば報酬1 |
| Chat（参照あり） | LM judge（0-1スコア） |
| Chat（オープン） | LM judge（0-1スコア） |

#### インフラ改善

| 改善 | 効果 |
|------|------|
| Continuous batching | 生成長のばらつきによる無駄計算を54%削減 |
| Inflight updates | KVキャッシュ破棄なしで重み更新、最大4x高速化 |
| Fully asynchronous | 学習とrollout生成を非同期化 |

#### 学習設定

| パラメータ | 値 |
|------------|-----|
| 最大応答長 | 32K tokens |
| 平均応答長 | 約14.6K tokens |
| Training/Inference GPU比 | 8:20ノード（32B） |
| 32B RL期間 | 約5日（初期）→21日（3.1まで延長） |

---

## 7. 評価結果

### Olmo 3 Think 32B vs 他モデル

| ベンチマーク | Olmo 3.1 Think 32B | Qwen 3 32B | Qwen 2.5 32B |
|--------------|---------------------|------------|--------------|
| MATH | **96.2** | 95.4 | 80.2 |
| AIME 2024 | 80.6 | 80.8 | 15.7 |
| AIME 2025 | **78.1** | 70.9 | 13.4 |
| HumanEvalPlus | **91.5** | 91.2 | 82.6 |
| LiveCodeBench v3 | 83.3 | **90.2** | 49.9 |
| IFEval | **93.8** | 86.5 | 81.9 |
| IFBench | **68.1** | 37.3 | 36.7 |
| MMLU | 86.4 | **88.8** | 84.6 |
| AlpacaEval 2 LC | 69.1 | 75.6 | 81.9 |

### Olmo 3 Base 32B vs 他のfully-openモデル

| ベンチマーク | Olmo 3 32B | Marin 32B | Apertus 70B |
|--------------|------------|-----------|-------------|
| Math | **61.9** | 49.3 | 39.7 |
| Code | **39.7** | 30.8 | 23.3 |
| MC STEM | 74.5 | **75.9** | 70.0 |
| MC Non-STEM | **85.6** | 84.5 | 78.5 |
| GenQA | 79.8 | **80.3** | 75.0 |

### 長文脈評価（RULER）

| Model | 4K | 8K | 16K | 32K | 65K |
|-------|-----|-----|------|------|------|
| Olmo 3 32B | 96.1 | 94.6 | 90.4 | 86.2 | 79.7 |
| Qwen 2.5 32B | 96.0 | 94.5 | 95.1 | 92.7 | 80.7 |
| Gemma 3 27B | 84.5 | 84.2 | 85.4 | 87.1 | 84.6 |

---

## 8. 技術的な学び・教訓

### 成功した戦略

1. **Model Souping（モデルマージ）**
   - Midtraining: 異なるシードの2ランをマージで性能向上
   - Long-context: 最後の3チェックポイントをマージ

2. **Delta Learning**
   - SFTで改善しないデータでもDPOで改善可能
   - 強いモデル vs 弱いモデルのペアが重要

3. **Thinking/Instruction dataの事前導入**
   - Midtrainingに含めることでベースモデル性能も向上

4. **特殊トークンはSFTまで待つ**
   - `<|im_start|>`等をmidtrainingに含めると評価時に出力してしまう

5. **Active samplingでRL安定化**
   - 非ゼロ勾配サンプルを連続的にサンプリング
   - バッチサイズ維持、学習の安定化

### 失敗・困難だった点

1. **Quality-aware upsamplingの複雑さ**
   - トピックごとに別々のカーブが必要
   - Midtrainingでは単純な自然分布が同等の性能

2. **LongPplフィルタリング**
   - gzipフィルタリングより良くならなかった

3. **DPOデータ量のチューニング**
   - タスクによって最適量が異なる
   - 多すぎると性能低下（early stoppingが重要）

4. **RL学習の計算コスト**
   - 推論が学習の8-14倍のGPUを使用
   - 生成長のばらつきが大きな無駄を生む

### デコンタミネーション

- **decon**パッケージを新開発
- 2フェーズ: 検出（n-gramマッチング）→ クラスタ拡張
- FlanやNemotronに多くのベンチマークデータが含まれていた
- GSM8Kは完全にリークしていたが、フォーマット不一致で性能影響なし

### 今後の課題

1. **RL理論の発展**
   - ハイパーパラメータ最適化が経験的
   - 複数ドメインのRLにおける相互作用の理解

2. **長文脈拡張の効率化**
   - 現状50-100Bトークン必要
   - より少ないトークンでの拡張手法

3. **評価コストの削減**
   - reasoningモデルの評価は計算コストが高い
   - 開発計算量の10-20%を評価が占める

---

## 9. コスト情報

### 学習にかかった時間（32B）

| ステージ | 期間 | GPU |
|----------|------|-----|
| Pretraining | 約47日 | 512→1024 H100 |
| Midtraining | 約1.5日 | 512×2 H100 |
| Long-context | 約1日 | 1024 H100 |
| Post-training (SFT/DPO/RL) | 約9日 | 256-64 H100 |
| **合計** | **約56日** | **1024 H100** |

### 推定コスト

- H100 \$2/hour として: **約\$2.75M**
- DeepSeek V3のH800時間換算\$5.576Mより安価

---

## 10. 重要なハイパーパラメータまとめ

| ステージ | パラメータ | 7B | 32B |
|----------|------------|-----|------|
| Pretrain | トークン数 | 5.93T | 5.5T |
| Pretrain | バッチサイズ | 4M | 8M |
| Pretrain | ピーク学習率 | 3×10⁻⁴ | 6×10⁻⁴ |
| Midtrain | トークン数 | 100B | 100B |
| Long-ctx | トークン数 | 50B | 100B |
| Long-ctx | YaRN適用 | Full層のみ | Full層のみ |
| SFT | エポック数 | 2 | 2 |
| DPO | エポック数 | 1 | 1 |
| RL | 最大応答長 | 32K | 32K |
| RL | クリップ下限 | - | - |
| RL | クリップ上限 | 高め設定 | 高め設定 |

---

## 重要なポイント

### 技術的貢献

1. **完全オープンなModel Flow**: データからコードまで全て公開
2. **Duplodocus**: Rustによる大規模分散重複除去
3. **OlmoRL**: GRPO + 複数改良の統合RLフレームワーク
4. **Delta Learning**: DPOにおける品質差の重要性
5. **Constrained Data Mixing**: swarmベースのデータ配合最適化

### 実験的知見

1. Model Soupingの有効性
2. Midtrainingへのthinking traces導入
3. Length controlによるDPO改善
4. Active samplingによるRL安定化
5. デコンタミネーションの重要性と実装

### スケール・コスト

1. 56日、1024 H100で32Bモデル完成
2. 推論が学習の8-14倍のGPU使用（RL時）
3. Continuous batching で54%の無駄削減
