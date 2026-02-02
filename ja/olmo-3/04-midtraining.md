# Midtraining と合成データ

本文書では、Olmo 3の学習パイプラインにおけるMidtrainingステージと、そこで使用される合成データについて技術的詳細を解説する。

## 1. Midtrainingの位置づけ

### 1.1 3段階学習パイプライン

Olmo 3の学習は以下の3段階で構成される。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Pretraining                                                       │
│  - Data: Dolma 3 Mix (6T tokens)                                            │
│  - Goal: General language understanding & generation                        │
│  - Features: Web text, academic PDFs, code, diverse sources                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Midtraining                                                       │
│  - Data: Dolma 3 Dolmino Mix (100B tokens)                                  │
│  - Goal: Enhance Math, Code, QA, Reasoning capabilities                     │
│  - Features: High-quality synthetic data + Post-training preparation        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Long-context Extension                                            │
│  - Data: Dolma 3 Longmino Mix (50B-100B tokens)                             │
│  - Goal: Extend context length from 8K to 65K                               │
│  - Features: olmOCR science PDFs + synthetic long-form data                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  Post-training (SFT/DPO/RL)                                                 │
│  - Think / Instruct / RL-Zero variants                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Midtrainingの役割

Midtrainingは以下の重要な役割を果たす。

1. **ターゲット能力の強化**: Math, Code, QA, Reasoningなど特定ドメインの能力向上
2. **Post-training準備**: Instruction data や Thinking traces を含め、後続の SFT/RL に向けた基盤を構築
3. **効率的な能力注入**: 100B tokensという比較的少量のデータで大幅な性能向上を実現

---

## 2. Dolma 3 Dolmino Mix の詳細構成

### 2.1 全体構成

| カテゴリ | データソース | Pool | 100B Mix | 割合 |
|---------|-------------|------|----------|------|
| **Math (synth)** | TinyMATH Mind | 899M | 898M | 0.9% |
| **Math (synth)** | TinyMATH PoT | 241M | 241M | 0.24% |
| **Math (synth)** | CraneMath | 5.62B | 5.62B | 5.63% |
| **Math (synth)** | MegaMatt | 3.88B | 1.73B | 1.73% |
| **Math (synth)** | Dolmino Math | 10.7B | 10.7B | 10.7% |
| **Code** | StackEdu (FIM) | 21.4B | 10.0B | 10.0% |
| **Python (synth)** | CraneCode | 18.8B | 10.0B | 10.0% |
| **QA (synth)** | Reddit To Flashcards | 21.6B | 5.90B | 5.9% |
| **QA (synth)** | Wiki To RCQA | 4.22B | 3.0B | 3.0% |
| **QA (synth)** | Nemotron Synth QA | 487B | 5.0B | 5.0% |
| **Thinking (synth)** | Math Meta-Reasoning | 1.05B | 381M | 0.38% |
| **Thinking (synth)** | Code Meta-Reasoning | 1.27B | 459M | 0.46% |
| **Thinking (synth)** | Program-Verifiable | 438M | 159M | 0.16% |
| **Thinking (synth)** | OMR Rewrite FullThoughts | 850M | 850M | 0.85% |
| **Thinking (synth)** | QWQ Reasoning Traces | 4.77B | 1.87B | 1.87% |
| **Thinking (synth)** | General Reasoning Mix | 2.48B | 1.87B | 1.87% |
| **Thinking (synth)** | Gemini Reasoning Traces | 246M | 246M | 0.25% |
| **Thinking (synth)** | Llama Nemotron Reasoning | 20.9B | 1.25B | 1.25% |
| **Thinking (synth)** | OpenThoughts2 Reasoning | 5.6B | 1.25B | 1.25% |
| **Instruction (synth)** | Tulu 3 SFT | 1.61B | 1.1B | 1.1% |
| **Instruction (synth)** | Dolmino 1 Flan | 16.8B | 5.0B | 5.0% |
| **PDFs** | olmOCR science PDFs (HQ) | 240B | 4.99B | 5.0% |
| **Web** | STEM-Heavy Crawl | 5.21B | 4.99B | 5.0% |
| **Web** | Common Crawl (HQ) | 1.32T | 22.4B | 22.5% |
| **Total** | | 2.19T | 99.95B | 100% |

### 2.2 データキュレーションの方法論

#### Microanneal Framework

軽量で分散的なフィードバックループを使用した評価手法。

```
標準的なMicroanneal設定:
1. ターゲットデータセットを選択
2. 5B tokensをサンプリング
3. 5B web tokensとマッチング
4. 計10B tokensでAnnealing実行
5. Web-onlyベースラインと比較
```

この手法により、80以上のMicroannealを25種類の数学データソースに対して実行し、迅速にデータ品質を評価できる。

#### Integration Tests

100B tokenの候補ミックス全体での統合テスト。

- Microannealで有望なデータソースを特定後、実際の100Bミックスで検証
- 後続のSFT訓練後の性能も評価
- 5ラウンドの統合テストを実施

---

## 3. 各合成データセットの詳細

### 3.1 CraneMath (FineMath4+ のリライト)

#### 背景

SwallowMath (Fujii et al., 2025) は、FineMath4+ を Llama モデルでリライトした高品質な数学データセット。しかし、Llama Community License の制約があり、訓練したモデル名に "Llama" を含める必要がある。

#### CraneMathの生成手法

```
CraneMathの生成プロセス:

ソースデータ: FineMath4+ (Allal et al., 2025)
  - Common Crawlから数学教育コンテンツを抽出
  - 品質スコア4段階中3以上をフィルタリング

リライトモデル: Qwen3-32B (Yang et al., 2025a)
  - SwallowMathと同じプロンプトを使用
  - Llamaの代わりにQwen3を使用することでライセンス問題を回避

生成結果: 5.62B tokens
  - SwallowMath: 2.3B tokens
  - 差異: Qwen3は Llama より "chattier"（より冗長な出力）
```

#### Microanneal結果

| データソース | MATH改善 | GSM8K改善 | トークン数 |
|-------------|---------|----------|----------|
| SwallowMath | +16.0 pts | +24.5 pts | 3.6B |
| **CraneMath** | **+18.5 pts** | **+27.4 pts** | 5.6B |
| FineMath4+ | +1.5 pts | +1.2 pts | 9.6B |

CraneMathは元のFineMath4+より少ないトークン数で、大幅に高い効果を発揮。

### 3.2 MegaMatt

#### 背景

Megamath-Web-Pro-Max (Wang et al., 2025) は MegaMath-Web をLlamaでリライトしたデータセット。同様にライセンス問題あり。

#### MegaMattの生成手法

```
MegaMattの生成プロセス:

ソースデータ: Megamath-Web-Pro
  - MegaMath-Webのフィルタリング版
  - 2023年6月以降のCommon Crawlダンプから取得（より高品質）

リライトモデル: Qwen3-32B
  - Megamath-Web-Pro-Maxと同様のフィルタリング適用

生成結果: 3.88B tokens
  - 元のMegamath-Web-Pro-Max: 70B tokens
  - 最新データのみに絞ることで効率化
```

#### Microanneal結果

| データソース | MATH改善 | GSM8K改善 |
|-------------|---------|----------|
| Megamath-Web-Pro-Max | +7.0 pts | +13.3 pts |
| **MegaMatt** | **+8.0 pts** | **+13.0 pts** |
| MegaMath-Web | +1.3 pts | +0.7 pts |

### 3.3 TinyMATH

#### 概要

MATHベンチマークの訓練セット（7,500問題）から合成的に生成した高品質データ。

#### 生成手法

```python
# TinyMATHの生成プロセス
for problem in MATH_training_set:  # 7,500問題
    # 1. 100個の類似問題を生成
    similar_problems = generate_similar(problem, n=100)

    for new_problem in similar_problems:
        # 2. Pythonコード解法を生成 (Program-of-Thought)
        pot_solution = generate_python_solution(new_problem)

        # 3. 自然言語での解説を2種類生成 (MIND)
        mind_solutions = generate_conversational_solution(new_problem, variants=2)
```

#### 構成

| バリアント | 内容 | トークン数 |
|-----------|-----|----------|
| TinyMATH-PoT | Pythonコード解法 | 241M |
| TinyMATH-MIND | 会話形式の自然言語解説 | 899M |
| **合計** | | **1.14B** |

#### 性能効果

Microanneal (50/50 web mixで10B tokens):
- MATH: **+13.2 pts**
- GSM8K: **+13.9 pts**

**トークンあたりの効果が最も高い**データソース。ただし、ターゲット評価タスクに特化しているため、一般化能力とのトレードオフあり。

### 3.4 CraneCode (SwallowCode手法)

#### 背景

SwallowCode (Fujii et al., 2025) は高品質なコードリライトデータセット。Llamaモデル使用のためライセンス制約あり。

#### CraneCodeの生成手法

```python
# CraneCodeの生成プロセス（2段階リライト）

# Step 1: ソースデータ準備
source = "the-stack-v2-smol"  # Python subset
filtered_data = apply_filters(source, [
    syntax_error_filter,  # 構文エラー除去
    linter_output_filter,  # Linter警告除去
])

# Step 2: Stage 1 - スタイル改善 (SGCR)
for code in filtered_data:
    styled_code = rewrite_for_style(
        code,
        model="Qwen2.5-Coder-32B-Instruct",
        prompt=SWALLOW_STYLE_PROMPT  # Pythonスタイルガイド準拠
    )

# Step 3: Stage 2 - コード最適化 (SCOR)
for styled_code in stage1_output:
    optimized_code = rewrite_for_optimization(
        styled_code,
        model="Qwen2.5-Coder-32B-Instruct",
        prompt=SWALLOW_OPTIMIZATION_PROMPT
    )
```

#### 2段階リライトパイプライン

| Stage | 名称 | 目的 |
|-------|-----|-----|
| SGCR | Style-Guide Compliant Rewrite | Pythonスタイルガイドへの準拠 |
| SCOR | Style-Code Optimized Rewrite | コードの最適化・効率化 |

#### 性能比較

| データソース | トークン数 | HumanEval | CruxEval | MBPP |
|-------------|----------|-----------|---------|------|
| Pre-anneal baseline | - | 21.51 | 35.46 | 27.11 |
| SwallowCode | 10.0B | 31.80 | 35.74 | 34.67 |
| CraneCode (10B) | 10.0B | 26.51 | 33.28 | 34.94 |
| CraneCode (25B) | 18.87B | 35.06 | 35.92 | 31.72 |
| **CraneCode SGCR** | 18.87B | 33.78 | **41.75** | **36.76** |

CraneCodeはSwallowCodeと比較してトークンあたりの効果はやや劣るが、より多くのトークンを投入することで同等以上の性能を達成。

### 3.5 Meta-reasoning (認知能力タスク)

#### 理論的背景

Kargupta et al. (2025) と Gandhi et al. (2025) の研究により、ベースモデルにおけるmeta-reasoning能力がRL訓練の軌跡に大きく影響することが示されている。

#### 7つのコア認知能力

Olmo 3では以下の認知能力をターゲット。

| 能力 | 説明 | 参考文献 |
|-----|-----|---------|
| **Self-awareness** | 自己認識・自己モニタリング | Toy et al. (2024) |
| **Evaluation** | 解答や方略の評価能力 | Fleming and Daw (2017) |
| **Goal Management** | 目標設定・管理能力 | Ackerman and Thompson (2017) |
| **Hierarchical Organization** | 階層的な問題分解 | Haupt (2018) |
| **Backward Chaining** | 目標から逆算する推論 | Olieslagers et al. (2024) |
| **Backtracking** | 誤りからの修正・再試行 | Joyce (2009) |
| **Conceptual Reasoning** | 概念的・抽象的推論 | Markovits et al. (2015) |

#### データ生成プロセス

```python
# Meta-reasoningデータの生成

# 1. ソースデータの収集
math_sources = [
    "NuminaMath",      # Luo et al., 2025a
    "OpenMathReasoning"  # Moshkov et al., 2025
]
code_sources = [
    "CodeContests",    # Li et al., 2023a
    "APPS",            # Hendrycks et al., 2021a
    "PolyGen"          # Ahmad et al., 2025
]

# 2. 詳細なアノテーション付与（Pandalla-Math形式）
annotations = [
    "problem_classification",   # 問題の分類
    "difficulty_analysis",      # 難易度分析
    "solution_approaches",      # 解法アプローチ
    "common_pitfalls",          # よくある間違い
    "verification_methods"      # 検証方法
]

# 3. 各認知能力タスクに対するthinking trace生成
for capability in SEVEN_CORE_CAPABILITIES:
    for annotated_problem in annotated_data:
        thinking_trace = generate_trace(
            problem=annotated_problem,
            target_capability=capability,
            models=["GPT-4.1", "o4-mini"]
        )
```

#### 具体的なタスク例

- **Backward chaining**: 解答から元の数学問題を逆算
- **Debugging**: プログラムのバグを発見・修正
- **Solution verification**: 解答の正しさを検証

#### 性能効果

Microanneal結果（math/codeベースライン比較）:
- Minerva Math: **+14 pts**
- HumanEval: **+14 pts**
- MBPP: **+20 pts**

---

## 4. 思考トレース (Thinking Traces) の効果

### 4.1 Thinking Tracesの役割

Thinking tracesは、Post-training（特にOlmo 3 ThinkとOlmo 3 RL-Zero）の基盤を構築するためにMidtrainingに含まれる。

```
Preparation in Midtraining
        ↓
┌───────────────────────────────────────────────────┐
│ Thinking Traces in Base Model                     │
│ - Learning reasoning patterns                     │
│ - Acquiring verification/backtracking behaviors   │
│ - Internalizing structured thought processes      │
└───────────────────────────────────────────────────┘
        ↓
Development in Post-training (SFT → DPO → RL)
        ↓
Olmo 3 Think: Achieves superior RL trajectories
```

### 4.2 Thinking Tracesのソース

| ソース | トークン数 | 特徴 |
|-------|----------|------|
| Math Meta-Reasoning | 381M | 数学問題に対する認知能力タスク |
| Code Meta-Reasoning | 459M | コード問題に対する認知能力タスク |
| Program-Verifiable | 159M | プログラムで検証可能な推論タスク |
| OMR Rewrite FullThoughts | 850M | OpenMathReasoningの軽量リライト |
| QWQ Reasoning Traces | 1.87B | QWQモデルからの推論トレース |
| General Reasoning Mix | 1.87B | 多ドメインの推論データ |
| Gemini Reasoning Traces | 246M | Geminiからの推論トレース |
| Llama Nemotron Reasoning | 1.25B | Llama Nemotronからのトレース |
| OpenThoughts2 Reasoning | 1.25B | OpenThoughts2からのトレース |

### 4.3 Thinking Traces有無の比較

| 評価指標 | Thinking Traces なし | Thinking Traces あり | 差分 |
|---------|---------------------|---------------------|------|
| MC STEM | 63.6 | 64.9 | +1.3 |
| MC Non-STEM | 74.0 | 75.7 | +1.7 |
| GenQA | 66.7 | 68.1 | +1.4 |
| Math | 43.1 | 48.7 | **+5.6** |
| Code | 23.3 | 24.4 | +1.1 |
| FIM | 29.2 | 31.9 | +2.7 |
| **Average** | 48.8 | 50.7 | **+1.9** |

**重要な知見**: Thinking tracesとInstruction dataの包含は、Post-training前のベースモデル段階から一貫した性能向上をもたらす。

### 4.4 Program-Verifiable Reasoning

検証可能な推論データの特徴:

```python
# Program-Verifiable Reasoningの生成

# 1. プログラムで検証可能な問題を生成
problems = generate_verifiable_problems(programmatic=True)

# 2. GPT-4.1/o4-miniからthinking tracesを蒸留
for problem in problems:
    trace = distill_thinking_trace(
        problem,
        models=["GPT-4.1", "o4-mini"]
    )

# 3. Python検証器で正解をフィルタリング
verified_traces = filter_by_verifier(traces, python_verifier)
```

効果（250M tokens、5B microanneal）:
- GSM8K: +1-2 pts
- MBPP: +1-2 pts

---

## 5. Model Souping の適用

### 5.1 概要

Model Souping（モデル平均化）は、複数の独立した訓練ランからのチェックポイントを線形結合することで性能向上を図る手法。

### 5.2 Midtrainingでの適用 (Olmo 3 Base 32B)

```
Midtraining Run 1 (seed A)
         ↘
          ↘  Linear Average
           →  Final Merged Model
          ↗
         ↗
Midtraining Run 2 (seed B)
```

#### 性能改善結果

| タスククラスタ | Run 1 | Run 2 | Merged | 改善幅 |
|--------------|-------|-------|--------|-------|
| MC STEM | - | - | - | **+0.9** |
| GenQA | - | - | - | **+0.4** |
| Math | baseline | +1.3 | +2.9 | **+2.9 vs Run1** |
| MMLU | - | - | - | **+1.0** |
| GSM Symbolic | - | - | - | **+5.0 / +2.0** |

### 5.3 Long-context Extensionでの適用

チェックポイント平均化:

```python
# 最後の3つのチェックポイントを平均化
checkpoints = [
    checkpoint_step_10000,
    checkpoint_step_11000,
    checkpoint_step_11921  # 最終ステップ
]
final_model = linear_average(checkpoints)
```

### 5.4 SFT段階での適用

Olmo 3 Thinkの最終SFTチェックポイントは、異なる学習率で訓練した2つのチェックポイントの線形加重マージ。

```python
# mergekit (Goddard et al., 2024) を使用
final_sft = weighted_merge(
    checkpoint_lr_high,
    checkpoint_lr_low,
    weights=[w1, w2]
)
```

---

## 6. まとめ

### 6.1 合成データの階層構造

トークンあたりの効果（効果/token）:

```
Tier 1: 最高効率（ターゲット特化）
├── TinyMATH variants
│   └── MATH訓練セットから直接合成
│   └── 効果は高いが汎化にトレードオフ

Tier 2: 高効率（リライトデータ）
├── CraneMath, SwallowMath, MegaMatt
│   └── 自然データの高品質リライト
│   └── バランスの取れた効果

Tier 3: 低効率（自然データ）
├── FineMath4+, MegaMath-Web
│   └── 大量だが効果/tokenは低い
│   └── 基盤として重要
```

### 6.2 ライセンス問題への対処

<details>
<summary><strong>Llama Community License の制約と回避策</strong></summary>

**Llama Community License の概要**

Meta社の Llama 2/3 は "Community License" の下で公開されており、以下の制約がある：

1. **派生モデルの命名**: Llamaで生成したデータで学習したモデルは名前に "Llama" を含める必要がある
2. **利用規約の継承**: 下流の利用者にも同じ制約が適用
3. **商用利用制限**: 月間7億ユーザー以上のサービスには特別ライセンスが必要

**SwallowMath/SwallowCodeの問題**

Fujii et al. (2025) の SwallowMath/SwallowCode は:
- Llama 3.1/3.3 モデルでデータを生成
- 学習済みモデルに "Llama" を含める必要
- **Olmo 3のような完全オープンモデルには使用不可**

**Olmo 3の解決策**

同一のプロンプト・パイプラインを、ライセンス制約のないモデルで再実行:

```
SwallowMath (Llama 3.3 70B) → CraneMath (Qwen3-32B)
SwallowCode (Llama 3.1 70B) → CraneCode (Qwen2.5-Coder-32B)
```

**Qwenを選択した理由:**
- Apache 2.0ライセンス（商用利用自由）
- 派生モデルの命名制約なし
- Llamaと同等以上の生成品質

**参考**: [Llama 3 Community License](https://www.llama.com/license/)

</details>

Olmo 3の重要な貢献として、Llamaモデルで生成されたデータの再生成:

| オリジナル | 再生成版 | 使用モデル |
|-----------|---------|-----------|
| SwallowMath | CraneMath | Qwen3-32B |
| Megamath-Web-Pro-Max | MegaMatt | Qwen3-32B |
| SwallowCode | CraneCode | Qwen2.5-Coder-32B-Instruct |

これにより完全にオープンなライセンスでのデータ利用が可能に。

### 6.3 重要なポイント

1. **Midtrainingの意義**: Pretrainingで獲得した一般能力を、ターゲット能力（Math, Code, Reasoning）に効率的に特化
2. **Microanneal手法**: 10Bトークン規模の軽量実験でデータ品質を迅速評価
3. **合成データの効果**: リライトにより同じソースデータから大幅な性能向上
4. **Thinking Tracesの早期導入**: Post-trainingの成功はBase modelでの準備に依存
5. **Model Souping**: 複数ランの平均化で一貫した性能向上

---

## 参考文献

- Olmo 3 Technical Report (2025)
- Fujii et al. (2025) - SwallowMath, SwallowCode
- Allal et al. (2025) - FineMath
- Wang et al. (2025) - Megamath-Web-Pro-Max / OctoThinker
- Kargupta et al. (2025) - Meta-reasoning capabilities
- Gandhi et al. (2025) - Verification and backtracking in base models
- Wortsman et al. (2022) - Model soups
