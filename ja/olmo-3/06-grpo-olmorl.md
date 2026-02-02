# GRPO と OlmoRL

本文書では、Olmo 3で採用されているGRPOアルゴリズムと、Allen AIが開発したOlmoRLフレームワークについて解説する。

## 1. GRPOの基本概念

<details>
<summary><strong>RLHFの歴史的背景: InstructGPT から GRPO まで</strong></summary>

**RLHFの発展史**

LLMにおける強化学習の手法は急速に進化してきた。

**Phase 1: 報酬モデル + PPO（2022-2023）**

**InstructGPT / ChatGPT の手法:**
```
1. SFT: 人間のデモンストレーションで教師あり学習
2. RM: 人間の選好から報酬モデルを学習
3. PPO: 報酬モデルを使ってポリシー最適化
```

**課題:**
- 報酬モデルのハック（reward hacking）
- Criticネットワークの学習不安定
- メモリ効率の悪さ（4モデル必要）

**Phase 2: RL-free選好学習（2023-2024）**

**DPO (Direct Preference Optimization):**
- 報酬モデルなしで選好を直接最適化
- 暗黙的に報酬を定義（log π_θ - log π_ref）
- オフライン学習が可能

**派生手法:** IPO, KTO, ORPO, SimPO など

**課題:**
- オフラインデータに依存
- 連続的な報酬を扱えない

**Phase 3: オンライン・Critic-free RL（2024-2025）**

**GRPO / OlmoRL:**
- Criticなし、参照モデルなし（オプション）
- グループ相対比較で学習
- Verifiable rewardsと相性が良い

**特に有効なタスク:**
- Math（SymPyで検証）
- Code（テスト実行で検証）
- Instruction Following（制約チェック）

**手法の系譜図**

```
InstructGPT (2022)
    │
    ├─→ PPO variants
    │      └─→ RLOO (Williams, 1992復活)
    │
    └─→ RL-free
           ├─→ DPO (2023) → IPO, KTO, SimPO
           │
           └─→ Online variants
                  └─→ GRPO (2024) → Dr GRPO, DAPO, OlmoRL
```

**参考**:
- [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)

</details>

### 1.1 GRPOとは

GRPO (Group Relative Policy Optimization) は、DeepSeek社が提案した強化学習アルゴリズムで、LLMのRLHF/RLVRにおいてPPOの代替として設計された。

**核心的なアイデア**: 同じプロンプトに対して複数の応答（グループ）を生成し、グループ内の相対的な品質差を学習シグナルとして活用する。

### 1.2 PPOとの比較

| 観点 | PPO | GRPO |
|------|-----|------|
| Value Network | 必要（Critic） | 不要 |
| 参照モデル | 必要（KL計算用） | 不要（オプション） |
| Advantage計算 | GAE (Generalized Advantage Estimation) | グループ内相対報酬 |
| メモリ使用量 | 高い（2モデル必要） | 低い（1モデルで可） |
| 実装複雑性 | 高い | 低い |

**PPOの課題**:
- Criticネットワークの学習が不安定
- 参照モデルのメモリオーバーヘッド
- ハイパーパラメータ調整が困難

**GRPOの利点**:
- Criticなしで学習可能
- グループ内比較により、報酬の絶対値ではなく相対的な品質を学習
- 実装がシンプルで、スケーラビリティが高い

### 1.3 DPOとの比較

| 観点 | DPO | GRPO |
|------|-----|------|
| 学習タイプ | オフライン（事前収集データ） | オンライン（動的生成） |
| データ要件 | chosen/rejectedペア | 同一プロンプトの複数応答 |
| 報酬モデル | 暗黙的（reference model） | 明示的（verifier等） |
| 適用場面 | Preference learning | Verifiable rewards |

**DPOの限界**:
- 事前に収集したデータに依存（オフライン）
- 現在のポリシーの能力を反映しない
- バイナリ比較のみ（連続的な報酬を扱えない）

**GRPOの優位性**:
- オンラインで現在のポリシーから生成
- 連続的な報酬を活用可能
- Math/Codeなど検証可能なタスクに適している

---

## 2. OlmoRLのアルゴリズム改良

OlmoRLは、vanilla GRPOに対して複数の重要な改良を加えている。これらはDAOP、Dr GRPOなど最近の研究知見を統合したものである。

### 2.1 Zero Gradient Filtering

**問題**:
グループ内の全ての応答が同じ報酬（全正解または全不正解）の場合、advantage = 0となり勾配が発生しない。このようなサンプルで学習すると：
- 計算資源の無駄
- 効果的なバッチサイズの減少
- 学習の不安定化

**解決策**:
報酬の標準偏差がゼロのグループを除外する。

```python
# 疑似コード
for group in batch:
    rewards = [r(x, y_i) for y_i in group]
    if std(rewards) == 0:
        continue  # このグループをスキップ
    # 学習を実行
```

**効果**:
DAPO (Yu et al., 2025) で提案。勾配シグナルのない無駄な計算を排除し、学習効率を向上。

### 2.2 Active Sampling

**問題**:
Zero gradient filteringにより有効バッチサイズが減少すると、学習が不安定になる。DAPOのdynamic samplingは3倍のオーバーサンプリングを行うが、非効率。

**OlmoRLの解決策**:
非同期フレームワークを活用し、連続的にサンプリングしてバッチを満たす。

```
1. 目標バッチサイズNを設定
2. Actorが継続的に応答を生成
3. 非ゼロ勾配のグループのみを収集
4. N個集まったら学習ステップを実行
5. 繰り返し
```

**効果**:
- オーバーサンプリングの無駄を削減
- バッチサイズを一定に保つ
- 学習の安定化（vanilla GRPOでは学習が進むにつれバッチサイズが減少する問題があった）

### 2.3 Token-level Loss

**問題**:
サンプルレベルの損失正規化は、長い応答に対するバイアスを生む。

**vanilla GRPOの損失**:
```
L = (1/B) * sum_samples(sum_tokens(loss))
```
これは長い応答ほど損失への寄与が大きくなる。

**Token-level lossの損失**:
```
L = (1/total_tokens) * sum_samples(sum_tokens(loss))
```
全トークン数で正規化することで、応答長によるバイアスを除去。

**数式**:
```
J(θ) = (1 / Σ|y_i|) * Σ Σ loss(t)
```

**効果**:
長い応答を不当に優遇/罰することを防ぎ、公平な学習シグナルを提供。

### 2.4 Clip Higher

**標準のPPO/GRPOクリッピング**:
```
clip(r_t, 1-ε, 1+ε)
```
通常、ε_low = ε_high = 0.2など対称的に設定。

**Clip Higherの設定**:
```
clip(r_t, 1-ε_low, 1+ε_high)  where ε_high > ε_low
```
例: ε_low = 0.2, ε_high = 0.28

**直感的な説明**:
- 下限クリップ（1-ε_low）: 悪い更新の制限（保守的）
- 上限クリップ（1+ε_high）: 良い更新の許容（積極的）

上限を緩めることで、**良い方向への更新をより大きく許容**する。

**なぜ効果があるか**:
- 正のadvantageを持つトークン（良い応答）に対する更新を促進
- より積極的な探索を可能にしつつ、崩壊は防ぐ
- DAPO (Yu et al., 2025) で提案・検証

### 2.5 Truncated Importance Sampling

**問題**:
OlmoRLでは推論エンジン（vLLM）と学習エンジン（PyTorch）が分離されている。両者の数値計算の違いにより、同じ重みでも確率が微妙に異なる。

**解決策**:
推論時と学習時の確率の比（importance ratio）を計算し、損失に乗じる。ただし、この比が大きくなりすぎると不安定になるため、閾値ρで切り詰める。

**数式**:
```
truncated_ratio = min(π_train(y|x) / π_vllm(y|x), ρ)
loss *= truncated_ratio
```

**効果**:
- 推論・学習エンジン間の不整合を補正
- 外れ値の影響を制限
- Yao et al. (2025) で提案

### 2.6 標準偏差正規化の除去

**vanilla GRPOのAdvantage計算**:
```
A_i = (r_i - mean(r)) / std(r)
```

**OlmoRLのAdvantage計算**:
```
A_i = r_i - mean(r)
```

**標準偏差正規化の問題（difficulty bias）**:
- 簡単な問題: 全て正解 → std ≈ 0 → A が巨大に
- 難しい問題: 全て不正解 → std ≈ 0 → A が巨大に
- 適度な難易度: std > 0 → A は適切

つまり、極端に簡単/難しい問題のadvantageが不当に増幅される。

**効果**:
Dr GRPO (Liu et al., 2025b) で提案。難易度によるバイアスを除去し、一貫した学習シグナルを提供。

---

## 3. KLペナルティなしで学習できる理由

### 3.1 従来のKLペナルティの役割

```
L_total = L_RL - β * KL(π_θ || π_ref)
```

KLペナルティは以下を防ぐために導入された：
1. **モード崩壊**: 報酬を最大化する特定の応答パターンへの収束
2. **報酬ハッキング**: 報酬モデルの脆弱性を悪用
3. **言語能力の劣化**: 基本的な言語モデリング能力の喪失

### 3.2 なぜOlmoRLではKLなしで安定するか

**1. Verifiable Rewards**
- Math/Codeは正解が明確に検証可能
- 報酬ハッキングの余地が少ない
- 「正解を出す」以外の抜け道がない

**2. グループ相対比較**
- 絶対的な報酬ではなく相対比較
- 極端な出力への偏りが抑制される
- 同じプロンプト内での多様性が維持される

**3. クリッピング機構**
- PPOスタイルのクリッピングが過度な更新を制限
- 1ステップでの変化量が制約される

**4. Token-level正規化**
- 長い応答への偏りを防ぐ
- 特定のパターンへの収束を抑制

**5. 実証的知見**
Olmo 3の実験で、KLを外しても：
- 過最適化が発生しない
- 学習が不安定化しない
- むしろ制約が緩くなり性能向上

> "We remove the KL loss as a common practice ... as it allows less-restricted policy updates, and removing it does not lead to over-optimization or destabilized training." (Olmo 3 Paper)

---

## 4. Verifierの詳細

OlmoRLは4つのドメインに対応したVerifierを使用する。

### 4.1 Math Verifier

**手法**: ルールベース + SymPy
```
1. 応答から最終回答を抽出
2. 基本的な正規化（空白、形式）
3. SymPyで数学的等価性を判定
4. 報酬: 1（正解） or 0（不正解）
```

**特徴**:
- `1/2` と `0.5` の等価性を認識
- 数式の異なる表現形式を処理
- バイナリ報酬（正解/不正解）

### 4.2 Code Verifier

**手法**: テストケース実行
```
1. 応答からコードを抽出
2. AWS Lambdaで分散実行
3. テストケースの通過率を計算
4. 報酬:
   - パターンA: 通過率（0.0〜1.0の連続値）
   - パターンB: 全通過で1、それ以外は0
```

**AWS Lambda採用の理由**:
- 検証がトレーナーをブロックしない
- スケーラブル
- 時間計算量のペナルティも含むテストケースに対応

### 4.3 Instruction Following (IF) Verifier

**手法**: 制約チェック関数群
```
1. プロンプトから制約を抽出
   例: "500語以下で書いて", "箇条書きで3点", "敬語を使って"
2. 各制約に対応するチェック関数を実行
3. 報酬: 全制約を満たせば1、それ以外は0
```

**データソース**: IF-RLVR (Pyatkin et al., 2025)
- IFEval、IFBench-Trainから制約を抽出
- 最大5つの制約を組み合わせ

### 4.4 Chat Verifier (LLM-as-a-Judge)

**2つのバリエーション**:

**A. Reference-based（参照付き）**
```
1. LLM Judge（Qwen3 32B）に入力
2. 参照回答との比較を指示
3. 0〜1のスコアを出力
```

**B. Open-ended（参照なし）**
```
1. LLM Judgeに入力
2. 応答品質のみで評価
3. 0〜1のスコアを出力
```

**設定**:
- Judge: Qwen3 32B（thinking mode OFF）
- Max input: 32,768 tokens
- Max output: 2,048 tokens

---

## 5. インフラ改善

OlmoRLは、長い推論トレースを効率的に処理するための大規模なインフラ改善を含む。

### 5.1 Continuous Batching

**問題**: Static Batching
```
[S1 ████████████████ EOS ░░░░░░░░░░░░]
[S2 ██████ EOS ░░░░░░░░░░░░░░░░░░░░░░]
[S3 ██████████████████████████████ EOS]
[S4 ████████████ EOS ░░░░░░░░░░░░░░░░]
```
(░ = 無駄な計算)

**Static Batchingの無駄**:
```
無駄率 = (max_length - avg_length) / max_length
Olmo 3の場合: (32K - 14.6K) / 32K = 54%
```
つまり、半分以上の計算が無駄。

**Continuous Batching**:
```
[S1 ████████████████ EOS][S5 ████████...]
[S2 ██████ EOS][S6 ██████████████████...]
[S3 ██████████████████████████████ EOS]
[S4 ████████████ EOS][S7 ████████████...]
```
終了したスロットに即座に新しいシーケンスを投入。

**効果**:
- GPUの空き時間を削減
- 推論スループットの大幅向上
- 特に長いシーケンスで効果大

### 5.2 Inflight Updates

**従来の同期更新**:
```
1. 全Actorが生成完了を待つ
2. KVキャッシュを破棄
3. 重みを同期
4. 生成を再開
```
→ GPUのアイドル時間が発生

**Inflight Updates**:
```
1. 重み更新を即座にActorに送信
2. 生成を停止せずに重みを差し替え
3. KVキャッシュを無効化せず継続
```

**要件**:
- vLLMのスレッドセーフな重み更新
- 非同期通信の実装

**効果**:
- 最大4倍のスループット向上
- Piche et al. (2025) に基づく実装

### 5.3 分散RLアーキテクチャ

```
                    ┌─────────────┐
                    │   Prompts   │
                    │    Queue    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Actor 1 │  │ Actor 2 │  │ Actor N │
        │ (vLLM)  │  │ (vLLM)  │  │ (vLLM)  │
        └────┬────┘  └────┬────┘  └────┬────┘
              │            │            │
              └────────────┼────────────┘
                           ▼
                    ┌─────────────┐
                    │   Results   │
                    │    Queue    │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Learner   │
                    │ (DeepSpeed) │
                    └─────────────┘
```

**構成（32Bモデル）**:
- Learner: 8 H100ノード
- Actors: 20ノード（各ノード1 vLLMインスタンス）
- 推論:学習 = 5:1（GPU時間ベース）

**非同期の利点**:
- 推論と学習のオーバーラップ
- リソースの効率的活用
- スケーラブルなアーキテクチャ

### 5.4 インフラ改善の効果

| 改善 | Tokens/sec | 累積改善 |
|------|------------|----------|
| OLMo 2 baseline | 881 | 1x |
| + Continuous batching | 975 | 1.1x |
| + Better threading | 1358 | 1.5x |
| + Inflight updates | 2949 | 3.3x |

**総合効果**: 約4倍の高速化

---

## 6. OlmoRLの目的関数（完全形）

最終的な目的関数は以下の要素を統合：

```
J(θ) = (1 / Σ|y_i|) * Σ_i Σ_t min(ρ_t^trunc, ρ) * min(r_t * A_t, clip(r_t) * A_t)
```

ここで:
- `Σ|y_i|`: Token-level正規化
- `ρ_t^trunc = min(π_train/π_vllm, ρ)`: Truncated importance sampling
- `r_t = π_θ(y_t|x,y_<t) / π_old(y_t|x,y_<t)`: Importance ratio
- `clip(r_t) = clip(r_t, 1-ε_low, 1+ε_high)`: Clip higher
- `A_t = r(x,y) - mean(r)`: Advantage（標準偏差正規化なし）

---

## 7. よくある質問と回答

**Q: GRPOがPPOより優れている点は？**

**A:** GRPOはCriticネットワークと参照モデルが不要なため、メモリ効率が良く実装もシンプル。同一プロンプトの複数応答からグループ相対比較を行うことで、報酬の絶対値ではなく相対的な品質を学習できる。特にMath/Codeなど検証可能なタスクで有効。

**Q: なぜKLペナルティなしで安定するのか？**

**A:** 主に3つの理由がある。(1) Verifiable rewardsは報酬ハッキングの余地が少ない、(2) グループ相対比較により極端な出力への偏りが抑制される、(3) クリッピング機構が過度な更新を制限する。Olmo 3の実験では、KL除去により制約が緩くなり、むしろ性能が向上した。

**Q: Token-level lossの重要性は？**

**A:** サンプルレベルの正規化では長い応答ほど損失への寄与が大きくなり、モデルが長い応答を出す/出さないバイアスを学習してしまう。Token-levelで正規化することで、応答長に関係なく公平な学習シグナルを提供できる。

**Q: Active samplingはどのような問題を解決するか？**

**A:** Zero gradient filteringでバッチから除外されるサンプルがあると有効バッチサイズが減少する。DAPOは3倍のオーバーサンプリングで対処したが非効率。OlmoRLは非同期フレームワークで継続的にサンプリングし、必要な数だけ収集することで効率的にバッチサイズを維持する。

**Q: Inflight updatesで4倍高速化できた理由は？**

**A:** 従来は重み同期のために全Actorの生成停止が必要だった。Inflight updatesでは生成を止めずに重みを差し替えるため、GPUのアイドル時間がほぼゼロになる。RLでは推論が計算のボトルネック（75%以上）であるため、この改善の効果が大きい。

---

## 参考文献

- Shao et al. (2024): GRPO original paper
- Yu et al. (2025): DAPO - Zero gradient filtering, Clip higher, Token-level loss
- Liu et al. (2025b): Dr GRPO - No std normalization
- Yao et al. (2025): Truncated importance sampling
- Piche et al. (2025): Inflight updates
- Olmo 3 Technical Report (2025): OlmoRL framework
