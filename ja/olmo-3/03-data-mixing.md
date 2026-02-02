# データミックス最適化

Olmo 3における事前学習データの最適な混合比率を決定するための手法を解説する。9兆トークンを超える多様なデータソースから、6兆トークンの学習データを構築するための体系的アプローチ。

---

## 1. Constrained Data Mixing（制約付きデータ混合）

### 概要

Olmo 3のデータ混合手法は、RegMix、Data Mixing Laws、CLIMBなどの先行研究に基づくswarm-basedアプローチを採用している。この手法は3つのステージで構成される。

### Stage 1: Swarm Construction（Swarm構築）

**目的**: 混合比率の探索空間をサンプリングし、多数の小型proxyモデルを学習

**手法**:
- **Proxyモデル**: Olmo 3アーキテクチャに基づく**30Mパラメータ**の小型モデル
- **学習トークン数**: 3Bトークン（Chinchillaの5倍）
- **Swarmサイズ**: ドメイン数の約5倍（例: 24トピックなら120モデル程度）
- **サンプリング**: 各mixはDirichlet分布からサンプリング（自然分布を中心に）

```
# Dirichlet分布からの混合比率サンプリング
mix_weights ~ Dirichlet(α * natural_distribution)
```

**評価**: 各proxyモデルをBase Easyスイートで評価

### Stage 2: Per-task Regression（タスクごとのBPB回帰）

**目的**: 任意の混合比率からタスク性能を予測するモデルを構築

**手法**:
- 各proxyモデルは「混合比率 → タスク性能」のデータポイントを提供
- 性能指標: **bits-per-byte (BPB)** - 低いほど良い
- タスクごとに一般化線形モデル（GLM）をフィッティング

```
BPB_task(mix) = f(mix_weights; θ_task)
```

これにより、新しい混合比率候補に対する性能を高速に予測可能。

### Stage 3: Mix Optimization（Mix最適化）

**目的**: 制約条件下で最適な混合比率を探索

**最適化問題**:
```
minimize  Σ_task BPB_task(mix) / |tasks|   # 平均BPBを最小化
subject to:
  Σ_domain mix[domain] = 1                  # 合計は1
  mix[domain] * total_tokens <= available_tokens[domain] * max_repeat
```

**制約条件**:
- **アップサンプリング制約**: 各ドメインの反復回数は約4-7倍まで
- 6Tトークンの学習を目標とする際、トークン数が少ないドメインには自然と上限が課される

**最適化手法**: 事前分布または自然分布から初期化したガイド付き探索

---

## 2. Conditional Mixing（条件付き混合）

### 問題

実際の開発では、データソースは継続的に更新される（フィルタの改善、ドメインの追加、品質問題の修正など）。毎回swarmを全て再計算するのは計算コストが高い。

### 解決策

既存の最適化済みmixを**単一の仮想ドメイン**として固定し、その上で新規/変更されたドメインのみを対象にbase procedureを再実行。

```
Round 1: Web text (24 WebOrganizer topics) + source-level mix
         ↓ 結果を固定
Round 2: Web text固定 (75%) + Stack-Edu言語mix (25%)
         ↓ 結果を固定
Round 3: 上記固定 + PDF data (24 topics)
```

### 利点

- 探索空間の次元削減によりswarmサイズと計算コストを削減
- 遅れて完成するデータソース（PDFなど）を効率的に統合可能

---

## 3. Quality-aware Upsampling（品質を考慮したアップサンプリング）

### なぜフィルタリングでなくアップサンプリングか

従来手法（DCLMなど）では品質分類器に基づく**flat filtering**（上位25%を選択など）を使用。しかし、Olmo 3の実験では**品質に応じたアップサンプリング**がデータ制約下で優れた性能を示した。

**例**: 1Tトークンプールから250Bトークンを構築する場合
- Flat filtering: 上位25%を1回ずつ使用
- Quality-aware upsampling: 上位5%を7回、残りを1回使用

**性能比較（1Bモデル、100Bトークン、BPB）**:

| 手法 | QA Easy | Math Easy | Code Easy |
|------|---------|-----------|-----------|
| Top 50% (1.1x repeat) | 1.042 | 0.863 | 0.943 |
| Top 30% (1.8x repeat) | 1.031 | 0.870 | 0.880 |
| Top 10% (5.6x repeat) | 1.041 | 0.858 | 0.939 |
| Top 5% (11.1x repeat) | 1.065 | 0.843 | 0.930 |
| **Olmo 3 Upsampling** | **1.000** | **0.740** | **0.719** |

### 品質分類器の学習

**正例（高品質）**:
- OpenHermes-2.5
- ELI5 (Explain Like I'm 5)
- UltraChat-200k
- WildChat-1M

**負例（低品質）**:
- DCLM-RefinedWebから30GBをサンプリング

**分類器**: fastText（スケーラビリティのため、WebOrganizerのTransformerモデルから蒸留）

### トピック x 品質層の480サブセット

1. **トピック分類**: WebOrganizerで24カテゴリに分類
2. **品質分類**: 各トピック内で品質スコアのパーセンタイルを計算
3. **Vigintile分割**: 5パーセンタイル間隔で20バケットに分割

```
24 topics × 20 quality tiers = 480 disjoint subsets
```

これにより、トピックと品質の両方を細かく制御可能。

### アップサンプリングカーブ

**パラメトリック表現**: 切断べき指数関数族

```
f_{p,λ}(x) = {
  0,                      for x < a
  C(x - a)^p · e^{λ(x-a)}, for x >= a
}
```

**制約条件**:
1. **Token yield**: ∫f(x)dx = Z/X（目標トークン数/利用可能トークン数）
2. **Maximum upsampling**: 最上位バケットの平均 <= M（M=7に設定）
3. **Monotonicity**: λ >= 0（品質が高いほどアップサンプリング率も高い）

**実装詳細**:
- 各WebOrganizerトピックごとに個別のカーブを生成
- 最大アップサンプリング: M = 7
- 下位40%は破棄（a = 0.40）
- カーブの積分値 = そのトピックから抽出するトークン総数

```
例: 積分値2.0 → 平均2倍のアップサンプリング → 元の2倍のトークン数
```

---

## 4. WebOrganizerによるトピック分類

### 24トピックカテゴリ

| カテゴリ | カテゴリ |
|----------|----------|
| Adult Content | Art & Design |
| Crime & Law | Education & Jobs |
| Electronics & Hardware | Entertainment |
| Fashion & Beauty | Finance & Business |
| Food & Dining | Games |
| Health | History & Geography |
| Home & Hobbies | Industrial |
| Literature | Politics |
| Religion | Science, Math & Technology |
| Social Life | Software |
| Software Development | Sports & Fitness |
| Transportation | Travel |

### 実装

**オリジナル**: 140Mパラメータのtransformerモデル
**Olmo 3**: fastText分類器に蒸留（スケーラビリティ向上）

**学習データ**:
- WebOrganizerのLlama-labeledデータ
- 追加でgpt-4.1とo4-miniでラベル付けした506,746例

**性能（20,000文書holdout）**: Overall Precision/Recall = 0.762

---

## 5. Mixing と Upsampling の統合

データセットを2次元行列として捉える：
- **行**: WebOrganizerトピック
- **列**: 品質バケット

| 手法 | 対象 |
|------|------|
| Mixing | 行方向（トピック間）の分布を決定 |
| Quality Upsampling | 列方向（品質間）の分布を決定 |

### 統合方法の比較（1Bモデル、100Bトークン）

| 手法 | QA Easy | Math Easy | Code Easy |
|------|---------|-----------|-----------|
| Mixing Only | 1.005 | 0.778 | 0.872 |
| Quality Upsampling Only | 1.022 | 0.821 | 0.809 |
| Arithmetic Mean | 1.004 | 0.792 | 0.828 |
| Geometric Mean | 1.004 | 0.782 | 0.813 |
| Truncated exponential family | 1.002 | 0.782 | 0.787 |
| **Truncated power-exponential (Olmo 3)** | **0.993** | **0.758** | **0.783** |

切断べき指数関数族が最良の性能を示す。

---

## 6. よくある質問

### Q: なぜ30Mのproxyモデルで最適化できるのか？

**A:** Data Mixing Lawsの研究により、小型モデルでの混合比率の傾向が大型モデルに転移することが示されている。30Mモデルを3Bトークン（Chinchilla比5x）学習することで、コスト効率よく探索可能。ただし絶対的な性能は異なるため、最終的には大型モデルでの検証も行う。

### Q: Quality-aware upsamplingがflat filteringより優れる理由は？

**A:**
1. **多様性の維持**: 高品質データのみに絞ると、多様性が失われる
2. **高品質データの活用最大化**: 最上位データを複数回見せることで学習効果を高める
3. **トレードオフの最適化**: 品質と多様性のバランスを連続的に調整可能

### Q: Conditional mixingの計算量削減効果は？

**A:** 例えばN個のドメインがあり、M個が新規追加された場合：
- Naive: O(5N)のswarmサイズ
- Conditional: O(5M + 5)のswarmサイズ（既存N-Mドメインを1つの仮想ドメインに圧縮）

---

<details>
<summary><strong>他のデータミックス手法との比較: DoReMi, Manual Curation</strong></summary>

データミックスのアプローチは大きく3つに分類できる。

**1. Manual Curation（手動調整）**

**Llama 3のアプローチ:**
- 人間の直感とアブレーション実験に基づく
- ドメインごとのサンプリング比率を手動設定
- 知識蒸留とフィルタリングを組み合わせ

**利点**: 解釈可能、特定ドメインの重視が容易
**欠点**: スケーラビリティに欠ける、最適解の保証なし

**2. DoReMi（Xie et al., 2023）**

**原理**: Distributionally Robust Optimization
- proxy model のパープレキシティを最大化するようにミックスを調整
- 最悪ケースドメインの性能を改善

**数式**:
```
max_α min_domain E_x~D_domain[-log p(x; θ)]
```

**利点**: 弱いドメインを自動的に強化
**欠点**: 計算コストが高い、品質情報を活用しない

**3. Olmo 3 Constrained Data Mixing**

**DoReMiとの違い:**
- DoReMi: 最悪ケース最適化（min-max）
- Olmo 3: 平均BPB最小化 + 制約条件

**利点**:
- 品質に基づくアップサンプリングと組み合わせ可能
- Conditional mixingで計算効率化
- 480サブセット（24トピック×20品質）の細粒度制御

**比較表**

| 手法 | 自動化 | 品質考慮 | 計算コスト |
|------|-------|---------|----------|
| Manual | ✗ | △ | 低 |
| DoReMi | ✓ | ✗ | 高 |
| **Olmo 3** | ✓ | ✓ | 中 |

**参考**: [DoReMi (Xie et al., 2023)](https://arxiv.org/abs/2305.10429)

</details>

---

## 参考文献

- Chen et al. (2026): Olmixの条件付き混合手法
- Liu et al. (2024): RegMix
- Ye et al. (2025): Data Mixing Laws
- Diao et al. (2025): CLIMB
- Wettig et al. (2025): WebOrganizer
- Li et al. (2024): DCLM
