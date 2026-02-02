# 長文脈拡張

本文書では、Olmo 3の長文脈拡張について詳細に解説する。Olmo 3は8,192トークンから65,536トークンへの文脈長拡張を行っている。

---

## 1. RoPE（Rotary Position Embedding）の基本

### 1.1 概要

RoPE（Rotary Position Embedding）は、Transformerの位置エンコーディング手法の一つである。従来の絶対位置エンコーディングや相対位置エンコーディングと異なり、回転行列を用いてトークンの位置情報を埋め込む。

### 1.2 数学的原理

RoPEは、query/keyベクトルを2次元ずつのペアに分割し、各ペアに対して位置 $m$ に応じた回転を適用する：

$$
\text{RoPE}(x_m, m) = R_{\theta,m} x_m
$$

ここで回転行列 $R_{\theta,m}$ は以下のように定義される：

$$
R_{\theta,m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

周波数 $\theta_i$ は次元 $i$ に対して以下で計算される：

$$
\theta_i = \theta_{\text{base}}^{-2i/d}
$$

Olmo 3では $\theta_{\text{base}} = 5 \times 10^5$ を使用している。

### 1.3 RoPEの利点

1. **相対位置エンコーディングの実現**: 2つのトークン間のattention scoreは、その相対位置のみに依存する
2. **外挿性**: 理論上、訓練時よりも長い系列に対して一定の汎化が期待できる
3. **効率性**: 事前計算とキャッシュが可能で、計算コストが低い

---

## 2. YaRN（Yet another RoPE extensioN）の仕組み

### 2.1 RoPE拡張手法の比較

Olmo 3では、RoPEを訓練時の文脈長を超えて拡張するために複数の手法を実験した：

| 手法 | 説明 |
|------|------|
| 基底周波数スケーリング | $\theta_{\text{base}}$ を増加（例: $8 \times 10^6$） |
| Position Interpolation | 位置インデックスをスケールダウン |
| **YaRN** | 周波数帯域ごとに異なるスケーリングを適用 |

<details>
<summary><strong>他のRoPE拡張手法: PI, NTK-aware, Longformer</strong></summary>

RoPEの位置外挿問題に対する主要なアプローチを比較する。

**Position Interpolation (PI)**

**原理**: 位置インデックスを線形スケールダウン

```
m' = m / s  (s = target_length / train_length)
```

**利点**: シンプルで実装容易
**欠点**: 局所的なパターン認識が劣化（高周波も圧縮される）

**NTK-aware Scaling**

**原理**: $\theta_{base}$ を増加させて全周波数を均等にスケール

```
θ'_base = θ_base * s^(d/(d-2))
```

**利点**: 局所パターンへの影響が少ない
**欠点**: 長距離の位置関係が不正確

**YaRN（Olmo 3採用）**

**原理**: 周波数帯域ごとに異なる戦略を適用

```
低周波 → PI（長距離の位置関係を保持）
高周波 → そのまま（局所パターンを維持）
中間  → 両者を補間
```

**利点**: 両方の利点を兼ね備える
**欠点**: ハイパーパラメータ（帯域境界）の調整が必要

**比較表**

| 手法 | 局所パターン | 長距離関係 | 学習トークン | 使用モデル |
|------|------------|----------|-------------|-----------|
| PI | △ | ✓ | 少 | Llama 2 Long |
| NTK-aware | ✓ | △ | 中 | Code Llama |
| **YaRN** | ✓ | ✓ | 中 | Olmo 3, Qwen 2 |
| ALiBi | - | ✓ | 少 | MPT, BLOOM |

**注**: ALiBiはRoPEではなく線形バイアスによるアプローチ

**参考**:
- [Position Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595)
- [NTK-aware (bloc97, 2023)](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)

</details>

### 2.2 YaRNの原理

YaRN（Yet another RoPE extensioN）は、RoPEの周波数スペクトルを3つの帯域に分割し、それぞれに異なる処理を適用する：

1. **低周波帯域**: 位置補間を適用（長距離の位置関係を保持）
2. **高周波帯域**: 外挿を許容（局所的なパターンは変更不要）
3. **中間帯域**: 低周波と高周波の間でスムーズに補間

この方法により、短い文脈で獲得した局所的なパターン認識能力を保持しつつ、長距離の位置関係を適切にスケーリングできる。

### 2.3 YaRNの適用結果

Olmo 3の実験では、YaRNが他の手法と比較して最も良い性能を示した（RULER benchmarkで評価）。特に32K〜65Kの長い文脈長でその優位性が顕著であった。

---

## 3. Full Attention層のみへのYaRN適用

### 3.1 Olmo 3のアーキテクチャ

Olmo 3は、全32/64層のうち3/4の層でSliding Window Attention（SWA）を使用し、残り1/4の層でFull Attentionを使用するハイブリッド構造を採用している：

| モデル | 総層数 | Full Attention層 | SWA層 | SWAウィンドウサイズ |
|--------|--------|------------------|-------|---------------------|
| 7B | 32 | 8層 | 24層 | 4,096トークン |
| 32B | 64 | 16層 | 48層 | 4,096トークン |

### 3.2 YaRNをFull Attention層のみに適用する理由

**Sliding Window Attention層にYaRNを適用しない理由：**

1. **SWAは固定長ウィンドウで動作**: 4,096トークンのウィンドウ内では、位置エンコーディングの範囲が訓練時と同じ
2. **位置の外挿が発生しない**: SWAでは常にウィンドウサイズ以内の相対位置しか扱わない
3. **不必要な変更を避ける**: SWAの学習済みパターンを維持することで性能劣化を防ぐ

**Full Attention層にYaRNを適用する理由：**

1. **長距離依存関係を担当**: Full Attention層は系列全体を見るため、位置の外挿が必要
2. **8K→65Kへの拡張**: 8倍の文脈長拡張に対応する位置スケーリングが必須
3. **グローバルな情報統合**: 文書全体の情報を集約するために長距離の位置関係が重要

### 3.3 実験結果

Figure 13aの結果より、「YaRN on full attention layers only」が最も高いRULERスコアを達成。他の設定（全層への適用、基底周波数スケーリング等）よりも一貫して優れた性能を示した。

---

## 4. Context Parallelism（8-way）

### 4.1 概要

65,536トークンの長系列を効率的に訓練するため、Olmo 3では8-way Context Parallelism（CP）を採用している。

### 4.2 動作原理

Context Parallelismは、入力系列を複数のデバイスに分割して並列処理する手法：

```
65,536 tokens ÷ 8 devices = 8,192 tokens/device
```

各デバイスは8Kトークンを担当し、attention計算時に必要な情報を他デバイスと交換する。

### 4.3 All-gather ベースのCP Attention戦略

Olmo 3は、Chu et al. (2025) によるLlama3スタイルのContext Parallelismを採用：

1. **All-gather通信**: 各デバイスがKV（Key-Value）を全デバイスに送信
2. **ローカルAttention計算**: 自分のQueryと全デバイスのKVでattentionを計算
3. **不規則なAttention Maskのサポート**: Sliding Window AttentionやIntra-document Maskingとの互換性を維持

### 4.4 訓練構成

| モデル | デバイス数 | DP-rep | DP-shard | CP | スループット |
|--------|------------|--------|----------|-----|-------------|
| 7B | 256 | 32 | - | 8 | 4.0K TPS/device |
| 32B | 1024 | 16 | 8 | 8 | 1.3K TPS/device |

注: DP-rep = Data Parallel Replication、DP-shard = HSDP Sharding

---

## 5. 合成データ拡張

### 5.1 動機

長文脈理解の典型的なユースケースは、長い入力からの情報抽出と統合である。しかし、自然に存在する長文書のほとんどは、このようなタスクの教師信号を持たない。この問題を解決するため、Olmo 3では合成データを生成して長文脈訓練を強化している。

### 5.2 CWE（Common Word Extraction）

**目的**: 文書内の単語出現回数をカウントする能力を訓練

**生成パイプライン**:
1. 文書を8K〜32Kトークンのセクションに分割
2. 各セクションでtf-idfを計算し、頻出する1語の名詞句を5つ抽出
3. OLMo 2 Instruct 32Bに対して、各単語の正確な出現回数を問うQAペアを生成させる

**例**:
```
Q: この文書内で「machine learning」という単語は何回出現しますか？
A: 正確に23回出現しています。
```

### 5.3 REX（Rewriting EXpressions）

**目的**: 文書全体の情報を統合して再表現する能力を訓練

**生成パイプライン**:
1. 名詞句とその出現箇所のスニペット（k=8）を抽出
2. 12種類のビニェット形式のいずれかで情報を再構成するタスクを生成

**12種類のビニェット形式**:
1. 短い要約
2. 教授と学生の対話
3. 高校生向けの簡潔な説明
4. フラッシュカード
5. 学校のクイズ
6. ゲームショー
7. ディナーパーティーでの会話
8. ディベート
9. 真偽判定問題
10. 映画シーン
11. 百科事典的な説明
12. r/explainlikeimfive風の説明

### 5.4 合成データの効果

Figure 13cの結果より：
- Natural PDFs + Synth CWE/REX > Natural PDFs + Synth CWE > Natural PDFs
- 特に32K〜65Kの長い文脈でREXの効果が顕著

---

## 6. Best-fit Packing と Intra-document Masking

### 6.1 従来の問題点

標準的なpretraining/midtrainingでは、文書を連結して固定長に分割する（concatenate-then-split）。しかし、長文脈拡張時にこの方法を使うと：

- 実際の訓練インスタンスの長さが、基となる文書長分布より短くなる
- 長文書の途中で切断され、文書全体を見る訓練ができない

### 6.2 Best-fit Packing

**原理**: Ding et al. (2024) によるbest-fit document packingアルゴリズムを採用

1. 文書を長さでソート
2. 各訓練インスタンス（65,536トークン）に対して、最も効率的に詰められる文書の組み合わせを選択
3. パディングを最小限に抑えつつ、文書の分割を減少

**効果**:
- 分割される文書数の大幅な削減
- 長文脈ベンチマークでの性能向上（Figure 13d）
- 特に長い文脈長（32K〜65K）での改善が顕著

### 6.3 Intra-document Masking

**定義**: 各訓練シーケンスが、同一の基底文書から来たトークンのみにattendするようにマスキング

**目的**:
1. **Cross-document Signal の防止**: 異なる文書間の偽のattentionパターンを排除
2. **長距離依存関係の学習促進**: 同一文書内での長距離依存関係に集中
3. **性能劣化の防止**: cross-document信号はspuriousなattentionパターンを導入し、long-range性能を低下させる

**実装上の考慮**:
- 不規則なattention maskのサポートが必要
- Context Parallelismとの互換性を確保（All-gather方式が対応）
- Sliding Window Attentionとの組み合わせも可能

---

## 7. 50-100Bトークンが必要な理由

### 7.1 オープンモデルのレシピ比較

| モデル | Long-context拡張トークン数 | 目標文脈長 |
|--------|---------------------------|-----------|
| SmolLM3 | 100B | 128K |
| GLM 4.5 | 100B | 128K |
| DeepSeek V3 | 123B | 128K |
| Llama 3.1 | 800B | 128K |
| **Olmo 3 7B** | **50B** | **65K** |
| **Olmo 3 32B** | **100B** | **65K** |

### 7.2 なぜ多くのトークンが必要か

**Figure 13eの実験結果より**:
- 1B拡張 → 5B → 10B → 25B → 50B → 100B と増加するにつれ、RULERスコアが向上
- 特に65Kの長い文脈長での改善が顕著
- 短い拡張（1B〜5B）では、長文脈での性能が著しく低い

**理由の考察**:
1. **位置エンコーディングの適応**: 8K→65Kの8倍拡張には、十分なトークン数での訓練が必要
2. **長距離依存関係の学習**: 稀な長距離パターンを十分に観察するためのデータ量
3. **短文脈性能の維持**: 長文脈データと短文脈データのミックス（34%:66%）で両方の性能を維持
4. **文書多様性**: 多様な長文書を十分に見るためのトークン数

### 7.3 効率的な拡張の可能性

一部のモデル（AFM、Nemotron Nano 2）は20B未満で64K〜128Kへの拡張に成功している。これは：
- 高品質なデータ選択
- 適切なアーキテクチャ選択
- 最適化されたハイパーパラメータ

により、トークン効率を向上できる可能性を示唆している。

---

## 8. まとめ

Olmo 3の長文脈拡張レシピの5つの重要な要素：

1. **YaRN on Full Attention Only**: SWA層は変更せず、Full Attention層のみにYaRNを適用
2. **高品質データ（olmOCR Science PDFs）**: 22M以上の科学論文PDFを処理
3. **合成データ拡張（CWE/REX）**: 長文脈理解タスクの教師信号を合成生成
4. **Best-fit Packing + Intra-document Masking**: 効率的なバッチ構成と正確なattention制御
5. **十分なトークン予算（50B/100B）**: 長文脈での性能を確保

これらの要素により、Olmo 3は65Kトークンの文脈長をサポートし、RULERおよびHELMETベンチマークで競争力のある性能を達成している。

---

## 参考文献

- Olmo 3 Technical Report (arXiv:2512.13961)
- Su et al. (2024) - RoPE
- Peng et al. (2023) - YaRN
- Chu et al. (2025) - Llama3-style Context Parallelism
- Ding et al. (2024) - Best-fit Document Packing
- Pham et al. (2025) - CLIPPER
