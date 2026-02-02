# Sliding Window Attention

## 概要

Sliding Window Attention (SWA) は、Transformer の Self-Attention の計算効率を向上させるための手法である。Olmo 3 では、長いコンテキスト長での学習と推論を効率化するために採用されている。

---

## 1. SWA の仕組み

### 1.1 Full Attention vs. Sliding Window Attention

**Full Attention（従来の Self-Attention）**

各トークンがシーケンス内の全てのトークンに対してアテンションを計算する。

```
入力シーケンス: [T1, T2, T3, T4, T5, T6, T7, T8]

T8 が参照できるトークン: T1, T2, T3, T4, T5, T6, T7
                        ↑ 全トークンを参照
```

**Sliding Window Attention**

各トークンが固定サイズの「ウィンドウ」内のトークンのみに対してアテンションを計算する。

```
ウィンドウサイズ w = 4 の場合

T8 が参照できるトークン: T5, T6, T7 (直前の4トークン)
                        ↑ ウィンドウ内のみ参照
```

### 1.2 図解：アテンションパターンの比較

```
【Full Attention マスク】        【Sliding Window Attention マスク (w=4)】
     T1 T2 T3 T4 T5 T6 T7 T8         T1 T2 T3 T4 T5 T6 T7 T8
T1 [ *  -  -  -  -  -  -  - ]    T1 [ *  -  -  -  -  -  -  - ]
T2 [ *  *  -  -  -  -  -  - ]    T2 [ *  *  -  -  -  -  -  - ]
T3 [ *  *  *  -  -  -  -  - ]    T3 [ *  *  *  -  -  -  -  - ]
T4 [ *  *  *  *  -  -  -  - ]    T4 [ *  *  *  *  -  -  -  - ]
T5 [ *  *  *  *  *  -  -  - ]    T5 [ -  *  *  *  *  -  -  - ]
T6 [ *  *  *  *  *  *  -  - ]    T6 [ -  -  *  *  *  *  -  - ]
T7 [ *  *  *  *  *  *  *  - ]    T7 [ -  -  -  *  *  *  *  - ]
T8 [ *  *  *  *  *  *  *  * ]    T8 [ -  -  -  -  *  *  *  * ]

* = アテンション可能, - = マスク（アテンション不可）

Full Attention: 下三角行列（全履歴を参照）
SWA: 斜めの帯状パターン（ウィンドウ内のみ参照）
```

### 1.3 多層での情報伝播

SWA 層を積み重ねることで、間接的に長距離の情報にアクセス可能になる。

```
層1 (SWA, w=4): T8 → T5,T6,T7 を参照
層2 (SWA, w=4): T8 → T5,T6,T7 を参照（各トークンは層1でさらに4トークン前を参照済み）
                   → 間接的に T1 までの情報がT8に伝播

有効受容野 = ウィンドウサイズ × 層数
```

---

## 2. 計算量・メモリ使用量の比較

### 2.1 計算量

| アテンション方式 | 時間計算量 | n=8192, w=4096 での比較 |
|-----------------|-----------|------------------------|
| Full Attention | O(n^2) | n^2 = 67,108,864 |
| Sliding Window | O(n * w) | n * w = 33,554,432 |

**シーケンス長 n に対する計算量の変化:**

```
n       Full Attention    SWA (w=4096)    削減率
4,096   16,777,216        16,777,216      0%（ウィンドウ = シーケンス長）
8,192   67,108,864        33,554,432      50%
16,384  268,435,456       67,108,864      75%
32,768  1,073,741,824     134,217,728     87.5%
65,536  4,294,967,296     268,435,456     93.75%
```

**Olmo 3 の場合:**
- コンテキストウィンドウ: 8,192トークン（事前学習・中間学習）
- SWA ウィンドウサイズ: 4,096トークン
- 計算量削減: 50%

### 2.2 メモリ使用量

Self-Attention では、アテンションスコア行列を格納する必要がある。

| アテンション方式 | メモリ使用量（アテンションスコア） |
|-----------------|-------------------------------|
| Full Attention | O(n^2) |
| Sliding Window | O(n * w) |

**KVキャッシュ（推論時）:**

```
Full Attention KV キャッシュ:
- 各層で全トークンの K, V を保持
- メモリ: O(n * d_head * n_heads * 2 * L)

SWA KV キャッシュ:
- 各 SWA 層でウィンドウサイズ分の K, V のみ保持
- メモリ: O(w * d_head * n_heads * 2 * L_swa)
```

### 2.3 スループット

Olmo 3 での実測値（Olmo 3 論文より）:

| モデル | スループット (tokens/sec/GPU) | シーケンス長 |
|--------|----------------------------|-------------|
| 7B | 7,700 | 8,192 |
| 32B | 1,960 | 8,192 |

SWA の採用により、長いシーケンス長でも効率的な学習が可能になっている。

---

## 3. Olmo 3 での具体的な適用パターン

### 3.1 レイヤー構成

Olmo 3 では、**4層中3層が SWA、1層が Full Attention** というパターンを採用している。

```
【Olmo 3 7B (32層)】           【Olmo 3 32B (64層)】

Layer  0: SWA                  Layer  0: SWA
Layer  1: SWA                  Layer  1: SWA
Layer  2: SWA                  Layer  2: SWA
Layer  3: Full                 Layer  3: Full
Layer  4: SWA                  Layer  4: SWA
...                            ...
Layer 28: SWA                  Layer 60: SWA
Layer 29: SWA                  Layer 61: SWA
Layer 30: SWA                  Layer 62: SWA
Layer 31: Full ← 最終層        Layer 63: Full ← 最終層

SWA層: 24層 (75%)              SWA層: 48層 (75%)
Full層: 8層 (25%)              Full層: 16層 (25%)
```

### 3.2 アーキテクチャパラメータ（Table 33 より）

| パラメータ | Olmo 3 7B | Olmo 3 32B |
|-----------|----------|-----------|
| 層数 | 32 | 64 |
| Hidden size | 4,096 | 5,120 |
| Q heads | 32 | 40 |
| KV heads | 32 (MHA) | 8 (GQA) |
| SWA | 3/4 layers; w=4,096 | 3/4 layers; w=4,096 |
| RoPE scaling | YaRN (Full層のみ) | YaRN (Full層のみ) |
| RoPE base theta | 5 * 10^5 | 5 * 10^5 |

### 3.3 SWA 層の配置ルール

```python
# 擬似コード: Olmo 3 のアテンションパターン決定
def get_attention_type(layer_idx, total_layers):
    """
    4層に1回 Full Attention を使用
    最終層は必ず Full Attention
    """
    if layer_idx == total_layers - 1:  # 最終層
        return "full"
    elif (layer_idx + 1) % 4 == 0:     # 4の倍数番目
        return "full"
    else:
        return "sliding_window"
```

---

## 4. 長文脈対応との関係

### 4.1 SWA と長文脈拡張の相互作用

Olmo 3 は以下の3段階で訓練される:

1. **事前学習**: 8,192トークン
2. **中間学習**: 8,192トークン
3. **長文脈拡張**: 最大65,536トークン（7B: 50B tokens, 32B: 100B tokens）

**SWA の長文脈への貢献:**

```
事前学習・中間学習時:
- コンテキスト長: 8,192
- SWA ウィンドウ: 4,096
- 計算量削減: 50%

長文脈拡張時 (例: 65,536トークン):
- Full Attention のみの場合: O(65536^2) = 4.3 billion operations
- SWA (w=4,096) の場合: O(65536 * 4096) = 268 million operations
- 計算量削減: 93.75%
```

### 4.2 YaRN と SWA の組み合わせ

長文脈拡張時の位置エンコーディング調整:

```
【RoPE 拡張戦略 (Olmo 3)】

Full Attention 層:
  → YaRN を適用（位置補間で長距離を扱えるように）

SWA 層:
  → YaRN を適用しない（ウィンドウ内は元の位置関係を維持）
```

**この設計の理由:**
- SWA 層は常にウィンドウサイズ（4,096）以内のトークンのみを参照
- ウィンドウ内の位置関係は訓練時と同じなので調整不要
- Full Attention 層でのみ長距離位置を扱うため、そこに YaRN を適用

Olmo 3 論文より:
> "We find that applying YaRN only to full attention layers yields the best overall performance."

### 4.3 長文脈ベンチマーク結果

| モデル | RULER (4K) | RULER (8K) | RULER (16K) | RULER (32K) | RULER (65K) |
|--------|-----------|-----------|------------|------------|------------|
| Olmo 3 32B | 95.7 | 93.9 | 92.9 | 90.1 | 81.7 |
| Qwen 2.5 32B | 96.3 | 95.7 | 94.1 | 89.1 | - |
| Gemma 3 27B | 97.6 | 95.7 | 91.9 | 91.6 | - |

---

## 5. なぜ最後の層は Full Attention なのか

### 5.1 情報集約の必要性

言語モデルの最終層は、次トークン予測のために全体の文脈情報を集約する重要な役割を持つ。

```
【情報フローの観点】

入力層
  ↓
SWA層 × 複数 : 局所的な特徴を抽出・変換
  ↓
Full層 : 中間集約点として全体情報を統合
  ↓
SWA層 × 複数 : さらに特徴を洗練
  ↓
...
  ↓
最終 Full層 : 全コンテキストを統合して予測に必要な表現を生成
  ↓
出力（Language Model Head）
```

### 5.2 理論的背景

**受容野（Receptive Field）の観点:**

```
L層の SWA を通過後の有効受容野:
  有効受容野 = min(w * L, n)

例: w=4096, 12 SWA層連続
  12 * 4096 = 49,152 トークン

→ 65Kコンテキストでは全てをカバーできない
→ 最終層で Full Attention が必要
```

**情報ボトルネックの回避:**

```
全て SWA の場合:
- 遠距離のトークン間で直接的なアテンションが一度もない
- 情報が中間表現を介して伝播するため、情報劣化の可能性

定期的な Full Attention:
- 直接的な長距離依存関係を学習可能
- 情報ボトルネックを緩和
```

### 5.3 実験的知見

Olmo 3 の設計では、以下の知見に基づいている:

1. **最終層を Full Attention にする理由**:
   - 出力直前で全体の文脈情報にアクセスできることを保証
   - 特に質問応答や要約タスクで、入力の冒頭にある重要情報を参照する必要がある

2. **4層に1回 Full Attention にする理由**:
   - 計算コストと性能のバランス
   - 25% の Full 層でも十分な長距離モデリング能力を維持
   - Longformer (Beltagy et al., 2020) の知見を継承

### 5.4 関連研究との比較

| モデル | SWA 適用 | Full Attention 配置 | 備考 |
|--------|---------|-------------------|------|
| Longformer | 全層 SWA + Global | タスク固有トークンに Global | 特定トークンが全体を参照 |
| Mistral 7B | SWA | 全層 SWA | w=4,096 |
| Gemma 2 | 3/4 層 SWA | ハイブリッド | Olmo 3 と類似 |
| Olmo 3 | 3/4 層 SWA | 4層に1回 + 最終層 | ハイブリッド方式 |

<details>
<summary><strong>SWA を採用しないモデル（Qwen3, DeepSeek）</strong></summary>

全てのモデルが SWA を採用しているわけではない。Qwen3 や DeepSeek は計算効率化に別のアプローチを取っている。

**Qwen3: NoPE（No Positional Encoding）**

Qwen3 では **3/4 の層で位置エンコーディングを完全に省略**する NoPE を採用。

```
【Qwen3 のレイヤー構成】
Layer 0: RoPE あり
Layer 1: NoPE（位置エンコなし）
Layer 2: NoPE
Layer 3: NoPE
Layer 4: RoPE あり
...
```

- **利点**: 長さの汎化（length generalization）に効果的
- **理由**: 位置エンコーディングがないため、学習時より長いシーケンスでも位置外挿の問題が発生しにくい

**DeepSeek V3: MLA（Multi-head Latent Attention）**

DeepSeek V3 では **MLA（Multi-head Latent Attention）**を採用。GQA のように K/V ヘッドを共有するのではなく、K/V を低次元空間に圧縮して保存する。

```
【アテンション方式の比較】

MHA (Multi-Head Attention):
  KV キャッシュ = n_heads × head_dim × seq_len × 2

GQA (Grouped Query Attention):
  KV キャッシュ = n_kv_heads × head_dim × seq_len × 2
  （n_kv_heads < n_heads）

MLA (Multi-head Latent Attention):
  KV キャッシュ = latent_dim × seq_len
  （latent_dim << n_heads × head_dim）
```

- **Ablation study の結果**: MLA > GQA > MHA（性能順）
- **利点**: KV キャッシュのメモリ使用量を大幅に削減しつつ、性能を維持・向上

**まとめ：効率化アプローチの比較**

| モデル | 効率化手法 | 対象 |
|--------|----------|------|
| Olmo 3, Mistral, Gemma | SWA | アテンション計算量 |
| Qwen3 | NoPE | 位置エンコーディング |
| DeepSeek V3 | MLA | KV キャッシュ圧縮 |

これらは排他的ではなく、組み合わせることも可能。モデル設計においては、ターゲットとするユースケース（長文脈、推論速度、メモリ効率）に応じて選択される。

**参考文献:**
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- [From DeepSeek V3 to V3.2: Architecture](https://sebastianraschka.com/blog/2025/technical-deepseek.html)
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)

</details>

---

## 6. よくある質問

### Q: SWA とは何ですか？なぜ使うのですか？

**A:**
SWA は、各トークンが参照できる範囲を固定サイズのウィンドウに制限するアテンション機構です。Full Attention が O(n^2) の計算量を要するのに対し、SWA は O(n*w) で済みます。Olmo 3 では 8Kコンテキストでウィンドウサイズ4Kを使い、計算量を50%削減しながら、複数層を重ねることで間接的に長距離の情報伝播を可能にしています。

### Q: なぜ全ての層を SWA にしないのですか？

**A:**
全層 SWA だと、シーケンスの両端にあるトークン間で直接アテンションが発生せず、情報が多くの中間層を経由して伝わることになります。これは情報劣化やボトルネックの原因となります。Olmo 3 では 4層に1回 Full Attention を配置し、最終層は必ず Full Attention にすることで、効率性と長距離依存関係のモデリング能力を両立しています。

### Q: 長文脈拡張時の YaRN の適用について説明してください

**A:**
Olmo 3 では YaRN を Full Attention 層のみに適用し、SWA 層には適用しません。SWA 層は常にウィンドウサイズ（4,096トークン）以内を参照するため、位置エンコーディングの外挿が不要です。一方、Full Attention 層では長距離の位置関係を扱う必要があるため、YaRN による位置補間が有効です。この選択的な適用により、長文脈ベンチマーク RULER で最良の性能を達成しています。

---

## 参考文献

- Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150
- Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). YaRN: Efficient Context Window Extension of Large Language Models. arXiv:2309.00071
- Olmo Team (2025). Olmo 3. arXiv:2512.13961
