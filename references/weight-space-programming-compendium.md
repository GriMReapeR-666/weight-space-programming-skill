# Weight-Space Programming (WSP) & Model Surgery: Domain Knowledge Compendium

This reference documents the empirical physics, linear algebraic foundations, and safety boundaries of **Weight-Space Programming (WSP)** and **Autonomous Model Surgery**, synthesized from the MIT CSAIL WSP research (*Takehi, Teoh, Zhang, Khattab; Nov 2, 2026*), the ROME/MEMIT model editing collapse literature (*Gupta et al., EMNLP 2024*), and community replication benchmarks (*OpenWiSH, wsp-kit, OpenWSP*).

---

## 1. Mechanics of Weight-Space Programming (WSP)

### 1.1 The Paradigm Shift
Traditional machine learning treats neural network weights as a write-only black box managed by stochastic gradient descent and optimizer moments (AdamW, Muon, Lion). 
**Weight-Space Programming** treats model checkpoints as readable, interpretable, and executable source code artifacts:
$$\theta' = \theta + \Delta \theta$$
where $\Delta \theta$ is derived through **mechanistic diagnosis, SVD subspace projection, and targeted linear algebra** rather than backpropagation.

### 1.2 The MIT CSAIL Benchmark Findings (Takehi et al., 2026)
* **Qwen3.5-0.8B Pathology**:
  The model suffered from repetitive reasoning loops that caused **85 of 96 answers (93.0%)** to hit the 32,768-token limit without producing an answer.
* **The Diagnostic Chain**:
  1. *Behavioral Observation*: Recognition that the failure was not mathematical ignorance, but an amplification loop.
  2. *Layer Attribution*: Measuring directional push $g_\ell = u^\top (d_{\ell+1} - d_\ell)$ revealed that MLP blocks at **layers 15 and 19** dominated the amplification.
  3. *Neuron Isolation*: Neuron 11 in layer 19 was identified as the primary writer of the loop direction $u$.
  4. *Copying Head Interaction*: Head 7 in layer 15 and Head 5 in layer 19 formed a recursive attention copying circuit.
* **The Surgical Intervention**:
  Astra applied a rank-1 projection:
  $$W' = W - \alpha \cdot u (u^\top W)$$
  to **64 selected columns** in layers 15 and 19 (scaling factor $\alpha = 6$).
* **Measured Impact**:
  - **Weights Modified**: Exactly **131,072 parameters (0.016% of total model)**.
  - **MATH-500**: Jumped from **7.9% to 48.1%** (+40.2%).
  - **GSM8K**: Jumped from **13.8% to 67.1%**.
  - **Token Cap Truncation**: Dropped from **93.0% down to 33.1%**.
* **The Critical Caveat (Concentrated vs. Distributed Capabilities)**:
  On **Gemma 3 270M**, where mathematical errors were distributed across the entire residual stream rather than concentrated in an identifiable loop direction, weight editing produced $\le 1$ point gain. **Editing removes localized bottlenecks; it cannot synthesize distributed knowledge without external training or distillation.**

---

## 2. Multi-Layer Edit Gate: Preventing Catastrophic Collapse

A single unconstrained parameter update can destroy a model's representation manifold. The WSP runtime enforces five strict gate policies:

### 2.1 Scope Restrictions (Forbidden Tensors)
The following parameter tensors are permanently locked from weight-space mutation:
| Forbidden Tensor | Structural Vulnerability |
| :--- | :--- |
| `embed_tokens` / `wte` | Perturbing input embeddings shifts the coordinate frame for all downstream layers. |
| `lm_head` / `output.weight` | Modifying the vocabulary projection distorts next-token probability calibration globally. |
| `final_norm` / `norm.weight` | Normalizes the entire residual stream; scalar shifts cascade through all token logits. |

### 2.2 Whole-Tensor Scaling Prohibition
Edits representing uniform scalar multiplication ($W' = W \cdot c$) are strictly rejected. Global knobs cause outsized distribution shifts and disable general reasoning.

### 2.3 The 1% Fraction Cap
Edits touching $> 1.0\%$ of the model's total parameter count are rejected as "too broad." (The successful Qwen edit touched only $0.016\%$, 62× below this threshold).

### 2.4 Normalized $L_2$ Distance Gate (Gupta et al., EMNLP 2024)
Per-neuron Frobenius displacement must not exceed the disabling threshold:
$$\text{Dist}(W, W') = \frac{\|W' - W\|_F}{N} \le 1.0 \times 10^{-4}$$
* Safe, precise edits typically exhibit $\text{Dist} \approx 8.16 \times 10^{-7}$.
* Disabling edits that collapse the model exhibit $\text{Dist} \ge 3.34 \times 10^{-4}$ (400× larger).

### 2.5 Matched Gaussian Control Falsification
Every proposed delta $\Delta W_{\text{real}}$ must be benchmarked against an orthogonal Gaussian perturbation $\Delta W_{\text{control}}$ of identical Frobenius norm:
$$\|\Delta W_{\text{control}}\|_F = \|\Delta W_{\text{real}}\|_F$$
If the random noise control matches the behavioral improvement of the real delta, the causal hypothesis is flagged as **FALSIFIED** (spurious correlation) and discarded.

---

## 3. Selective Precision Engineering & Block Dequantization

### 3.1 GGUF Block-Level Quantization Topology
Quantized formats do not permit arbitrary independent scalar edits. In **GGUF Q4_0**, weights are stored in **32-element blocks**:
* **Bytes per block**: 18 bytes (2 bytes for FP16 scale $d$, 16 bytes for 32 packed 4-bit nibbles).
* **Nibble range**: $[0, 15]$, offset by $-8$ to represent $[-8, 7]$.
* **Reconstruction Formula**:
  $$\hat{w}_j = (\text{nibble}_j - 8) \cdot d$$

### 3.2 Surgical Block Modification Protocol
1. Calculate byte offset for target matrix coordinate $(r, c)$.
2. Extract the containing 18-byte block.
3. Decode block into 32 Float32 values.
4. Apply the targeted rank-1 or SVD modification in FP32.
5. Compute new scale factor: $d' = \max(|w'|) / 7$.
6. Quantize modified values back to 4-bit nibbles and repack 18 bytes.
7. Verify normalized reconstruction distortion:
   $$\text{Distortion} = \frac{\|w' - \hat{w}_{\text{new}}\|_F}{32} \le 0.20$$

---

## 4. Subspace-Constrained Knowledge Distillation

When weight-space edits alone are insufficient (distributed capabilities), WSP acts as the control layer for **Null-Space Constrained Distillation**:

$$\min_\theta \alpha \mathcal{L}_{\text{KD}}(P_{\text{teacher}}^T \parallel P_{\text{student}}^T) + \beta \mathcal{L}_{\text{CE}}(f_\theta(x), y)$$

$$\text{subject to } \quad \Delta W_{\text{safe}} = \Delta W \cdot (I - Q_{\text{anchor}} Q_{\text{anchor}}^\top)$$

* $Q_{\text{anchor}}$ is an orthonormal basis (via QR decomposition) of critical preserved task directions.
* Ensures parameter updates mathematically satisfy $\Delta W_{\text{safe}} \cdot V_{\text{anchor}} = 0$, guaranteeing **0% regression on baseline capabilities**.

---

## 5. Architectural Adapter Scaling Laws

Empirical observations across model architectures:
* **Deep layers are more tolerant**: Layers at depth $\ge 60\%$ encode abstract representations and tolerate editing with minimal collateral damage; shallow layers ($\le 25\%$) cascade across all residual connections.
* **Larger models need smaller fraction edits**: 
  - 17K policy: 3.5% edit
  - 0.8B Qwen: 0.016% edit
  - 4B Qwen: 0.00012% edit (2 columns in 1 layer)
* **Architecture Fragility Hierarchy**:
  1. *Most Tolerant*: GPT-J (RoPE + GQA provide redundant attention pathways).
  2. *Moderate*: GPT2-XL / LFM (standard MHA).
  3. *Most Fragile*: LLaMA-2 / ModernBERT (SwiGLU + RMSNorm make the residual stream sensitive to scalar drift).
