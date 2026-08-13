# BARGE_for_review

> **Paper**: *Bridging the Structural Gap: Adapting Autoregressive Generation for Recommendation*
>
> This document is the supplementary material accompanying our submission.
> Due to licensing constraints the source code cannot be released
> at this time; instead, this file provides
> (i) the description of every module
> introduced by BARGE, at a level of detail sufficient for a competent
> practitioner to re-implement the method from scratch; and
> (ii) additional experiments that could not fit into the main paper.

---

## Contents

- [Part I. Implementation Details](#part-i-implementation-details)
  - [I.1  Overview and notation](#i1--overview-and-notation)
  - [I.2  Multi-view RQ-VAE tokenizer (DPD tokenizer)](#i2--multi-view-rq-vae-tokenizer-dpd-tokenizer)
  - [I.3  Item Context-Aware Attention (ICA)](#i3--item-context-aware-attention-ica)
  - [I.4  Dual-Path Decoding with OR-fusion (DPD)](#i4--dual-path-decoding-with-or-fusion-dpd)
  - [I.5  Hierarchical Path Reranker (HPR)](#i5--hierarchical-path-reranker-hpr)
  - [I.6  Training objective and two-stage pipeline](#i6--training-objective-and-two-stage-pipeline)
  - [I.7  Inference procedure](#i7--inference-procedure)
  - [I.8  Hyper-parameters used in the paper](#i8--hyper-parameters-used-in-the-paper)
- [Part II. Additional Experiments](#part-ii-additional-experiments)
  - [II.A  Additional ablation studies](#iia--additional-ablation-studies)
  - [II.B  Codebook collision rate](#iib--codebook-collision-rate)
  - [II.C  Catalog size scaling](#iic--catalog-size-scaling)
  - [II.D  Domain shift](#iid--domain-shift)
---

## Part I. Implementation Details

### I.1  Overview and notation

**Data flow.** Given a user history $u = (v_1, \ldots, v_T)$, every item
$v_t$ is turned by a frozen multi-view RQ-VAE into an $L \times H$ block
of discrete codes. Flattened per item, the model consumes a sequence of
length $T \cdot L \cdot H$ (interleaved per layer:
`[l0_h0, l0_h1, l1_h0, l1_h1, ...]`). An encoder-decoder Transformer then
autoregressively produces the $L \times H$ codes of the next item, one
$H$-tuple per RQ layer.

**Symbols used throughout.**

| Symbol | Meaning | Typical value |
|:------:|---------|:-------------:|
| $B$ | batch size | 256 |
| $T$ | history length (items) | 20 |
| $L$ | number of RQ layers | 4 |
| $H$ | number of views (heads) | 2 |
| $D$ | token embedding dim | 128 |
| $D_z$ | RQ-VAE latent dim | 32 |
| $D_\text{path}$ | HPR projection dim | 128 |
| $C_\ell$ | codebook size at RQ layer $\ell$ | $[512,256,128,64]$ |

**Modules added by BARGE (all live on top of a TIGER-style backbone).**

| Name | Where it plugs in | Purpose |
|------|-------------------|---------|
| `MvRqVae` | Tokenizer | Turn one embedding into $H$ complementary code streams. |
| `_compute_ica_context` | Encoder input, **before** position embeddings | Item-level context gating on the fused sub-token stream. |
| `n_heads` decoder towers | Per-view decoder | Independent per-view AR decoding sharing only encoder memory. |
| `HierarchicalPathReranker` | Per decoder tower, per layer $\ell \ge 1$ | Path-level dual-tower InfoNCE reranker. |
| `_fuse_per_view_scores` | Post beam-search | Fusion across $H$ views. |

### I.2  Multi-view RQ-VAE tokenizer (DPD tokenizer)

Implemented by `MvRqVae` and `MultiHeadQuantize`.

**Shapes.** Given $x \in \mathbb{R}^{B \times D_\text{text}}$:
- `encoder(x)` → $z \in \mathbb{R}^{B \times D_z}$ ($D_z=32$).
- optional learned rotation
  $z \leftarrow \text{rotation}(z)$ where `rotation` is an
  `nn.Linear(D_z, D_z, bias=False)` wrapped in
  `nn.utils.parametrizations.orthogonal(...)`, initialised to identity
  so training starts equivalent to no rotation.
- `torch.chunk(z, H, dim=-1)` → $H$ tensors of shape
  $(B, D_z/H)$ = disjoint subspaces (no auxiliary orthogonality loss
  needed).

**Per-layer quantization.** Each RQ layer owns $H$ independent codebooks
`self.embeddings[h] = nn.Embedding(C_\ell, D_z / H)`. Given the current
residual `res` (initially $z$), each layer runs:

```
for h in 0..H-1:
    dist_h    ← cosine(res_chunk_h, codebook_h)              # (B, C_ℓ)
    ids_h     ← argmin(dist_h, dim=-1)                        # (B,)
    emb_out_h ← quantize_forward(res_chunk_h, ids_h)          # (B, D_z/H)
        # forward mode: GUMBEL_SOFTMAX (default) | STE | ROTATION_TRICK
emb  ← concat(emb_out_0, ..., emb_out_{H-1}, dim=-1)          # (B, D_z)
ids  ← stack(ids_0, ..., ids_{H-1}, dim=1)                    # (B, H)
res  ← res - emb                                              # residual update
```

The Gumbel-softmax path is used during training. The decoder receives $\sum_\ell \hat r_\ell$ (with inverse rotation) and is trained with the standard MSE + commitment loss.

**Output structure.** After Stage-1 training, each item is stored as
`sem_ids_structured ∈ ℤ^{B × L × H}` and `sem_ids ∈ ℤ^{B × L·H}`.

### I.3  Item Context-Aware Attention (ICA)

Implemented in `_compute_ica_context`.

**Placement.** Inside `_prepare_encoder_input`, ICA is applied to the
**already multi-view fused** sub-token stream, **before** adding
position embeddings. It only exists on the encoder side.

**Learnable parameters.**

| Name | Shape |
|------|-------|
| `ica_query` | $(1, 1, D)$ |
| `ica_attn` | `nn.MultiheadAttention(embed_dim=D, num_heads=max(1, D//32), batch_first=True)` |
| `ica_norm` | `LayerNorm(D)` |
| `ica_proj` | `Linear(D, D) → GELU → Linear(D, D)` |
| `ica_gate` | `Linear(2·D, D) → Sigmoid` |

**Call chain.** Input `seq_emb ∈ (B, T·L, D)` after multi-view fusion,
`seq_mask ∈ (B, T·L)`:

```
# (1) group per item
item_tokens    ← seq_emb.view(B, T, L, D)
kv_tokens_flat ← item_tokens.reshape(B*T, L, D)
kv_mask_flat   ← seq_mask.view(B*T, L)          # True = valid

# (2) 1-token cross-attention against each item's L sub-tokens
q            ← ica_query.expand(B*T, 1, D)
ctx, _       ← ica_attn(q, kv_tokens_flat, kv_tokens_flat,
                        key_padding_mask=~kv_mask_flat)   # (B*T, 1, D)
ctx          ← ica_norm(ctx.squeeze(1))                    # (B*T, D)
# all-padded rows are set to 0 to prevent NaN propagation

# (3) project + broadcast back to every sub-token position
ctx_proj     ← ica_proj(ctx.view(B, T, D))                 # (B, T, D)
ctx_flat     ← ctx_proj.unsqueeze(2).expand(-1, -1, L, -1).reshape(B, T*L, D)

# (4) sigmoid-gated residual add
gate         ← ica_gate(concat([seq_emb, ctx_flat], dim=-1))  # (B, T·L, D)
ica_delta    ← gate * ctx_flat                                # returned
src          ← seq_emb + ica_delta                            # done in caller
```

**Notes.**
- Cost is $O(B \cdot T \cdot L \cdot D)$ (KV length is $L$, typically 4),
  strictly linear in sequence length.
- The gate is fed by both the original sub-token and the projected
  item-context, letting the model down-weight ICA when the item is
  already unambiguous.
- ICA has no dedicated loss; it is trained purely by the downstream NTP
  gradient.

### I.4  Dual-Path Decoding with OR-fusion (DPD)

**Two decoder towers** (`self.dual_decoder=True`, `n_heads=H`):

| Component | Storage | Shared across views? |
|-----------|---------|:--------------------:|
| Encoder | single | ✓ |
| `sparse_embedder_heads[h]` (per-view SID emb tables) | per view | ✗ |
| `start_token_embeddings[h]` (BOS) | per view | ✗ |
| `transformer_decoders[h]` | per view | ✗ |
| `sparse_output_projections[ℓ][h]` (D_attn → C_ℓ) | per view, per layer | ✗ |
| `path_rerankers[h]` (§I.6) | per view | ✗ |
| `reranker_path_embedders[h][ℓ]` (§I.6) | per view, per layer | ✗ |

Both decoders cross-attend to the same encoder memory. Their
self-attention only sees the SIDs of their own view (the training-time
teacher-forcing stream is view-$h$'s $L$ codes). The two towers
therefore operate on the two disjoint sub-vector streams produced in
§I.2 without interfering during autoregressive decoding. 

**Per-view beam search.** `generate_per_view_sem_id` runs $H$
independent beam searches over the $L$ RQ layers using
`sparse_output_projections[ℓ][h]` for the logits at layer $\ell$. Each
view returns

```
sids_h  ∈ (B, K, L)   # per-view SID candidates
lps_h   ∈ (B, K)      # NTP log-probabilities of each beam
```

**Per-view item scoring** (loop in `generate_recommendations`, one
per batch row):
- For each candidate SID `s = (c_1, ..., c_L)` from view $h$, look up the
  item collision set $\Phi_h(s) \subseteq \mathcal{V}$ via
  `self.item_ids_by_view[h]`.
- Item score for one candidate: `delta = lp - log|Φ_h(s)|`.
- If the *same* item is captured by multiple beams within the *same*
  view, we combine those `delta` values with **log-sum-exp** (numerically
  stable form, see code) so that the intra-view score matches the
  cross-view fusion below.

**Cross-view OR-fusion** (`_fuse_per_view_scores`). All items that
appear in *any* view's pool form a single union candidate set, and the
per-item scores from the views that contain it are combined by a
configurable reduction (`lse` by default; `max` / `mean` / `rrf` are
also supported for ablation). We use `lse` in the main experiments.

### I.5  Hierarchical Path Reranker (HPR)

Implemented by `HierarchicalPathReranker`. One instance per view (`path_rerankers[h]`).

**Parameters.** For each RQ layer $\ell \in \{0, \ldots, L-1\}$:

| Name | Definition |
|------|------------|
| `context_projs[ℓ]` | `Linear(D_attn, D_path)` (`D_path = 128`) |
| `path_projs[ℓ]`    | `Linear(D, D_path)` |
| `log_temperatures[ℓ]` | `nn.Parameter(log(1/0.07))`, clamped to `[log(0.01), log(100)]` |

Both projections are followed by `F.normalize(..., dim=-1)`, so the
similarity is a cosine.

**Path embedding tables.** HPR uses **private** per-view, per-layer
codeword tables `reranker_path_embedders[h][ℓ] = nn.Embedding(C_ℓ, D)`,
decoupled from `sparse_embedder_heads`. This isolates the reranker's
representation from the NTP output head's representation, which is what
we found stable in practice.

**Score for one layer.** Given a beam candidate at layer $\ell$ with
cumulative path embedding $p_\ell = \sum_{j=0}^{\ell} E^{(h)}_j[c_j]$:

```
ctx     ← normalize(context_projs[ℓ](h0))               # (B, D_path)
path    ← normalize(path_projs[ℓ](p_ℓ))                 # (B, D_path)
score   ← (ctx * path).sum(-1) * temp[ℓ].clamp(0.01, 100)
```

where `h0` is the **BOS-position decoder hidden state** of view $h$.

**Training loss.** `compute_all_layers_loss` — for each view, the mean
of per-layer symmetric InfoNCE across $\ell = 1, \ldots, L-1$.

Per-layer loss (`compute_layer_loss_infonce`):
- Positives: diagonal of the in-batch cosine matrix
  `logits = ctx @ path.T * temp`.
- **Duplicate masking**: rows with identical cumulative path embedding
  (i.e. same SID prefix) have their off-diagonal entries set to $-10^4$.
- Optional **Method-B structured hard negatives**
  (`_build_hard_negatives`), controlled by
  `reranker_hard_neg_swap_current` and `reranker_hard_neg_swap_previous`:
  - *swap_current*: `GT_prefix[0..ℓ-1] + non-GT code at layer ℓ`.
  - *swap_previous*: full GT path with one earlier layer $k < \ell$
    replaced by a non-GT code.
  - Hard negatives are appended only to the c2p direction's denominator
    so labels stay `arange(B)`; the p2c direction stays symmetric
    in-batch. Warm-up is controlled by `reranker_hard_neg_warmup_epochs`.
- The per-layer loss is $(\mathcal{L}_\text{c2p} + \mathcal{L}_\text{p2c})/2$.

**Inference-time fusion in beam search** (excerpt from
`generate_per_view_sem_id`, one view, one layer):

```
# expanded_log_probas: (B, pre_prune_k) NTP log-probs of expanded candidates
# new_path_embs       : (B, pre_prune_k, D) cumulative path embs via HPR-private tables
h_ctx_exp    ← bos_hidden_h.unsqueeze(1).expand(-1, pre_prune_k, -1)
reranker_lp  ← log_softmax(path_rerankers[view_h].score(
                    h_ctx_exp, new_path_embs, layer_idx), dim=-1)
fused_scores ← expanded_log_probas + λ_ℓ * reranker_lp
# then top-K pruning by fused_scores; carry-forward log-prob is NTP only
```

### I.6  Training objective and two-stage pipeline

**Stage 1 — tokenizer.** Train `MvRqVae` with

```
loss = reconstruction_loss + quan_loss_weight * quantize_loss
```

`quantize_loss` is the standard VQ commitment loss summed over all
$L \cdot H$ code positions. Then freeze the tokenizer and pre-compute
SIDs for the whole item catalog.

**Stage 2 — generator.** The generator loss (in `Barge.forward`) is

```
loss = Σ_h Σ_ℓ CE(logits_{h,ℓ},   gt_c_{h,ℓ}, label_smoothing=0.1)
       + reranker_loss_weight * (1/H) Σ_h  HPR_loss_h
```

### I.7  Inference procedure

`generate_recommendations` runs the following pipeline per batch. Below,
`H` is the number of views, `K` the number of recommendations returned,
`K_wide` a per-layer wide beam width used to build the union pool.

```
def recommend(batch, K):
    # (1) shared encoder pass (ICA lives inside _prepare_encoder_input)
    memory ← encoder(_prepare_encoder_input(batch))

    # (2) per-view beam search over L RQ layers
    per_view = []
    for h in 0..H-1:
        beam        ← BOS_h
        beam_lp     ← 0
        beam_path   ← 0                                      # cumulative path emb
        h0          ← decoder_h(beam).bos_hidden             # (B, D_attn)
        for ℓ in 0..L-1:
            logits           ← output_head[ℓ][h](decoder_h(beam, memory)[:, ℓ])
            log_p            ← log_softmax(logits, dim=-1)
            expand           ← beam × top_{pre_prune_k}(log_p)
            new_path_emb     ← beam_path + reranker_path_embedders[h][ℓ](new_code)
            if HPR active on ℓ:
                r_lp         ← log_softmax(path_rerankers[h].score(h0, new_path_emb, ℓ))
                fused        ← expand.log_p + λ_ℓ * r_lp
            else:
                fused        ← expand.log_p
            beam             ← top-K_ℓ prefixes ranked by fused
            beam_lp          ← NTP log-prob carried forward (not fused)
            beam_path        ← gather path embs matching pruned beam
        per_view.append((beam.sids, beam_lp))                # (B, K_wide, L), (B, K_wide)

    # (3) view -> item scoring (per view, per batch row)
    per_view_scores = [{} for _ in range(H)]
    for h, (sids_h, lps_h) in enumerate(per_view):
        for cand, lp in zip(sids_h, lps_h):
            items = item_ids_by_view[h][tuple(cand)]         # collision set
            if not items:  continue
            delta = lp - log(len(items))
            for v in items:
                per_view_scores[h][v] = lse_merge(per_view_scores[h].get(v), delta)

    # (4) OR-fusion across views (mode ∈ {lse, max, mean, rrf})
    fused = _fuse_per_view_scores(per_view_scores, mode=self.or_fusion_mode)

    # (5) sort and return top-K
    return top_K_by_score(fused, K)
```

Points worth stressing:
- Steps (2) and (3) run one view at a time and the results are only
  combined in (4); no cross-view state exists inside beam search.
- Within-view aggregation (step 3) and cross-view aggregation
  (step 4, `lse` mode) share the same LSE reduction, so the intra-
  and inter-view scales are consistent.

### I.8  Hyper-parameters used in the paper

**Tokenizer (Stage 1).**

| Hyper-parameter | Value |
|---|---|
| Item text embedding dim $D_\text{text}$ | $768$ |
| Encoder MLP hidden dims | $[512, 256, 128]$ |
| RQ latent dim $D_z$ | $32$ |
| Number of RQ layers $L$ | $4$ |
| Codebook sizes $(C_1, \ldots, C_L)$ | $[512, 256, 128, 64]$ |
| Number of views $H$ | $2$ |
| Commitment weight $\beta$ | $0.25$ |
| Gumbel-softmax temperature | $0.2$ |
| Learned Householder rotation $R$ | enabled |
| Optimizer | AdamW, lr $1\text{e-}3$, wd $1\text{e-}3$ |
| Iterations | $20{,}000$ |
| Batch size | $1024$ |

**Generator (Stage 2).**

| Hyper-parameter | Value |
|---|---|
| SID embedding dim | $128$ |
| Attention hidden dim $D_\text{attn}$ | $512$ |
| Heads per Transformer block | $4$ |
| Encoder layers | $2$ |
| Decoder layers per view | $2$ (each of the $H$ decoders) |
| Dropout | $0.2$ |
| Max sequence length | $256$ |
| Softmax temperature | $1.0$ |
| Beam width $(K_1, \ldots, K_L)$ | $[20, 20, 20, 20]$ |
| Optimizer | AdamW, lr $1\text{e-}3$, wd $0.01$ |
| Batch size | $256$ |
| Epochs | up to $200$ (early stop on NDCG@20) |
| Label smoothing | $0.1$ |

**BARGE-specific knobs.**

| Hyper-parameter | Value |
|---|---|
| ICA MHA heads | $D / 32 = 4$ |
| OR-fusion mode | `lse` (log-sum-exp) |
| HPR inference weight $\lambda_\ell$ | $[0.25, 0.25, 0.25, 0.25]$ |
| HPR shortlist size $N_\ell$ | $[400, 400, 400, 400]$ |
| HPR active layers $\mathcal{L}_\text{on}$ | $\{0, 1, 2, 3\}$ |

---

## Part II. Additional Experiments

> *Due to the strict space limit of the rebuttal response, we place
> part of experiments here and analyse them. Thank you for your
> understanding!*

### II.A  Additional ablation studies

We remove one component at a time from the full BARGE: **w/o ICA** disables the item-context gate; **w/o HPR** sets $w_\text{HPR}=0$ and $\lambda_\ell\equiv 0$; **w/o DPD** collapses to $H=1$. BARGE reports mean ± std over 3 seeds.

**Amazon Beauty.**

| Variant | R@5 | N@5 | R@10 | N@10 |
|---|:---:|:---:|:---:|:---:|
| BARGE w/o ICA | 0.0611 | 0.0434 | 0.0868 | 0.0516 |
| BARGE w/o HPR | 0.0612 | 0.0437 | 0.0866 | 0.0519 |
| BARGE w/o DPD | 0.0559 | 0.0391 | 0.0768 | 0.0464 |
| **BARGE** | **0.0654 ± 0.0006** | **0.0460 ± 0.0015** | **0.0927 ± 0.0004** | **0.0547 ± 0.0007** |

**Amazon Sports.**

| Variant | R@5 | N@5 | R@10 | N@10 |
|---|:---:|:---:|:---:|:---:|
| BARGE w/o ICA | 0.0339 | 0.0232 | 0.0497 | 0.0283 |
| BARGE w/o HPR | 0.0344 | 0.0228 | 0.0515 | 0.0284 |
| BARGE w/o DPD | 0.0308 | 0.0207 | 0.0469 | 0.0259 |
| **BARGE** | **0.0369 ± 0.0009** | **0.0252 ± 0.0016** | **0.0544 ± 0.0007** | **0.0308 ± 0.0008** |

Removing any component hurts on all metrics. DPD is the most important (≈15% drop in R@5 on both datasets); ICA and HPR bring smaller but consistent gains on top of it.

### II.B  Codebook collision rate

We report the collision rate of the SIDs produced by our multi-view RQ-VAE tokenizer on all three datasets. Let $N$ be the number of items and $U$ the number of distinct SIDs assigned by the tokenizer; the aggregate collision rate is defined as $1 - U / N$ (0% means every item receives a unique SID).

| Dataset | # items $N$ | Tokenizer | # unique SIDs $U$ | Collision rate $1 - U/N$ |
|---|:---:|:---:|:---:|:---:|
| Amazon Beauty  | 12101 | RQ-VAE          | 11801 | 2.48\% |
| Amazon Beauty  | 12101 | Ours (OSQ-VAE)  | 12096 | 0.04\% |
| Amazon Sports  | 18357 | RQ-VAE          | 17743 | 3.34\% |
| Amazon Sports  | 18357 | Ours (OSQ-VAE)  | 18284 | 0.40\% |
| Amazon Toys    | 11924 | RQ-VAE          | 11660 | 2.21\% |
| Amazon Toys    | 11924 | Ours (OSQ-VAE)  | 11886 | 0.32\% |

Compared with the vanilla RQ-VAE tokenizer, our OSQ-VAE further reduces the collision rate on every dataset, keeping it below 0.5% (i.e. over 99.5% of items receive a unique SID).

### II.C  Catalog size scaling

The public Amazon splits used above contain on the order of $10^4$ items. To check that BARGE remains effective at a much larger scale, we additionally run it on an industrial dataset with **>100M users, >300k items and >100M interactions**, i.e. one order of magnitude more items and roughly four orders of magnitude more interactions than the public benchmarks. BARGE continues to outperform the baseline, indicating that the design does not rely on a small item space.

### II.D  Domain shift

We also probe cross-domain transfer by taking the BARGE and TIGER checkpoint trained on Sports and evaluating it directly on Beauty without any fine-tuning. Both models produce essentially zero recommendation quality in this setting. This is expected and is a shared limitation of SID-based generative recommenders: the tokenizer's SIDs are constructed and specialised for one catalog, so the generator learns to predict the specific SID distribution; SIDs produced on a different catalog do not lie in the same code space, and the model cannot map to them without retraining the tokenizer. Trading cross-domain reusability for a domain-specialised representation is an inherent property of this family of methods, not a BARGE-specific artefact.



### II.E  Cold-start item experiment

Following TIGER, we construct the cold-start set as follows: we deduplicate the last item of every user sequence and randomly sample 5% of them as cold-start items; during training these items are removed from both histories and targets, and the tokenizer (RQ-VAE / OSQ-VAE) is trained only on the remaining non-cold items. The frozen tokenizer is then used to assign SIDs to the cold items, and evaluation is performed on a complete test set. The codebook is fixed to `[256, 256, 256] + random suffix`.

Split on Amazon Toys: 374 / 7475 cold items (5%, seed=42), 18179 training samples.

| Method | R@5 | N@5 | R@10 | N@10 |
|---|:---:|:---:|:---:|:---:|
| TIGER | 0.0378 | 0.0244 | 0.0563 | 0.0303 |
| **BARGE** | **0.0464** | **0.0325** | **0.0644** | **0.0383** |

### II.F  Inference latency breakdown

We profile BARGE's inference on the Amazon Toys test set and aggregate the per-tag timings into the four components introduced in Part I. The total wall-clock time is 851 ms/batch:

| Module | Share |
|---|:---:|
| Encoder (incl. ICA) | **1.84%** |
| **DPD** (per-view decoding + prefix filter + view→item) | **94.63%** |
| **HPR** (path embedding + dual-tower scoring + fusion) | **0.98%** |
| Other | ≈ 2.55% |

1. **HPR is essentially free at inference.** The three HPR tags together account for only about **1%** of the total time, because each step performs just two small linear projections and one dot product.
2. **DPD does not add latency through parallelism; the cost is paid in GPU memory.** The $H$ per-view decoders, output heads and beam kernels are executed on the same encoder memory as independent CUDA streams / batched GEMMs, and their compute overlaps almost entirely as long as the GPU is not saturated. Going from $H=1$ to $H=2$ therefore mostly increases **GPU memory** consumption (extra decoder weights, KV-cache and beam tensors) rather than wall-clock time — in essence, DPD trades **memory for inference speed**, fully utilising otherwise idle GPU capacity.

