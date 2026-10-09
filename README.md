# KV Cache Playground

An interactive course that follows context and a user question through a trained two-layer toy decoder, then examines KV architecture, speculative decoding, multi-token prediction, cache transfer, eviction, and serving. Each chapter connects first principles and equations to numerical labs.

**Live guide:** [ravisankarg.github.io/kv-cache-playground](https://ravisankarg.github.io/kv-cache-playground/)

## Start here

Open `docs/index.html` through GitHub Pages, or serve the page locally from the repository root:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/docs/`. The page loads checked-in weights from `docs/data/course_model.json` and GPU architecture traces from `docs/data/gpu_demo.json`. Inference and interactive calculations run locally in the browser; no model download or JavaScript package installation is needed.

## Chapters and experiments

| Chapter | First principles and equations | Numerical labs |
| --- | --- | --- |
| 1. The decoder | Prompt tokens, embeddings, both residual/MLP blocks, Q/K/V, causal masking, RoPE, softmax, prefill, decode, context caps | Change facts or question order; inspect vocabulary probabilities, every attention score, both caches, and strict/sliding overflow |
| 2. KV architectures | MHA/GQA/MQA head sharing, cache-byte formulas, MLA latent representations and matrix absorption | Switch GPU layouts; compare expanded and absorbed attention numerically |
| 3. Faster decoding | Candidate verification, cache rollback, exact speculative-sampling correction, auxiliary MTP training | Accept/reject proposals; compare analytic probability mass with 20,000 sampled trials; inspect both trained future heads |
| 4. Reuse and transfer | Compatible same-model handoff, per-head ridge maps, RoPE factoring, causal prefix boundaries | Compare an invertible cache basis change with an independent random decoder; change ridge strength and prefix block size |
| 5. Memory and serving | Eviction and renormalization, affine quantization, logical/physical paging, online softmax | Compare retention policies and output TV; quantize real toy K/V; inspect block addresses and tile-by-tile attention accumulation |
| 6. Further directions | Orthogonal rotation, cross-layer sharing, sparse cache retrieval, chunked prefill | Isolate rotation plus quantization on an outlier vector; link the other directions to primary sources |

The original architecture walkthrough remains available at [`docs/trace.html`](docs/trace.html).

## The trained two-layer decoder

`experiments/build_course.py` trains a synthetic associative-recall task. Prompts have three fixed key slots, **france, japan, india in that order**, with three distinct values sampled from **paris, tokyo, delhi, rome, cairo, lima**. Questions name two distinct keys. The model emits the corresponding values in question order, followed by EOS. These assignments are synthetic, not geography claims.

The decoder has 23 vocabulary IDs, two pre-norm blocks, width 32, four query/KV heads of width 8, a width-64 gated MLP, learned token and position embeddings, RoPE, and a 32-token runtime cap. A second vocabulary head learns the second future token with loss weight 0.3. It is a simple independent auxiliary head, rather than DeepSeek-V3's sequential MTP modules.

Value assignments are split before sampling: the weighted value-index sum modulo 5 equals zero for evaluation only; all other assignments are used for training. The checked-in run used seed 991, 8,000 batches of 64 sampled examples, and an NVIDIA GeForce RTX 3050 Laptop GPU with PyTorch 2.8.0+cu128. On 256 sampled held-out prompts, next-token, teacher-forced base second-token, and MTP second-token accuracy were each 100%. These measurements describe this narrow lookup task.

```bash
python experiments/build_course.py --steps 8000
node tests/numerical.mjs
```

The Python generator requires CUDA and exports weights, training metadata, a GPU reference distribution, cache-equivalence audits, and held-out transfer results. The Node audit checks browser/GPU parity, independent matrix-prefill equivalence, causal invariance, rollback, tokenizer behavior, context limits, speculative probability mass, online softmax, and recall presets. `tests/browser.html?width=390` runs the interaction audit at phone width when served from the repository root; the default width is 1280.

## Implementation map

| File | Role |
| --- | --- |
| `docs/index.html`, `docs/course.css` | Chapter content, equations, controls, responsive layout |
| `docs/engine.js` | Pure decoder math, independent matrix prefill, cached decode, speculative mass, quantization, online attention |
| `docs/course.js` | Numerical visualizations and linked lab controls |
| `experiments/build_course.py` | CUDA training, audits, per-head cache mapping, JSON export |
| `kv_cache_demo.py` | Untrained CUDA MHA/GQA/MQA/MLA-like architecture traces |
| `tests/numerical.mjs`, `tests/browser.html` | Numerical and browser interaction audits |

## Recompute the trace on a GPU

The simulator requires a working NVIDIA GPU and CUDA-enabled PyTorch. Install the PyTorch build that matches your system and driver using the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then run:

```bash
python kv_cache_demo.py --variant all
```

This writes a fresh trace to `docs/data/gpu_demo.json`. The command deliberately exits with an error if CUDA is unavailable; it never falls back to CPU. Options:

```bash
python kv_cache_demo.py --variant gqa --layers 3 --context 6 --tokens 8
python kv_cache_demo.py --variant mla --context 8 --tokens 10 --output /tmp/mla-trace.json
```

The built-in token stream contains ten fixed labels. `--tokens` chooses how many vectors are processed; `--context` sets the cache limit. The strict trace stops at the cap. The sliding trace continues and evicts the oldest position before adding a new row.

## What the architecture trace computes

Each token is a deterministic random vector of width 32. Two decoder blocks are initialized with deterministic random weights. A block follows a pre-norm residual pattern:

1. RMS normalization; query, key and value projections; rotary position embedding on Q/K.
2. Scaled dot-product attention over keys and values currently in that layer's cache.
3. Output projection, residual connection, and a small gated MLP.
4. Retain the new K/V representation for the next token.

The decode loop only exposes positions already processed, so its causal mask is enforced by cache visibility. The trace records each head's query, score and softmax weights; each layer's cache shape and a numeric preview; and the block output. The architecture and overflow labs visualize those GPU-computed records, separately from the trained browser decoder.

| Variant | What is stored per token, per layer | Demo dimensions |
| --- | --- | --- |
| MHA | K and V for all 4 query heads | `[tokens, 4, 8]` each |
| GQA | K and V for 2 shared KV heads | `[tokens, 2, 8]` each |
| MQA | K and V for 1 shared KV head | `[tokens, 1, 8]` each |
| MLA-like | Shared 8-value content latent plus a 2-value decoupled RoPE key | `[tokens, 8]` and `[tokens, 2]` |

The MLA path is deliberately compact and illustrative. It caches a low-rank content vector and a small positional key, then projects the latent back into per-head K/V features for attention. DeepSeek-V2's MLA includes additional architectural details and should be consulted for a faithful model implementation.

## At the context limit

There is no universal overflow behavior. A strict model/runtime cap may stop or reject another decode step; a sliding-window runtime may evict old rows; other systems may use sink tokens, chunks, or model-specific positional extensions. The final token inside the cap is processed normally and attends to every visible cache row. It predicts the next token, but that next token gets its own K/V rows only after the runtime feeds it through the layers under an allowed policy.

The site contrasts a strict stop with a sliding window. Its byte chart uses the stored scalar count × selected FP16/BF16 or FP32 bytes and excludes allocator overhead, quantization scales, and temporary tensors.

## Scope

The main decoder is trained only on the fixed-key synthetic lookup task. It is not a pretrained conversational model, and edits outside that task are explicitly flagged in the page. Its learned positional table does not support extrapolation beyond 32 positions.

The architecture trace uses fixed token labels and untrained random weights, so it demonstrates storage and attention mechanics rather than architecture quality. The controlled transfer case re-expresses the target's own cache in an orthogonal basis; the independent case uses a randomly initialized source decoder. Neither reproduces real pretrained-model transfer. Quantization, paging, online softmax, and rotation labs isolate their named operations rather than implementing complete production algorithms. No lab measures production latency or claims a general language benchmark.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
- Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- Heo et al., [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/abs/2608.03893)
- [H₂O](https://arxiv.org/abs/2306.14048), [StreamingLLM](https://arxiv.org/abs/2309.17453), and [KIVI](https://arxiv.org/abs/2402.02750)
- [PagedAttention](https://arxiv.org/abs/2309.06180) and [FlashAttention](https://arxiv.org/abs/2205.14135)
- [TurboQuant](https://arxiv.org/abs/2504.19874), [Cross-layer attention](https://arxiv.org/abs/2405.12981), and [Sarathi-Serve](https://arxiv.org/abs/2403.02310)

## Publish

The site is static and lives under `docs/`. A GitHub Actions workflow in `.github/workflows/pages.yml` deploys that directory to GitHub Pages on pushes to `main`. The repository's Pages source should be set to **GitHub Actions**.
