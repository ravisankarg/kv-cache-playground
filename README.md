# KV Cache Playground

An interactive, first-principles walkthrough of how a causal decoder creates and updates its key/value cache, one token at a time. Compare **MHA**, **GQA**, **MQA**, and a compact **MLA** variant; inspect queries, attention weights, stored cache rows, and the behavior at a context limit.

**Live guide:** [ravisankarg.github.io/kv-cache-playground](https://ravisankarg.github.io/kv-cache-playground/)

## Start here

Open `docs/index.html` through GitHub Pages, or serve the page locally from the repository root:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/docs/`. The page reads the checked-in GPU trace at `docs/data/gpu_demo.json`; it does not need a model download or JavaScript package installation.

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

## What the code computes

Each token is a deterministic random vector of width 32. Two decoder blocks are initialized with deterministic random weights. A block follows a pre-norm residual pattern:

1. RMS normalization; query, key and value projections; rotary position embedding on Q/K.
2. Scaled dot-product attention over keys and values currently in that layer's cache.
3. Output projection, residual connection, and a small gated MLP.
4. Retain the new K/V representation for the next token.

The decode loop only exposes positions already processed, so its causal mask is enforced by cache visibility. The trace records each head's query, score and softmax weights; each layer's cache shape and a numeric preview; and the block output. The page visualizes those GPU-computed records.

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

This is a **mechanics demonstration**, not a trained model. The token stream is fixed, weights are untrained, and there is no tokenizer, vocabulary projection, sampling, batching, or speed benchmark. The default trace uses a small CUDA workload so intermediate values are easy to inspect. It makes no claim about language quality or production inference speed.

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Shazeer, [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## Publish

The site is static and lives under `docs/`. A GitHub Actions workflow in `.github/workflows/pages.yml` deploys that directory to GitHub Pages on pushes to `main`. The repository's Pages source should be set to **GitHub Actions**.
