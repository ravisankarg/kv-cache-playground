#!/usr/bin/env python3
"""A tiny, GPU-only causal decoder that exposes KV-cache updates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


TOKENS = ["<BOS>", "the", "cache", "grows", "one", "token", "at", "a", "time", "<EOS>"]
VARIANTS = ("mha", "gqa", "mqa", "mla")


def _randn(shape: tuple[int, ...], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device)


def _rope(x: torch.Tensor, position: int) -> torch.Tensor:
    dim = x.shape[-1]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    inv = 10000.0 ** (-torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) / dim)
    angle = position * inv
    c, s = angle.cos(), angle.sin()
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2)


def _preview(x: torch.Tensor, n: int = 4) -> list[float]:
    return [round(float(v), 5) for v in x.detach().reshape(-1)[:n].cpu()]


def _matrix_preview(rows: list[torch.Tensor], n: int = 4) -> list[list[float]]:
    return [_preview(row, n) for row in rows]


def _build_weights(variant: str, d_model: int, heads: int, head_dim: int, seed: int,
                   device: torch.device) -> dict[str, torch.Tensor | int]:
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def w(shape: tuple[int, ...], fan_in: int) -> torch.Tensor:
        return _randn(shape, gen, device) / math.sqrt(fan_in)

    kv_heads = {"mha": heads, "gqa": heads // 2, "mqa": 1}.get(variant)
    params: dict[str, torch.Tensor | int] = {
        "wo": w((heads, head_dim, d_model), heads * head_dim),
        "wg": w((d_model, 2 * d_model), d_model),
        "wu": w((d_model, 2 * d_model), d_model),
        "wd": w((2 * d_model, d_model), 2 * d_model),
    }
    if variant == "mla":
        # Simplified MLA: one shared content latent plus a decoupled RoPE key.
        rank, rope_dim = 8, 2
        d_nope = head_dim - rope_dim
        params.update({
            "rank": rank, "rope_dim": rope_dim, "d_nope": d_nope,
            "wq_nope": w((d_model, heads, d_nope), d_model),
            "wq_pe": w((d_model, heads, rope_dim), d_model),
            "w_dkv": w((d_model, rank), d_model),
            "w_uk": w((rank, heads, d_nope), rank),
            "w_uv": w((rank, heads, head_dim), rank),
            "w_kpe": w((d_model, rope_dim), d_model),
        })
    else:
        assert kv_heads is not None
        params.update({
            "kv_heads": kv_heads,
            "wq": w((d_model, heads, head_dim), d_model),
            "wk": w((d_model, kv_heads, head_dim), d_model),
            "wv": w((d_model, kv_heads, head_dim), d_model),
        })
    return params


def _process_token(x: torch.Tensor, position: int, cache: dict[str, Any], p: dict[str, Any],
                   variant: str, heads: int, head_dim: int, context: int,
                   policy: str) -> tuple[torch.Tensor, dict[str, Any]]:
    d_model = x.shape[0]
    # Pre-norm attention sublayer.
    x_norm = x * torch.rsqrt(x.square().mean() + 1e-6)
    q_preview: list[list[float]] = []
    score_rows: list[list[float]] = []
    weight_rows: list[list[float]] = []

    if variant == "mla":
        q_nope = torch.einsum("d,dhf->hf", x_norm, p["wq_nope"])
        q_pe = _rope(torch.einsum("d,dhf->hf", x_norm, p["wq_pe"]), position)
        latent = x_norm @ p["w_dkv"]
        k_pe = _rope(x_norm @ p["w_kpe"], position)
        cache["latent"].append(latent)
        cache["rope"].append(k_pe)
        cache["positions"].append(position)
        if policy == "sliding" and len(cache["positions"]) > context:
            for key in ("latent", "rope", "positions"):
                cache[key].pop(0)
        latents = torch.stack(cache["latent"])
        k_rope = torch.stack(cache["rope"])
        keys_nope = torch.einsum("tr,rhd->thd", latents, p["w_uk"])
        values = torch.einsum("tr,rhd->thd", latents, p["w_uv"])
        attended = []
        scale = math.sqrt(head_dim)
        for h in range(heads):
            scores = (keys_nope[:, h] @ q_nope[h] + k_rope @ q_pe[h]) / scale
            weights = scores.softmax(dim=0)
            attended.append(weights @ values[:, h])
            score_rows.append([round(float(v), 5) for v in scores.detach().cpu()])
            weight_rows.append([round(float(v), 5) for v in weights.detach().cpu()])
            q_preview.append(_preview(torch.cat((q_nope[h], q_pe[h])), head_dim))
        stored = {
            "label_k": "cKV latent · first 4 dims",
            "label_v": "kRoPE · both dims",
            "k": _matrix_preview(cache["latent"]),
            "v": _matrix_preview(cache["rope"], 2),
            "positions": list(cache["positions"]),
            "shapes": {"cKV": [len(cache["positions"]), int(p["rank"])],
                       "kRoPE": [len(cache["positions"]), int(p["rope_dim"])]},
            "scalars_per_token": int(p["rank"]) + int(p["rope_dim"]),
        }
    else:
        q = _rope(torch.einsum("d,dhf->hf", x_norm, p["wq"]), position)
        k = _rope(torch.einsum("d,dhf->hf", x_norm, p["wk"]), position)
        v = torch.einsum("d,dhf->hf", x_norm, p["wv"])
        cache["k"].append(k)
        cache["v"].append(v)
        cache["positions"].append(position)
        if policy == "sliding" and len(cache["positions"]) > context:
            for key in ("k", "v", "positions"):
                cache[key].pop(0)
        keys = torch.stack(cache["k"])
        values = torch.stack(cache["v"])
        kv_heads = int(p["kv_heads"])
        group_size = heads // kv_heads
        attended = []
        for h in range(heads):
            kh = h // group_size
            scores = keys[:, kh] @ q[h] / math.sqrt(head_dim)
            weights = scores.softmax(dim=0)
            attended.append(weights @ values[:, kh])
            score_rows.append([round(float(vv), 5) for vv in scores.detach().cpu()])
            weight_rows.append([round(float(vv), 5) for vv in weights.detach().cpu()])
            q_preview.append(_preview(q[h]))
        stored = {
            "label_k": "K · head 0, first 4 dims",
            "label_v": "V · head 0, first 4 dims",
            "k": _matrix_preview(cache["k"] and [row[0] for row in cache["k"]]),
            "v": _matrix_preview(cache["v"] and [row[0] for row in cache["v"]]),
            "positions": list(cache["positions"]),
            "shapes": {"K": [len(cache["positions"]), int(p["kv_heads"]), head_dim],
                       "V": [len(cache["positions"]), int(p["kv_heads"]), head_dim]},
            "scalars_per_token": 2 * int(p["kv_heads"]) * head_dim,
        }

    attn = torch.stack(attended).reshape(-1) @ p["wo"].reshape(heads * head_dim, d_model)
    residual = x + attn
    mlp_in = residual * torch.rsqrt(residual.square().mean() + 1e-6)
    gate = torch.nn.functional.silu(mlp_in @ p["wg"])
    mlp = (gate * (mlp_in @ p["wu"])) @ p["wd"]
    y = residual + mlp
    trace = {
        "input": _preview(x), "q_by_head": q_preview,
        "scores_by_head": score_rows, "weights_by_head": weight_rows,
        "attention_output": _preview(attn), "mlp_output": _preview(mlp),
        "block_output": _preview(y), "cache": stored,
    }
    return y, trace


def run_variant(variant: str, layers: int, context: int, token_count: int,
               d_model: int, heads: int, head_dim: int, device: torch.device,
               policy: str) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20261002)
    embeddings = _randn((token_count, d_model), generator, device)
    weights = [_build_weights(variant, d_model, heads, head_dim, 1000 + layer, device)
               for layer in range(layers)]
    caches: list[dict[str, Any]] = []
    for _ in range(layers):
        caches.append({"positions": [], "k": [], "v": [], "latent": [], "rope": []})

    steps = []
    max_steps = min(token_count, context) if policy == "strict" else token_count
    for position in range(max_steps):
        hidden = embeddings[position]
        layer_traces = []
        for layer in range(layers):
            hidden, trace = _process_token(hidden, position, caches[layer], weights[layer],
                                           variant, heads, head_dim, context, policy)
            layer_traces.append(trace)
        steps.append({"position": position, "token": TOKENS[position % len(TOKENS)],
                      "layers": layer_traces, "final_hidden": _preview(hidden)})

    kv_heads = {"mha": heads, "gqa": heads // 2, "mqa": 1}.get(variant)
    scalars_per_layer = (2 * kv_heads * head_dim if variant != "mla" else 8 + 2)
    return {
        "variant": variant, "policy": policy, "steps": steps,
        "config": {"d_model": d_model, "heads": heads, "head_dim": head_dim,
                   "kv_heads": kv_heads, "layers": layers,
                   "mla_kv_rank": 8 if variant == "mla" else None,
                   "mla_rope_dim": 2 if variant == "mla" else None},
        "scalars_per_token_all_layers": scalars_per_layer * layers,
        "fp16_bytes_per_token": scalars_per_layer * layers * 2,
        "fp32_bytes_per_token": scalars_per_layer * layers * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=10,
                        help="Tokens presented to the decoder; may exceed --context for sliding-window demo.")
    parser.add_argument("--output", type=Path, default=Path("docs/data/gpu_demo.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this demo. No CPU fallback is provided. "
                         "Run in a GPU-enabled environment with a working NVIDIA driver and CUDA PyTorch.")
    if args.layers < 1 or args.context < 1 or not 1 <= args.tokens <= len(TOKENS):
        raise SystemExit("layers/context must be positive and tokens must be between 1 and 10.")
    if args.context > len(TOKENS):
        raise SystemExit("context must be <= 10 for the built-in token sequence.")

    device = torch.device("cuda")
    torch.cuda.init()
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    variants_result = {}
    for variant in variants:
        variants_result[variant] = {
            "strict": run_variant(variant, args.layers, args.context, args.tokens,
                                  32, 4, 8, device, "strict"),
            "sliding": run_variant(variant, args.layers, args.context, args.tokens,
                                   32, 4, 8, device, "sliding"),
        }
    payload = {
        "title": "KV Cache Playground GPU Trace",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "execution": "PyTorch CUDA; all arithmetic and cache tensors were computed on the selected GPU.",
        "synthetic": True,
        "token_sequence": TOKENS[:args.tokens],
        "max_context": args.context,
        "config": {"d_model": 32, "heads": 4, "head_dim": 8, "layers": args.layers},
        "variants": variants_result,
        "scope": "Random, untrained weights expose tensor mechanics; this is not a language-model quality benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"GPU: {payload['device']} | PyTorch {payload['torch']} + CUDA {payload['cuda']}")
    print(f"Wrote {args.output} | context={args.context}, layers={args.layers}, tokens={args.tokens}")
    print("fp16 cache bytes/token (all layers): " + ", ".join(
        f"{name.upper()} {data['strict']['fp16_bytes_per_token']} B"
        for name, data in variants_result.items()))


if __name__ == "__main__":
    main()
