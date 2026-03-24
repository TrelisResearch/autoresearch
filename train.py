"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import random
import time
from dataclasses import dataclass, asdict

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# RecursiveGPT — same blocks but recur segment runs K times (shared weights)
# Architecture: prelude (2 layers) → recur (4 layers × K) → coda (2 layers)
# Gate: g = gate_min + (1-gate_min)*sigmoid(gate_proj(cat[u,s])) ∈ [gate_min, 1]
# Update: s = s + g*(u - s)   (step 0 always forced full update)
# ---------------------------------------------------------------------------

class RecursiveGPT(nn.Module):
    def __init__(self, config, k_recurse=4, use_gate=False, gate_min=0.1, lora_rank=0, use_grad_ckpt=False):
        super().__init__()
        self.config = config
        self.k_recurse = k_recurse
        self.use_gate = use_gate
        self.gate_min = gate_min
        self.lora_rank = lora_rank
        self.use_grad_ckpt = use_grad_ckpt
        n_embd = config.n_embd
        head_dim = n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim

        # Segment sizes — must sum to config.n_layer
        self.n_pre = 2
        self.n_rec = 4
        self.n_cod = 2
        assert self.n_pre + self.n_rec + self.n_cod == config.n_layer, \
            f"Expected n_layer={self.n_pre+self.n_rec+self.n_cod}, got {config.n_layer}"

        # Compute window sizes for all unique layers
        ws = self._make_window_sizes(config)
        self.pre_ws  = ws[:self.n_pre]
        self.rec_ws  = ws[self.n_pre : self.n_pre + self.n_rec]
        self.cod_ws  = ws[self.n_pre + self.n_rec :]

        self.transformer = nn.ModuleDict({
            "wte":    nn.Embedding(config.vocab_size, n_embd),
            "pre":    nn.ModuleList([Block(config, i) for i in range(self.n_pre)]),
            # Recur blocks: use layer_idx=0 so none get ve_gates (simplicity)
            "rec":    nn.ModuleList([Block(config, 0) for _ in range(self.n_rec)]),
            "cod":    nn.ModuleList([Block(config, self.n_pre + self.n_rec + i)
                                     for i in range(self.n_cod)]),
        })
        self.lm_head = nn.Linear(n_embd, config.vocab_size, bias=False)

        # Value embeddings for prelude (idx 0..n_pre-1) and coda (idx n_pre+n_rec..n_layer-1)
        n_total = config.n_layer
        ve_indices = [i for i in range(n_total)
                      if i < self.n_pre or i >= self.n_pre + self.n_rec]
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in ve_indices if has_ve(i, n_total)
        })

        # Inject: fuse anchor e + current state s → input for recur block
        # Init: weight = [I | 0] so inject(cat[e,s]) ≈ e at t=0
        self.inject = nn.Linear(2 * n_embd, n_embd, bias=False)

        # Gate: per-token scalar in [gate_min, 1], conditioned on proposed update u and state s
        # Bias init +2 → sigmoid(2)≈0.88 → gate≈0.89 (mostly open at start)
        self.gate_proj = nn.Linear(2 * n_embd, 1, bias=True)

        # Learned step embeddings — broadcast over (B, T) at each recurrence step
        # Critical: without these, shared weights see identical inputs every step
        # Init to zero so step 0 is pure prelude output; learned over training
        # Non-zero init: small random values so the model knows which step it's on from step 0
        # (zero init meant all recurrences were indistinguishable early in training)
        self.step_embeds = nn.Parameter(torch.randn(k_recurse, n_embd) * 0.01)

        # Optional per-step LoRA on inject (helps model distinguish recursion depth)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(k_recurse, 2 * n_embd, lora_rank))
            self.lora_B = nn.Parameter(torch.zeros(k_recurse, lora_rank, n_embd))

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rope(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _make_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_map = {"L": (long_w, 0), "S": (short_w, 0)}
        ws = [char_map[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        ws[-1] = (long_w, 0)
        return ws

    def _precompute_rope(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                                  device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        torch.nn.init.normal_(self.transformer.wte.weight, 0.0, 1.0)
        torch.nn.init.normal_(self.lm_head.weight, 0.0, 0.001)
        for seg in [self.transformer.pre, self.transformer.rec, self.transformer.cod]:
            for block in seg:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
                if block.attn.ve_gate is not None:
                    torch.nn.init.zeros_(block.attn.ve_gate.weight)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Inject: identity-like init  [I | 0]
        torch.nn.init.zeros_(self.inject.weight)
        self.inject.weight.data[:, :n_embd].copy_(torch.eye(n_embd))
        # Gate: bias +2 → gates start mostly open
        torch.nn.init.zeros_(self.gate_proj.weight)
        torch.nn.init.constant_(self.gate_proj.bias, 2.0)
        if self.lora_rank > 0:
            torch.nn.init.normal_(self.lora_A, 0.0, 0.01)
            torch.nn.init.zeros_(self.lora_B)
        # bf16 embeddings
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rope(self.rotary_seq_len, head_dim)

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        n_embd = self.config.n_embd
        dmodel_lr_scale = (n_embd / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({n_embd}/768) = {dmodel_lr_scale:.6f}")

        # 2D matrix params → Muon (inject.weight is (H, 2H) → fine for Muon)
        matrix_params = (list(self.transformer.pre.parameters()) +
                         list(self.transformer.rec.parameters()) +
                         list(self.transformer.cod.parameters()) +
                         [self.inject.weight])
        # gate_proj: weight shape (1, 2H) — not suitable for orthogonalisation → AdamW
        gate_params = list(self.gate_proj.parameters()) if self.use_gate else []
        # step_embeds and lora: scalar / small tensors → AdamW
        step_params = [self.step_embeds]
        lora_params = ([self.lora_A, self.lora_B] if self.lora_rank > 0 else [])
        embedding_params = list(self.transformer.wte.parameters())
        ve_params = list(self.value_embeds.parameters())
        lm_head_params = list(self.lm_head.parameters())

        param_groups = [
            dict(kind='adamw', params=lm_head_params,   lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params,  lr=embedding_lr   * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ve_params,         lr=embedding_lr   * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=step_params,       lr=scalar_lr      * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        if gate_params:
            param_groups.append(dict(kind='adamw', params=gate_params, lr=scalar_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        if lora_params:
            param_groups.append(dict(kind='adamw', params=lora_params, lr=scalar_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        for shape in sorted({p.shape for p in matrix_params}):
            grp = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=grp, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.95,
                                     weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def num_scaling_params(self):
        wte   = sum(p.numel() for p in self.transformer.wte.parameters())
        ve    = sum(p.numel() for p in self.value_embeds.parameters())
        lmh   = sum(p.numel() for p in self.lm_head.parameters())
        pre   = sum(p.numel() for p in self.transformer.pre.parameters())
        rec   = sum(p.numel() for p in self.transformer.rec.parameters())
        cod   = sum(p.numel() for p in self.transformer.cod.parameters())
        inj   = self.inject.weight.numel()
        gate  = sum(p.numel() for p in self.gate_proj.parameters()) if self.use_gate else 0
        step  = self.step_embeds.numel()
        lora  = sum(p.numel() for p in [self.lora_A, self.lora_B]) if self.lora_rank > 0 else 0
        total = wte + ve + lmh + pre + rec + cod + inj + gate + step + lora
        return {'wte': wte, 'value_embeds': ve, 'lm_head': lmh,
                'prelude': pre, 'recur(shared)': rec, 'coda': cod,
                'inject': inj, 'gate_proj': gate, 'step_embeds': step, 'lora': lora, 'total': total}

    def estimate_flops(self):
        # Count effective FLOPs: recur block is run k_recurse times
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embed = (self.transformer.wte.weight.numel() +
                         sum(ve.weight.numel() for ve in self.value_embeds.values()))
        h, q = self.config.n_head, self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn = 12 * h * q * t * (self.n_pre + self.n_rec * self.k_recurse + self.n_cod)
        return 6 * (nparams - nparams_embed) + attn

    def _run_recur_blocks(self, u, cos, sin):
        """Run all shared recur blocks on u. Extracted as method for gradient checkpointing."""
        cos_sin = (cos, sin)
        for j, block in enumerate(self.transformer.rec):
            u = block(u, None, cos_sin, self.rec_ws[j])
        return u

    def forward(self, idx, targets=None, reduction='mean', k_override=None):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        k_recurse = k_override if k_override is not None else self.k_recurse

        x = self.transformer.wte(idx)
        x = norm(x)

        # --- Prelude (run once) ---
        for i, block in enumerate(self.transformer.pre):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.pre_ws[i])
        e = x          # anchor: frozen context from prelude
        s = e.clone()  # mutable recurrent state

        # --- Recurrence (K steps, shared recur blocks) ---
        # k=0 is always a full update (baseline recurrence, not penalised).
        # k=1..K-1 are gated: gate_mean/gate_std only track these steps.
        # This way gate_mean=0 → 1 effective recurrence, gate_mean=1 → K recurrences.
        gate_sum = x.new_zeros(1)
        gate_sq_sum = x.new_zeros(1)
        n_gated = 0
        for k in range(k_recurse):
            # Fuse anchor + state, then inject step identity signal
            u = self.inject(torch.cat([e, s], dim=-1))
            u = u + self.step_embeds[k]   # broadcast (n_embd,) over (B, T, n_embd)
            # Optional per-step LoRA delta on top of inject
            if self.lora_rank > 0:
                u = u + (torch.cat([e, s], dim=-1) @ self.lora_A[k]) @ self.lora_B[k]
            # Run shared recur blocks (with optional gradient checkpointing to save activation memory)
            if self.use_grad_ckpt:
                u = grad_checkpoint(self._run_recur_blocks, u, cos_sin[0], cos_sin[1], use_reentrant=False)
            else:
                for j, block in enumerate(self.transformer.rec):
                    u = block(u, None, cos_sin, self.rec_ws[j])
            # k=0: always full update; k≥1: gate (if enabled)
            if not self.use_gate or k == 0:
                # Full update — base recurrence, not included in gate_mean tracking
                s = u
            else:
                g = self.gate_min + (1 - self.gate_min) * torch.sigmoid(
                    self.gate_proj(torch.cat([u, s], dim=-1)))
                gate_sum = gate_sum + g.mean()
                gate_sq_sum = gate_sq_sum + g.square().mean()
                n_gated += 1
                s = s + g * (u - s)

        # gate_mean over gated steps only (k=1..k_recurse-1); 0 if no gated steps
        if n_gated > 0:
            gate_mean = gate_sum / n_gated
            gate_var = gate_sq_sum / n_gated - (gate_sum / n_gated) ** 2
            gate_std = gate_var.clamp_min(0).sqrt()
        else:
            gate_mean = x.new_zeros(1)
            gate_std = x.new_zeros(1)

        # --- Coda (run once) ---
        x = s
        for i, block in enumerate(self.transformer.cod):
            abs_i = self.n_pre + self.n_rec + i
            ve = self.value_embeds[str(abs_i)](idx) if str(abs_i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.cod_ws[i])

        x = norm(x)
        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                 ignore_index=-1, reduction=reduction)
            return ce, gate_mean, gate_std   # caller adds lambda * gate_mean to loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.06        # learning rate for matrix parameters (Muon) — 0.06 best from initial-tests
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers (ignored when USE_RECURSIVE=True)
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# Recursive architecture
USE_RECURSIVE  = True   # True = RecursiveGPT, False = standard GPT
K_RECURSE      = 2      # recurrence steps (effective depth = pre + rec*K + cod); K=2 → ~540 steps/5min
USE_GATE       = True   # True = learned gate; False = always full update (g=1, simple recursion)
GATE_MIN       = 0.1    # gate floor: 0.1 = leaky floor (prevents NaN from g=0 collapse)
LAMBDA_GATE    = 0.0    # penalty on gate_mean of gated steps (k≥1); positive = close gates; negative = open gates
VAR_REWARD     = 0.0    # reward gate variance across tokens: loss -= VAR_REWARD * Var(g)
LORA_RANK      = 0      # per-step LoRA rank (0=disabled); P3b showed LoRA+RANDOM_K is catastrophic
RANDOM_K       = True   # randomly sample K_eff in [1, K_RECURSE] each step during training
USE_GRAD_CKPT  = True   # gradient checkpointing on recur blocks (saves ~K× activation memory → BS=128 with K=4)
# When USE_RECURSIVE=True: DEPTH is set to PRELUDE+RECUR+CODA=8 automatically

# Experiment tracking
RUN_NAME = "p3c-k2-randomK-gate"  # change per experiment
WANDB_PROJECT = "autoresearch-recursive-gate"

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

# For recursive model, unique layer count = 2+4+2=8; for standard use DEPTH
_depth = (2 + 4 + 2) if USE_RECURSIVE else DEPTH
config = build_model_config(_depth)
print(f"Model config: {asdict(config)}")
print(f"Mode: {'RECURSIVE K=' + str(K_RECURSE) + ' lambda=' + str(LAMBDA_GATE) + ' lora_rank=' + str(LORA_RANK) if USE_RECURSIVE else 'STANDARD'}")

wandb.login(key=os.environ.get("WANDB_API_KEY"))
wandb.init(
    project=WANDB_PROJECT,
    name=RUN_NAME,
    config=dict(
        use_recursive=USE_RECURSIVE, k_recurse=K_RECURSE, gate_min=GATE_MIN,
        lambda_gate=LAMBDA_GATE, var_reward=VAR_REWARD, random_k=RANDOM_K,
        lora_rank=LORA_RANK, use_grad_ckpt=USE_GRAD_CKPT,
        depth=_depth, aspect_ratio=ASPECT_RATIO, head_dim=HEAD_DIM,
        window_pattern=WINDOW_PATTERN, total_batch_size=TOTAL_BATCH_SIZE,
        embedding_lr=EMBEDDING_LR, unembedding_lr=UNEMBEDDING_LR,
        matrix_lr=MATRIX_LR, scalar_lr=SCALAR_LR, weight_decay=WEIGHT_DECAY,
        adam_betas=ADAM_BETAS, warmup_ratio=WARMUP_RATIO,
        warmdown_ratio=WARMDOWN_RATIO, final_lr_frac=FINAL_LR_FRAC,
    ),
)

with torch.device("meta"):
    if USE_RECURSIVE:
        model = RecursiveGPT(config, k_recurse=K_RECURSE, use_gate=USE_GATE, gate_min=GATE_MIN, lora_rank=LORA_RANK, use_grad_ckpt=USE_GRAD_CKPT)
    else:
        model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    gate_mean_accum = 0.0
    gate_std_accum = 0.0
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            if USE_RECURSIVE:
                k_eff = random.randint(1, K_RECURSE) if RANDOM_K else K_RECURSE
                ce, gate_mean_t, gate_std_t = model(x, y, k_override=k_eff)
                # gate_mean penalty/reward + variance reward (rewards gate heterogeneity across tokens)
                loss = ce + LAMBDA_GATE * gate_mean_t - VAR_REWARD * (gate_std_t ** 2)
                gate_mean_accum += gate_mean_t.detach().item()
                gate_std_accum += gate_std_t.detach().item()
            else:
                loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)
    gate_mean_val = gate_mean_accum / grad_accum_steps if USE_RECURSIVE else 0.0
    gate_std_val = gate_std_accum / grad_accum_steps if USE_RECURSIVE else 0.0

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    gate_str = f" | gate_mean: {gate_mean_val:.3f} std: {gate_std_val:.3f}" if USE_RECURSIVE else ""
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}%{gate_str} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step % 10 == 0:
        log_dict = {"train/loss": debiased_smooth_loss, "train/lrm": lrm,
                    "train/mfu": mfu, "train/tok_per_sec": tok_per_sec,
                    "train/step_ms": dt * 1000}
        if USE_RECURSIVE:
            log_dict["train/gate_mean"] = gate_mean_val
            log_dict["train/gate_std"] = gate_std_val
        wandb.log(log_dict, step=step)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    if USE_RECURSIVE:
        # evaluate_bpb expects a single tensor; RecursiveGPT returns (ce, gate_mean, gate_std) tuple
        class _CEOnlyModel:
            def __call__(self, x, y, reduction='mean'):
                ce, _, __ = model(x, y, reduction=reduction)
                return ce
        val_bpb = evaluate_bpb(_CEOnlyModel(), tokenizer, DEVICE_BATCH_SIZE)
    else:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# Compute final gate_mean/gate_std from eval pass if recursive
final_gate_mean = 0.0
final_gate_std = 0.0
if USE_RECURSIVE:
    model.eval()
    with autocast_ctx, torch.no_grad():
        eval_loader = make_dataloader(tokenizer, min(DEVICE_BATCH_SIZE, 32), MAX_SEQ_LEN, "val")
        ex, ey, _ = next(eval_loader)
        _, gm, gs = model(ex.to(device), ey.to(device))
        final_gate_mean = gm.item()
        final_gate_std = gs.item()
    model.train()

final_log = {
    "final/val_bpb": val_bpb,
    "final/training_seconds": total_training_time,
    "final/peak_vram_mb": peak_vram_mb,
    "final/mfu_percent": steady_state_mfu,
    "final/total_tokens_M": total_tokens / 1e6,
    "final/num_steps": step,
    "final/num_params_M": num_params / 1e6,
}
if USE_RECURSIVE:
    final_log["final/gate_mean"] = final_gate_mean
    final_log["final/gate_std"] = final_gate_std
    final_log["final/effective_k"] = 1 + final_gate_mean * (K_RECURSE - 1)
wandb.log(final_log)
wandb.finish()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {_depth}")
if USE_RECURSIVE:
    print(f"gate_mean:        {final_gate_mean:.4f}")
    print(f"gate_std:         {final_gate_std:.4f}")
    print(f"effective_k:      {1 + final_gate_mean * (K_RECURSE - 1):.2f}")

# Auto-log to results.tsv
import subprocess, csv, os as _os
_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=_os.path.dirname(_os.path.abspath(__file__))).decode().strip()
_eff_k  = 1 + final_gate_mean * (K_RECURSE - 1) if USE_RECURSIVE else 1.0
_tsv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "results.tsv")
_header = "commit\tval_bpb\tgate_mean\tgate_std\teffective_k\tpeak_vram_mb\tstatus\tdescription\n"
if not _os.path.exists(_tsv_path):
    open(_tsv_path, "w").write(_header)
with open(_tsv_path, "a") as _f:
    _f.write(f"{_commit}\t{val_bpb:.6f}\t{final_gate_mean:.4f}\t{final_gate_std:.4f}\t{_eff_k:.2f}\t{peak_vram_mb:.1f}\tok\t{RUN_NAME}\n")
print(f"Logged to results.tsv")
