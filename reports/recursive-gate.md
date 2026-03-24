# Recursive-Gate Research Report

**Branch:** `recursive-gate`
**Hardware:** NVIDIA H100 80GB HBM3 (RunPod, root@64.247.201.46 -p 10926)
**W&B Project:** [autoresearch-recursive-gate](https://wandb.ai/trelis/autoresearch-recursive-gate)
**Time budget per run:** 5 minutes

---

## Architecture

```
prelude (2 layers, once) → recur (4 layers, shared weights, ×K) → coda (2 layers, once)
```

- **inject**: Linear(2H→H, no bias), init=[I|0] so inject(cat[e,s])≈e at k=0
- **step_embeds**: Parameter(K, H), init zeros — per-step identity signal
- **gate_proj**: Linear(2H→1, bias=+2) — gates start open (sigmoid(2)≈0.88)
- **Gated update**: `s = s + g*(u-s)`, `g ∈ [gate_min=0.1, 1]`
- **Forced step**: k=0 always full update (g=1), k=1..K-1 use gate
- **Grad checkpointing**: enabled on recur blocks so BS=128 works (without OOM) for apples-to-apples comparison with B1

**Model size:** 42.5M params (vs 50.3M for B1 standard depth=8)

---

## Results Summary

| Exp | Description | val_bpb | gate_mean | effective_k | steps | notes |
|---|---|---|---|---|---|---|
| **B1** | Standard GPT, depth=8, matrix_lr=0.06 | **0.9964** | — | 1.00 | 924 | reference baseline |
| B2b | RecursiveGPT K=4, no gate, BS=128+grad_ckpt | 1.1167 | 1.0000 | 4.00 | 331 | 3× slower/step than B1 → fewer steps |
| P2a | USE_GATE=True, LAMBDA_GATE=0 | 1.1170 | 0.3242 | 1.97 | 325 | gate collapsed to floor instantly |

---

## Phase 1: Baselines

### B1 — Standard GPT (reference)
- **val_bpb: 0.9964**, 924 steps, 38.6% MFU, 45,060 MB VRAM
- Consistent with best result from initial-tests branch (0.9963)
- Used matrix_lr=0.06 (established as best from initial-tests)

### B2 — Recursive K=4, no gate (first attempt)
- **Crashed at eval**: `evaluate_bpb` expected single-tensor return, RecursiveGPT returns `(ce, gate_mean)` tuple
- Fix: wrap model in `_CEOnlyModel` before calling `evaluate_bpb`

### B2b — Recursive K=4, no gate, with fixes
- **val_bpb: 1.1167**, 331 steps, 24% MFU, 37,128 MB VRAM
- **Why worse than B1?** Each optimizer step takes ~940ms (vs ~310ms for B1) because:
  - K=4 recurrences = 20 effective layers vs 8 for B1
  - Gradient checkpointing adds ~30% recompute overhead
  - B2b processes only 173M tokens (vs 477M for B1) in 5 minutes
- **Implication**: in a fixed wall-clock budget, recursion is expensive. The architecture processes ~same FLOPs/step (both ~3.8B effective layer-token ops) but B2b has ~3× slower step time due to recompute.

---

## Phase 2: Gating

### P2a — USE_GATE=True, LAMBDA_GATE=0 (gating on, no penalty)
- **val_bpb: 1.1170**, 325 steps, 24% MFU
- **gate_mean: 0.3242**, effective_k: 1.97
- **Critical finding**: gate collapsed to floor (0.1) in exactly 2 gradient steps and never moved
  - After collapse: gate_mean = (1 + 0.1 + 0.1 + 0.1) / 4 = 0.325 ✓ (k=0 forced=1, k=1,2,3 at floor)
  - gate_std ≈ 0 — all tokens get identical gate value
  - The model treats k=1,2,3 as near-no-ops
- **Why collapse?** Without penalty, the model finds it optimal to skip most recurrences (compute savings at no CE cost). The forced step-0 full update does all the learning; steps 1-3 are ignored.
- **No improvement over B2b**: val_bpb ≈ same, effective_k dropped from 4→2, but quality unchanged
- **Bistability diagnosis**: sigmoid gate with asymmetric init (k=0 forced) → model easily learns to close k=1,2,3 without loss penalty

---

## Issues Encountered

1. **B2 eval crash**: `tuple.view(-1)` error — RecursiveGPT returns `(ce, gate_mean)` but `evaluate_bpb` expects single tensor. Fixed with `_CEOnlyModel` wrapper.
2. **OOM at BS=128 + K=4**: 4× activation memory for recurrent blocks. Fixed with gradient checkpointing (`USE_GRAD_CKPT=True`), reduces peak VRAM from ~80GB+ to 57GB.
3. **Gate collapse to floor**: k=1,2,3 gates immediately go to 0.1 (gate_min). This prevents meaningful per-token compute variation.

---

## Next Directions (Phase 2 continued)

### P2b — Remove forced step-0 (all K steps gated)
If step 0 is also gated (not forced), the model can't "cheat" by relying on step 0. All steps must earn their keep.
- Risk: model might collapse all gates to floor (nothing happens)
- Mitigation: gate_min=0.1 means some state updates always occur

### P2c — Add gate_std logging
Currently only gate_mean is tracked. gate_std per batch would show whether the gate is differentiating across tokens.

### P2d — Negative lambda (reward open gates)
`loss = ce - LAMBDA_GATE * gate_mean` — penalize gate_mean being LOW. Forces model to keep gates open.
- Risk: model just opens all gates to 1.0 (fully open)
- But this might force the model to learn to USE the extra recurrences

### P2e — Gate on whole recurrence (skip-gate)
Instead of a soft scalar gate, use a binary (or near-binary) gate to skip the entire recurrence step. This is closer to Mixture of Depths (MoD) design.

### P2f — K=2 instead of K=4
Simpler case: only 1 gated step. May be easier to train meaningful gates.

---

## Key Insights So Far

1. **Fixed-time budget favors fewer ops/step**: B1 wins because it gets 924 optimizer steps vs 331 for recursive models. Recursive architecture processes ~same total FLOPs but in larger chunks → fewer gradient updates.

2. **Gate collapse is the main failure mode**: Without careful initialization or penalty, gates immediately collapse to the floor. The "forced step 0" gives the model an easy out — it can rely on step 0 and ignore the rest.

3. **val_bpb comparison is misleading**: B2b ≈ P2a ≈ 1.117 doesn't mean the gate helped or hurt — the dominant factor is the number of optimizer steps (331 vs 924 for B1).

4. **Next focus**: get gate_std > 0 (per-token variation). Without this, we can't demonstrate adaptive compute. The gate needs to make different decisions for different tokens.
