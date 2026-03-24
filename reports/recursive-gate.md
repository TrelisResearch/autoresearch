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

| Exp | Description | val_bpb | gate_mean | gate_std | effective_k | steps | notes |
|---|---|---|---|---|---|---|---|
| **B1** | Standard GPT, depth=8, matrix_lr=0.06 | **0.9964** | — | — | 1.00 | 924 | reference baseline |
| B2b | RecursiveGPT K=4, no gate | 1.1167 | 1.0000 | 0.000 | 4.00 | 331 | 3× slower/step → fewer steps |
| P2a | K=4, gate on, λ=0 | 1.1170 | 0.3242 | 0.000 | 1.97 | 325 | gate at floor (old definition) |
| P2b | K=4, gate on, λ=1e-3 | 1.1147 | 0.1001 | 0.000 | 1.30 | 325 | λ>0 pushes to floor faster |
| P2d | K=2, gate, var_reward=0.01 | 1.0561 | 0.1001 | 0.000 | 1.10 | 540 | K=2 wins on steps; var worked briefly |
| P2e | K=2, gate, var_reward=0.05 | 1.0616 | 0.1001 | 0.000 | 1.10 | 541 | stronger var delayed collapse, slight penalty |
| **P3a** | **K=2, RANDOM_K, no gate** | **1.0463** | 0.000 | 0.000 | 1.00 | **632** | **best recursive; RANDOM_K → more steps** |
| P3b | K=2, RANDOM_K, LoRA=8 | 1.6156 | 0.000 | 0.000 | 1.00 | ~600 | LoRA catastrophically hurt — see analysis |
| P3c | K=2, RANDOM_K, gate on, λ=0 | 1.0441 | 0.1001 | 0.000 | 1.10 | 630 | gate collapsed same as P2a — RANDOM_K helps steps but not gate |
| **P3d** | **K=2, RANDOM_K, gate_from_prelude** | **1.0399** | 0.1001 | 0.000 | 1.10 | **641** | **new best; gate_std=0.16-0.39 early then collapsed** |
| P3e | K=2, RANDOM_K, gate_from_prelude, var_reward=0.05 | TBD | — | — | — | ~640 | running |

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

## Phase 2b: K=2 Experiments (K=4 too slow for 5min budget)

### P2d — K=2, gate_min=0.1, var_reward=0.01
- **val_bpb: 1.056**, 540 steps, 31% MFU
- gate_std showed 0.27-0.32 for early steps 6-80, then collapsed to 0.000
- var_reward=0.01 briefly worked! But CE gradient overwhelmed it after ~80 steps
- **Finding**: K=2 → 540 steps vs K=4 → 331 steps — 63% more optimizer steps → better val_bpb

### P2e — K=2, gate_min=0.1, var_reward=0.05
- **val_bpb: 1.062**, 541 steps, 31% MFU
- gate_std ≈ 0.38-0.41 for steps 0-200, then collapsed to 0.000
- Stronger var_reward delayed collapse but CE gradient still eventually wins
- Slightly worse val_bpb than P2d because var_reward adds gradient noise

---

## Phase 3: Architecture Improvements

### P3a — RANDOM_K=True, K=2, no gate, step_embeds random init
- **val_bpb: 1.046** (new recursive best), 632 steps, 36% MFU
- RANDOM_K randomly samples K ∈ {1, 2} each optimizer step
  - K=1 steps: ~300ms; K=2 steps: ~570ms → average ~450ms vs 570ms fixed
  - 632 steps vs 540 (K=2 fixed) — 17% more steps
  - Higher MFU (36% vs 31%) due to mixed step lengths
- **gate_mean=0, gate_std=0** (USE_GATE=False by design)
- **Finding**: RANDOM_K gives more optimizer steps AND teaches model both K depths

### P3b — RANDOM_K=True + LoRA rank=8 per step (FAILED)
- **val_bpb: 1.616** (catastrophic — far worse than P3a's 1.046)
- Adds K × (2H×8 + 8×H) = 25K params as per-step LoRA on inject
- Hypothesis was: LoRA makes each recurrence step truly distinct → K=2 beats K=1 more clearly for hard tokens
- **Why it failed**: LoRA added optimization complexity that conflicted badly with RANDOM_K
  - With RANDOM_K, lora_B for step k only gets gradient when that step is active (50% of steps for K=2)
  - The combined gradient landscape became very difficult to navigate
  - LoRA adds K separate optimizer states that receive sparse, correlated gradients
  - The model can't efficiently learn both the base inject weight AND per-step LoRA deltas under random K sampling
- **Takeaway**: LoRA per-step and RANDOM_K are incompatible. Don't combine them.

### P3c — RANDOM_K=True + gate on, no LoRA
- **val_bpb: 1.044**, gate collapsed to floor (same as before)
- RANDOM_K + gate: gate_std stays 0 throughout — RANDOM_K doesn't help the gate
- When k_eff=1 is sampled (50% of steps), gate is not trained at all
- When k_eff=2, same (u-s)≈0 collapse as before
- **Takeaway**: RANDOM_K + gate gives same val_bpb as P3a (1.046). Gate doesn't add value here.

### P3d — gate_from_prelude (KEY ARCHITECTURAL FIX)
- **val_bpb: 1.040** (new recursive best), 641 steps, 36.8% MFU
- gate_std: **0.163-0.388 in early steps** (breakthrough! first time gate showed real per-token variation)
- **Why it works**: gate from e (prelude output), not cat[u,s]
  - e is well-defined from prelude (2 transformer layers) → varies across tokens
  - gradient ∂loss/∂gate_proj.weight ∝ e · (u-s) · g*(1-g)
  - (u-s) ≈ step_embeds[k] ≠ 0 even at init (unlike cat[u,s] case)
  - gate can learn token-level difficulty from contextual features
- **Why it still collapsed**: gate_std high at steps 6-30, then drops to 0.01-0.07 by step 40, 0.0 by end
  - CE gradient eventually dominates: model learns that k=1 doesn't improve CE → closes gate
  - step_embeds contribution to (u-s) may shrink as model converges
- **Takeaway**: gate_from_prelude is the right architecture. VAR_REWARD needed to sustain gate_std.

### P3e — gate_from_prelude + VAR_REWARD=0.05 (RUNNING)
- Hypothesis: gate_from_prelude gives real gradient signal; VAR_REWARD=0.05 rewards diversity to sustain gate_std > 0
- Config: K=2, USE_GATE=True, GATE_FROM_PRELUDE=True, RANDOM_K=True, VAR_REWARD=0.05, LAMBDA_GATE=0.0

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

## Seed Ideas (from user + analysis)

1. **RANDOM_K training** (implemented P3a): sample K each step → model learns CE at all depths → natural basis for gate. Key finding: more steps + higher MFU → better val_bpb.

2. **LoRA per step** (P3b): low-rank delta on inject per step → real step specialization. User: "somewhat independent of dynamic compute but may improve val_bpb at low compute addition"

3. **Transformer knowing which recursion it's on** (partially done with step_embeds): changed init from zeros to randn*0.01 so model distinguishes steps from training start. User suggested this helps "better assess cost trade-off with recursions."

4. **Min-one-recursion architecture**: k=0 is mandatory baseline, gate only controls k≥1. This is what the new `gate_mean` definition does (k=0 excluded from gate tracking). Trade-off: k=0 cost is "free" so model can always use at least one recurrence cheaply.

5. **Token-level vs step-level compute**: RANDOM_K trains at different step-level K, but gate is token-level. Future: combine — RANDOM_K for training, gate for inference-time per-token adaptation.

6. **Gate collapse root cause**: at init, inject(cat[e,s])=e (ignores s), so (u-s)≈0 for gated steps → CE gradient through gate ≈ 0 → lambda or noise dominates → instant floor collapse. Fix: LoRA makes (u-s) meaningful from step 0.

7. **Val_bpb vs steps trade-off**: in 5-min budget, B1=924 steps→0.996, P3a=632 steps→1.046, B2b=331 steps→1.117. Strong linear relationship between steps and quality.

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
