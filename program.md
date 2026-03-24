# Autoresearch Program — Gated Recursive Transformer

## Research Goal

Train a small GPT where the **same set of transformer layers is applied recursively K times**, then show that:
1. More recurrences → lower val_bpb (quality improves with compute)
2. A learned gate can **allocate compute dynamically** — fewer recurrences for easy tokens, more for hard ones
3. This trade-off emerges **from training**, not from manual engineering

The north star: `gate_mean` should correlate with task difficulty, and we should be able to trade off K (recurrences) against val_bpb at inference time.

---

## Architecture

```
prelude (2 layers, run once) → recur block (4 layers, shared weights, run K=4 times) → coda (2 layers, run once)
```

New modules added to standard GPT:
- **inject**: `Linear(2H → H, no bias)` — fuses prelude anchor `e` + current state `s` at each step
  - Init: weight = `[I | 0]` so `inject(cat[e, s]) = e` at start
- **step_embeds**: `Parameter(K, H)` — learned step identity signal added at each recurrence
  - Init: zeros → specialisation emerges through training
- **gate_proj** (Phase 2+): `Linear(2H → 1, bias=True)`, bias=+2 (gates start open)
  - `g = gate_min + (1 - gate_min) * sigmoid(gate_proj(cat[u, s]))`
  - `s = s + g * (u - s)` — step 0 always forced full update

---

## Key Insights from Literature

1. **Step embeddings are non-negotiable** (Universal Transformer, arXiv:2603.21676): without per-step signals, shared weights see identical inputs every recurrence → no specialisation.
2. **Gate bias init = +2** — start open, let CE loss train recur blocks, then learn to close where not needed.
3. **Leaky gate floor** `gate_min=0.1` — prevents dead g=0 absorbing state.
4. **Lambda sweep for gating**: too small → gates stay at 1 (no savings); too large → gates collapse to floor. Sweet spot: gate_mean ≈ 0.4–0.7 with gate_std > 0.
5. **Baseline first**: simple recursion (no gate) must improve val_bpb before adding gating complexity.

---

## Metrics

| Metric | Why |
|---|---|
| `val_bpb` | Primary quality metric |
| `gate_mean` | Average gate openness (1.0 = full compute, 0.1 = floor) |
| `gate_std` | Per-token variance — high = model differentiating easy vs hard |
| `effective_k` | `1 + gate_mean * (K-1)` — average recursions used |

---

## Experiment Plan

### Phase 1: Baselines
```python
# B1 — standard GPT
USE_RECURSIVE = False, DEPTH = 8, MATRIX_LR = 0.06

# B2 — simple recursion, no gating
USE_RECURSIVE = True, K_RECURSE = 4, USE_GATE = False, LAMBDA_GATE = 0.0
```
Key question: does B2 improve val_bpb vs B1?

### Phase 2: Gating
```python
USE_GATE = True, LAMBDA_GATE = 0.0  # gating on, no penalty yet
# then sweep: 1e-4, 5e-4, 1e-3, 2e-3, 5e-3
```

### Phase 3: Emergent trade-off
Test with K=1,2,3,4 at inference — does quality degrade gracefully?

### Phase 4: LoRA per step (later)
`LORA_RANK = 8` — per-step low-rank delta for step specialisation.

---

## Experiment Loop

```bash
cd /workspace/autoresearch
export WANDB_API_KEY=<key>
# Edit train.py hyperparams, then:
git commit -m "expN: description"
uv run python -u train.py > /workspace/run_expN.log 2>&1
grep "^val_bpb:\|^gate_mean:\|^effective_k:" /workspace/run_expN.log
```

## results.tsv columns
`commit | val_bpb | gate_mean | effective_k | peak_vram_mb | status | description`
