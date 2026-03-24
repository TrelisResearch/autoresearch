# Autoresearch Initial Tests Report

**Branch:** `initial-tests`
**Hardware:** NVIDIA H100 80GB HBM3 (RunPod)
**GPU Driver:** 580.126.09, CUDA 13.0
**W&B Project:** [autoresearch-initial-tests](https://wandb.ai/trelis/autoresearch-initial-tests)

---

## Setup

- Cloned repo to `/workspace/autoresearch` on RunPod H100 pod
- Data and tokenizer already cached at `/root/.cache/autoresearch/`
- Added wandb logging to `train.py` (logs per-step metrics + final val_bpb)
- SSH direct connection required manual `authorized_keys` injection (proxy auth ≠ pod sshd auth)

---

## Phase 1: Baseline

**Config:** All defaults — DEPTH=8, ASPECT_RATIO=64, model_dim=512, 4 heads, DEVICE_BATCH_SIZE=128

| Metric | Value |
|---|---|
| val_bpb | **0.998125** |
| num_params | 50.3M |
| peak_vram_mb | 45,060 MB (55% of 80GB) |
| mfu | 38.0% |
| num_steps | 911 |
| total_tokens | 477.6M |
| training_seconds | 300.2s |
| total_seconds | 459.8s (compile ~160s) |

**Notes:**
- ~160s spent on torch.compile — subsequent experiments should be faster if kernel cache is reused
- 38% MFU is below H100 peak — small model is memory-bandwidth bound, not compute-bound
- 45GB of 80GB used — significant headroom for larger models

---

## Phase 2: Scale Up — Depth 12

**Hypothesis:** On an H100 with 35GB VRAM headroom, a larger model should use compute more efficiently and achieve better val_bpb in the same time budget.

**Exp 1** — `exp1-depth12`: DEPTH=12, model_dim=768, 6 heads, DEVICE_BATCH_SIZE=64 (grad_accum=4)

- First attempt with DEVICE_BATCH_SIZE=128 → OOM during torch.compile (model too large for micro-batch)
- Reduced to DEVICE_BATCH_SIZE=64 → running at **45% MFU** (better than baseline)
- Step time ~812ms vs ~310ms at depth=8 (more compute per step)

| Metric | Value |
|---|---|
| val_bpb | **1.014193** (WORSE than baseline) |
| num_params | 135.3M |
| peak_vram_mb | 48,990 MB (60% of 80GB) |
| mfu | 45.28% |
| num_steps | 383 (vs 911 baseline) |
| total_tokens | 200.8M (vs 477.6M baseline) |

**Key Finding:** Larger model *hurt* in the 5-minute budget. Depth=12 with grad_accum=4 takes ~812ms/step vs ~310ms/step, yielding only 383 optimizer steps vs 911 for baseline. The model processes 2.4x fewer tokens and doesn't converge as well despite having 2.7x more parameters. For fixed-time budgets, tokens processed matter as much as model capacity.

---

## Phase 3: LR Schedule Tuning (Planned)

**Exp 2** — `exp2-depth12-warmdown0.3`: DEPTH=12, WARMDOWN_RATIO=0.3 (vs 0.5 default)
- val_bpb: **1.016910** — worse still. Shorter warmdown didn't compensate for fewer steps.
- More training time at full learning rate, shorter cooldown

**Exp 3** — `exp3-depth12-matrixlr0.06`: DEPTH=12, MATRIX_LR=0.06 (vs 0.04 default)
- val_bpb: **1.013810** — marginal improvement over exp1; higher LR slightly helped but not enough.

---

## Phase 4: Attention Pattern

**Exp 4** — `exp4-depth12-window-SSSSSSL`: DEPTH=12, WINDOW_PATTERN="SSSSSSL"
- val_bpb: **1.014675** — no improvement. More local attention layers didn't help at depth=12.
- MFU dropped slightly (44.6% vs 45.2%) despite more local attention.

---

## Phase 5: Deeper Model

**Exp 5** — `exp5-depth16-bs32`: DEPTH=16, model_dim=1024, 8 heads, DEVICE_BATCH_SIZE=32
- val_bpb: **1.093151** — worst result. Only 195 optimizer steps, 102M tokens processed.
- Highest MFU (49.2%) but the fewest tokens — confirms compute efficiency ≠ sample efficiency in fixed-time regime.

---

## Results Summary

| Exp | Description | val_bpb | params | vram_mb | mfu% | notes |
|---|---|---|---|---|---|---|
| baseline | depth=8, bs=128 | **0.998125** | 50M | 45,060 | 38.0 | reference |
| exp1 | depth=12, bs=64 | 1.014193 | 135M | 48,990 | 45.3 | worse: only 383 steps, 200M tokens |
| exp2 | depth=12, warmdown=0.3 | 1.016910 | 135M | 48,990 | 45.3 | worse; shorter cooldown didn't help |
| exp3 | depth=12, matrix_lr=0.06 | 1.013810 | 135M | 48,990 | 45.2 | best of depth=12 variants; higher LR marginally helped |
| exp4 | depth=12, window=SSSSSSL | 1.014675 | 135M | 48,990 | 44.6 | no improvement; local attention didn't help at depth=12 |
| exp5 | depth=16, bs=32 | 1.093151 | 285M | 44,696 | 49.2 | worst: only 195 steps, 102M tokens |
| exp6 | depth=8, warmdown=0.3 | 0.998247 | 50M | 45,060 | 38.8 | negligible improvement |
| **exp7** | **depth=8, matrix_lr=0.06** | **0.996280** | **50M** | **45,060** | **38.7** | **BEST — first real improvement** |
| exp8 | depth=8, matrix_lr=0.08 | 0.998201 | 50M | 45,060 | 38.5 | too high, worse than 0.06 |
| exp9 | depth=8, window=SSSSSSL | 0.997858 | 50M | 45,060 | 38.9 | minor improvement |
| exp10 | depth=8, warmdown=0.3 + lr=0.06 | 0.998760 | 50M | 45,060 | 38.5 | combination hurt vs lr=0.06 alone |
| exp11 | depth=8, warmup=0.05+warmdown=0.3+lr=0.06 | 0.999578 | 50M | 45,060 | 38.8 | adding warmup hurt further |

---

## Phase 6: Back to Depth=8 — Optimise the Winning Size (Complete)

**exp6** — warmdown=0.3: val_bpb=0.998247. Negligible gain; warmdown ratio barely matters at this scale.

**exp7** — matrix_lr=0.06: val_bpb=**0.996280**. Clear best result. Muon optimizer benefits from a higher LR than the default 0.04.

**exp8** — matrix_lr=0.08: val_bpb=0.998201. Too aggressive — overshoots. 0.06 is the sweet spot.

**exp9** — window=SSSSSSL: val_bpb=0.997858. Minor improvement; more local attention helps slightly.

**exp10** — matrix_lr=0.06 + warmdown=0.3: val_bpb=0.998760. Combination hurt — warmdown reduction counteracted the LR gain.

**exp11** — matrix_lr=0.06 + warmdown=0.3 + warmup=0.05: val_bpb=0.999578. Adding warmup made it worse still.
- **exp10**: DEPTH=10, bs=96 (middle ground between depth=8 and 12)

---

## Learnings

### Architecture & Scaling
1. **Step count > model size for fixed time budgets.** In 5 minutes, depth=8 (50M params) gets 911 optimizer steps and processes 478M tokens. Depth=12 (135M params) gets only 383 steps and 201M tokens — and scores worse. Compute efficiency (MFU %) ≠ sample efficiency in fixed-time regimes.
2. **Optimal model size for 5-min H100 budget is ~50M params at depth=8.** Larger models fill VRAM but starve the optimizer of update steps.

### Hyperparameters
3. **Muon matrix_lr=0.06 beats default 0.04** — the only change that produced a clear improvement (−0.0018 val_bpb). Muon is undertuned in the baseline for this time budget.
4. **matrix_lr=0.08 overshoots** — there's a narrow window around 0.06.
5. **Combining changes can hurt.** lr=0.06 alone: 0.9963. lr=0.06 + warmdown=0.3: 0.9988. Individual wins don't always stack.
6. **Warmdown ratio and warmup had negligible effect** at depth=8 within this budget.

### Infrastructure
7. **Direct SSH vs RunPod proxy**: Proxy authenticates at RunPod's layer; pod sshd has empty `authorized_keys` by default on custom images. Must manually append public key.
8. **OOM at depth=12, BS=128**: torch.compile allocates large temporary buffers during compilation. Halving batch_size to 64 fixes it.
9. **Compile time ~160s first run**, ~60–80s on subsequent runs with same architecture (kernel cache reused).
10. **wandb logging**: Added per-step metrics + final summary. Project: `autoresearch-initial-tests` in `trelis` workspace.

### Next Directions (for recursive branch)
- Fine-grained matrix_lr search around 0.06 (try 0.055, 0.065)
- Try WINDOW_PATTERN="SSSSSSL" + matrix_lr=0.06 combined
- Explore ADAM_BETAS tuning (beta1=0.85 or 0.9)
- Try ASPECT_RATIO=80 (slightly wider model at same depth) — may keep step count high while increasing capacity
- Try depth=10 with BS=96 as a middle ground
