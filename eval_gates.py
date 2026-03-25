"""
eval_gates.py — visualize per-token gate values from a trained gated RecursiveGPT.

Usage:
    uv run eval_gates.py checkpoint_p4l-gated-checkpoint.pt

Loads the checkpoint, runs sample sentences through the model, and prints each
token colored by its gate value:
  - Green (low gate, near floor 0.1) = model skips 2nd recurrence → "easy" token
  - Red   (high gate, near 1.0)      = model keeps 2nd recurrence → "hard" token

This verifies whether the learned gate actually tracks token difficulty.
"""

import sys
import math
import torch
import torch.nn.functional as F
from prepare import Tokenizer, MAX_SEQ_LEN
from train import RecursiveGPT, GPTConfig  # safe: training code guarded by if __name__ == "__main__"

# ANSI color helpers
def color_token(text, gate_val, gate_min=0.1):
    """Color token text by gate value: green=easy (low gate), red=hard (high gate)."""
    # Normalize gate to [0,1]: 0=floor(easy), 1=ceiling(hard)
    norm = (gate_val - gate_min) / (1.0 - gate_min)
    norm = max(0.0, min(1.0, norm))
    if norm < 0.33:
        color = "\033[92m"   # bright green = easy
    elif norm < 0.67:
        color = "\033[93m"   # yellow = medium
    else:
        color = "\033[91m"   # bright red = hard
    reset = "\033[0m"
    return f"{color}{text}{reset}"

def load_model(ckpt_path):
    """Load RecursiveGPT from checkpoint, infer config from state dict."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Infer config from state dict
    n_embd = sd["transformer.wte.weight"].shape[1]
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_rec = sum(1 for k in sd if k.startswith("transformer.rec.") and k.endswith(".attn.c_q.weight"))
    n_pre = sum(1 for k in sd if k.startswith("transformer.pre.") and k.endswith(".attn.c_q.weight"))
    n_cod = sum(1 for k in sd if k.startswith("transformer.cod.") and k.endswith(".attn.c_q.weight"))
    # Infer n_head from ve_gate.weight shape=(n_kv_head, 32); fall back to c_q shape heuristic
    ve_gate_key = next((k for k in sd if "ve_gate.weight" in k), None)
    if ve_gate_key is not None:
        n_head = sd[ve_gate_key].shape[0]
    else:
        # c_q.weight shape = (n_head * head_dim, n_embd); assume HEAD_DIM=128
        n_head = sd["transformer.pre.0.attn.c_q.weight"].shape[0] // 128
    # Infer K from step_embeds
    k_recurse = sd["step_embeds"].shape[0]
    use_gate = "gate_proj.weight" in sd
    # Infer gate variant from gate_proj input size:
    #   gate_proj.weight shape = (1, gate_in_dim)
    #   gate_in_dim == n_embd → gate_from_prelude or gate_from_diff (use prelude-only path)
    #   gate_in_dim == 2*n_embd → gate_from_postlude0 or default cat[u,s] (use postlude0 path)
    gate_from_prelude = False
    gate_from_postlude0 = False
    if use_gate:
        gate_in_dim = sd["gate_proj.weight"].shape[1]
        if gate_in_dim == n_embd:
            gate_from_prelude = True
        else:
            gate_from_postlude0 = True  # cat[e, s_k0] gate

    cfg = GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=n_pre + n_rec + n_cod,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
    )
    model = RecursiveGPT(
        cfg,
        k_recurse=k_recurse,
        use_gate=use_gate,
        gate_from_prelude=gate_from_prelude,
        gate_from_diff=False,
        gate_from_postlude0=gate_from_postlude0,
        gate_min=0.1,
        step_embed_scale=0.1,
        inject_init="identity",
        lora_rank=0,
        use_grad_ckpt=False,
    )
    if use_gate:
        variant = "gate_from_prelude" if gate_from_prelude else ("gate_from_postlude0" if gate_from_postlude0 else "gate_default")
        print(f"  Gate variant: {variant} (gate_proj input dim={gate_in_dim})")
    model.load_state_dict(sd)
    model.eval()
    model.bfloat16()  # FA3 requires bf16/fp16 inputs
    return model, k_recurse

SAMPLE_TEXTS = [
    # Code: should be hard around keywords/operators, easy on whitespace/common words
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    # Math: numbers and operators should be hard
    "The integral of x^2 from 0 to 1 equals 1/3, and the derivative of sin(x) is cos(x).",
    # Simple prose: should be mostly easy
    "The cat sat on the mat. The dog ran in the park.",
    # Rare/technical words: should be hard
    "Mitochondria are the powerhouses of the cell. The endoplasmic reticulum synthesizes proteins.",
    # Numbers and dates: mixed
    "In 1969, Neil Armstrong walked on the Moon at coordinates 0.6741° N, 23.4732° E.",
]

def analyze_text(model, tokenizer, text, k_recurse=2, device="cpu"):
    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_SEQ_LEN - 1:
        tokens = tokens[:MAX_SEQ_LEN - 1]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    # Dummy targets (shifted right)
    y = torch.full_like(x, -1)
    y[0, :-1] = x[0, 1:]

    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device != 'cpu' else torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16)
    with torch.no_grad(), autocast_ctx:
        ce, gate_mean, gate_std, skip_frac, g_prelude = model(
            x, y, reduction='none', return_gate_values=True)

    if g_prelude is None:
        print("  [no gate values — model not gated or only K=1]")
        return

    # g_prelude: (1, T, 1) → (T,)
    gate_vals = g_prelude[0, :, 0].cpu().float().tolist()
    token_ids = tokens

    decoded = []
    for tid in token_ids:
        try:
            tok_str = tokenizer.decode([tid])
        except Exception:
            tok_str = f"[{tid}]"
        decoded.append(tok_str)

    print(f"\n  gate_mean={gate_mean.item():.3f}  gate_std={gate_std.item():.3f}  skip_frac@0.5={sum(g<0.5 for g in gate_vals)/len(gate_vals):.1%}")
    print()

    # Print colored tokens
    line = "  "
    for tok_str, g in zip(decoded, gate_vals):
        line += color_token(repr(tok_str)[1:-1], g)
    print(line)

    # Print gate value bar chart (every 10th token)
    print()
    print("  Gate values (█=high/hard, ░=low/easy):")
    bar_line = "  "
    for i, g in enumerate(gate_vals):
        norm = (g - 0.1) / 0.9
        if norm > 0.75:   bar = "█"
        elif norm > 0.5:  bar = "▓"
        elif norm > 0.25: bar = "░"
        else:              bar = "·"
        bar_line += bar
    print(bar_line)

    # Top-5 hardest and easiest tokens
    indexed = sorted(enumerate(gate_vals), key=lambda x: x[1])
    easiest = indexed[:5]
    hardest = indexed[-5:][::-1]
    print(f"\n  Top-5 easiest (gate near floor {0.1:.2f}):")
    for idx, g in easiest:
        print(f"    [{idx:3d}] g={g:.3f}  {repr(decoded[idx])}")
    print(f"\n  Top-5 hardest (gate near ceiling 1.00):")
    for idx, g in hardest:
        print(f"    [{idx:3d}] g={g:.3f}  {repr(decoded[idx])}")

    # Positional analysis: is gate correlated with position?
    T = len(gate_vals)
    thirds = T // 3
    early_mean  = sum(gate_vals[:thirds]) / max(thirds, 1)
    middle_mean = sum(gate_vals[thirds:2*thirds]) / max(thirds, 1)
    late_mean   = sum(gate_vals[2*thirds:]) / max(T - 2*thirds, 1)
    print(f"\n  Positional gate means:")
    print(f"    early  (pos 0..{thirds-1}):    mean={early_mean:.3f}")
    print(f"    middle (pos {thirds}..{2*thirds-1}): mean={middle_mean:.3f}")
    print(f"    late   (pos {2*thirds}..{T-1}):  mean={late_mean:.3f}")
    pos_var = max(early_mean, middle_mean, late_mean) - min(early_mean, middle_mean, late_mean)
    if pos_var > 0.05:
        print(f"    ** Positional bias detected (range={pos_var:.3f}) — gate varies by position")
    else:
        print(f"    Positional bias small (range={pos_var:.3f}) — gate tracks content, not position")


def positional_analysis(model, tokenizer, device, n_batches=20, batch_size=4, seq_len=512):
    """Run n_batches of val data and plot average gate value by sequence position.

    If gate is purely positional, we'd see a monotonic curve.
    If gate tracks content, the curve should be flat.
    """
    from prepare import make_dataloader
    print(f"\n{'='*70}")
    print("POSITIONAL ANALYSIS: avg gate value by sequence position")
    print(f"Running {n_batches} batches (batch_size={batch_size}, seq_len={seq_len})")

    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val")
    pos_sums = [0.0] * seq_len
    pos_counts = [0] * seq_len

    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    model.eval()
    with torch.no_grad(), autocast_ctx:
        for _ in range(n_batches):
            x, y, _ = next(val_loader)
            x, y = x.to(device), y.to(device)
            _, _, _, _, g_prelude = model(x, y, reduction='none', return_gate_values=True)
            if g_prelude is None:
                print("No gate values available.")
                return
            # g_prelude: (B, T, 1) → (B, T)
            g = g_prelude[:, :, 0].cpu().float()  # (B, T)
            for t in range(g.shape[1]):
                pos_sums[t] += g[:, t].sum().item()
                pos_counts[t] += g.shape[0]

    # Print ASCII bar chart of gate by position (bucketed into 32 bins)
    n_bins = 32
    bin_size = seq_len // n_bins
    bin_means = []
    for b in range(n_bins):
        s = sum(pos_sums[b*bin_size:(b+1)*bin_size])
        c = sum(pos_counts[b*bin_size:(b+1)*bin_size])
        bin_means.append(s / c if c > 0 else 0.0)

    gate_min = min(bin_means)
    gate_max = max(bin_means)
    gate_range = gate_max - gate_min
    print(f"\n  Gate by position (each col = {bin_size} tokens, min={gate_min:.3f}, max={gate_max:.3f}, range={gate_range:.3f})")
    print()

    bar_height = 8
    for row in range(bar_height, 0, -1):
        threshold = gate_min + (row / bar_height) * gate_range
        line = "  "
        for bm in bin_means:
            line += "█" if bm >= threshold else " "
        line += f"  {gate_min + (row/bar_height)*gate_range:.3f}"
        print(line)
    print("  " + "─" * n_bins)
    print(f"  start{' '*int(n_bins/2-5)}end")

    if gate_range > 0.02:
        # Find if it's monotonically increasing/decreasing or peaked
        first_half  = sum(bin_means[:n_bins//2]) / (n_bins//2)
        second_half = sum(bin_means[n_bins//2:]) / (n_bins//2)
        if abs(first_half - second_half) > 0.01:
            direction = "higher at start" if first_half > second_half else "higher at end"
            print(f"\n  ** Positional gradient detected ({direction}): first_half={first_half:.3f}, second_half={second_half:.3f}")
        else:
            print(f"\n  Fluctuating pattern (no clear start/end trend) — gate likely tracks local content")
    else:
        print(f"\n  Gate is FLAT across positions (range={gate_range:.4f}) — gate tracks content type, not position")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run eval_gates.py checkpoint_p4l-gated-checkpoint.pt")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    print(f"Loading checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, k_recurse = load_model(ckpt_path)
    model = model.to(device)
    print(f"Model loaded: K={k_recurse}, gate_from_prelude=True\n")

    tokenizer = Tokenizer.from_directory()

    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"{'='*70}")
        print(f"Sample {i+1}: {text[:60]}{'...' if len(text)>60 else ''}")
        analyze_text(model, tokenizer, text, k_recurse=k_recurse, device=device)
    print(f"\n{'='*70}")
    print("Done. Green=easy (skipped), Red=hard (kept).")

    # Bulk positional analysis over validation data
    positional_analysis(model, tokenizer, device)

    # K-sweep: measure val_bpb at K=1, K=2, and gate-controlled (various thresholds)
    if torch.cuda.is_available():  # requires GPU for evaluate_bpb (uses CUDA token_bytes)
        k_sweep_analysis(model, tokenizer, device, k_recurse)


def gate_loss_correlation(model, tokenizer, device, n_batches=20, batch_size=4, seq_len=512):
    """Directly measure correlation between gate value and per-token CE loss.

    If gate ≈ conditional entropy proxy, high-gate tokens should have higher loss
    (they're harder to predict). Compute Pearson r(gate, loss) and report loss
    bucketed by gate value.
    """
    from prepare import make_dataloader
    print(f"\n{'='*70}")
    print("GATE vs TOKEN LOSS CORRELATION")
    print(f"Testing: do high-gate tokens have higher per-token CE loss?")
    print(f"Running {n_batches} batches (batch_size={batch_size}, seq_len={seq_len})")

    val_loader = make_dataloader(tokenizer, batch_size, seq_len, "val")

    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device != 'cpu' else torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16)
    model.eval()

    all_gates = []
    all_losses = []

    with torch.no_grad(), autocast_ctx:
        for _ in range(n_batches):
            x, y, _ = next(val_loader)
            x, y = x.to(device), y.to(device)
            ce_tokens, gate_mean, gate_std, skip_frac, g_prelude = model(
                x, y, reduction='none', return_gate_values=True)
            if g_prelude is None:
                print("  No gate values available.")
                return
            # ce_tokens: (B*T,) flat — need to align with gate values
            # g_prelude: (B, T, 1)
            # y: (B, T) — -1 for padding
            valid_mask = y.view(-1) != -1
            # gate is (B, T, 1) → (B*T,)
            gate_flat = g_prelude[:, :, 0].reshape(-1)
            # ce_tokens is flat but may include -1 padding positions
            # Only keep valid (non-padding) positions
            # Note: reduction='none' CE still has padding at -1 → 0 loss
            gate_valid = gate_flat[valid_mask].cpu().float()
            loss_valid = ce_tokens.view(-1)[valid_mask].cpu().float()
            all_gates.append(gate_valid)
            all_losses.append(loss_valid)

    all_gates = torch.cat(all_gates)   # (N,)
    all_losses = torch.cat(all_losses)  # (N,)

    # Pearson correlation
    g_mean = all_gates.mean()
    l_mean = all_losses.mean()
    cov = ((all_gates - g_mean) * (all_losses - l_mean)).mean()
    g_std = all_gates.std()
    l_std = all_losses.std()
    pearson_r = (cov / (g_std * l_std + 1e-8)).item()

    print(f"\n  N={len(all_gates):,} token pairs")
    print(f"  gate range: [{all_gates.min():.3f}, {all_gates.max():.3f}]  mean={all_gates.mean():.3f}")
    print(f"  loss range: [{all_losses.min():.3f}, {all_losses.max():.3f}]  mean={all_losses.mean():.3f}")
    print(f"\n  Pearson r(gate, loss) = {pearson_r:.4f}")

    if pearson_r > 0.2:
        print(f"  ** STRONG POSITIVE CORRELATION: high-gate tokens have higher loss")
        print(f"     Gate IS a conditional entropy proxy (confirms information-theoretic interpretation)")
    elif pearson_r > 0.05:
        print(f"  ** WEAK POSITIVE CORRELATION: gate partially tracks token difficulty")
    elif pearson_r > -0.05:
        print(f"  No significant correlation: gate and loss are approximately independent")
    else:
        print(f"  Negative correlation: high-gate tokens tend to be easier (unexpected)")

    # Bucket analysis: loss by gate decile
    n_buckets = 5
    g_min, g_max = all_gates.min().item(), all_gates.max().item()
    g_range = g_max - g_min
    if g_range < 0.01:
        print(f"\n  Gate range too small for bucket analysis (gates are uniform)")
        return

    print(f"\n  Loss by gate bucket:")
    for b in range(n_buckets):
        lo = g_min + b * g_range / n_buckets
        hi = g_min + (b + 1) * g_range / n_buckets
        mask = (all_gates >= lo) & (all_gates < hi)
        if b == n_buckets - 1:
            mask = all_gates >= lo
        if mask.sum() == 0:
            continue
        bucket_loss = all_losses[mask].mean().item()
        bucket_n = mask.sum().item()
        bar = "█" * int(bucket_loss * 5)
        print(f"    gate [{lo:.2f},{hi:.2f}): n={bucket_n:6,}  loss={bucket_loss:.4f}  {bar}")


def k_sweep_analysis(model, tokenizer, device, k_recurse):
    """Measure val_bpb at K=1, K=2, and various gate thresholds.

    This quantifies the value of adaptive gating vs fixed K:
    - K=1: cheapest inference (1 recurrence)
    - K=max: most expensive (k_recurse recurrences)
    - K=gate(τ): adaptive — skips recurrence for easy tokens
    """
    from prepare import evaluate_bpb
    print(f"\n{'='*70}")
    print("K-SWEEP: val_bpb vs inference compute")

    configs = []
    for k in range(1, k_recurse + 1):
        configs.append({"label": f"K={k} (fixed)", "k_override": k, "gate_threshold": 0.0})
    for tau in [0.2, 0.3, 0.5, 0.7]:
        configs.append({"label": f"gate(τ={tau:.1f})", "k_override": None, "gate_threshold": tau})

    results = []
    DEVICE_BATCH_SIZE = 8
    for cfg in configs:
        k_ov = cfg["k_override"]
        tau = cfg["gate_threshold"]

        class _KModel:
            def __call__(self_inner, x, y, reduction='mean'):
                with torch.amp.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.bfloat16):
                    out = model(x, y, reduction=reduction, k_override=k_ov, gate_threshold=tau)
                # RecursiveGPT returns (ce, gate_mean, gate_std, skip_frac) or 5-tuple
                return out[0] if isinstance(out, tuple) else out

        bpb = evaluate_bpb(_KModel(), tokenizer, DEVICE_BATCH_SIZE)
        results.append((cfg["label"], bpb))
        print(f"  {cfg['label']:25s}  bpb={bpb:.4f}")

    print(f"\n  Reference: P4a (K=2 no-gate 20-min)=0.9603, P4b (standard GPT 20-min)=0.9413")
    fixed = [(l, b) for l, b in results if 'fixed' in l]
    gated = [(l, b) for l, b in results if 'gate' in l]
    if fixed:
        print(f"  Best fixed K: {min(b for _, b in fixed):.4f} ({min(fixed, key=lambda x: x[1])[0]})")
    if gated:
        print(f"  Best gated:  {min(b for _, b in gated):.4f} ({min(gated, key=lambda x: x[1])[0]})")


def prefill_vs_decode_analysis(model, tokenizer, device):
    """Compare gate values for 'prefill' (prompt) vs 'decode' (generated) positions.

    In autoregressive inference, prefill processes the prompt and decode generates
    new tokens. The user's intuition: prefill positions may require more recurrence
    than decode positions (prompt establishes hard context; continuation flows naturally).

    Method: take prompt+continuation pairs, run full sequence through model, compare
    gate distributions for prompt positions vs continuation positions.
    """
    PROMPTS = [
        # (prompt, continuation) — prompt is 'prefill', continuation is 'decode'
        (
            "The transformer architecture uses self-attention to",
            " process sequences in parallel. Each attention head learns different relationships between tokens. The feedforward layers then apply nonlinear transformations."
        ),
        (
            "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])",
            "\n    return merge(left, right)\n\ndef merge(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1"
        ),
        (
            "In 1969, Neil Armstrong became the first human to walk on the Moon. The Apollo 11 mission launched on July 16 and",
            " landed in the Sea of Tranquility on July 20. Armstrong's famous words were 'one small step for man, one giant leap for mankind'. The crew returned safely on July 24."
        ),
        (
            "The mitochondria are the powerhouse of the cell. They produce ATP through oxidative phosphorylation. The inner membrane contains",
            " cristae that increase surface area for ATP synthase. The electron transport chain pumps protons across the membrane, creating a gradient that drives ATP production."
        ),
        (
            "x = 5\ny = 10\nz = x + y\nprint(f'Sum: {z}')\n\nfor i in range(10):\n",
            "    if i % 2 == 0:\n        print(f'{i} is even')\n    else:\n        print(f'{i} is odd')\n\nresult = [i**2 for i in range(20) if i % 3 == 0]"
        ),
    ]

    print(f"\n{'='*70}")
    print("PREFILL vs DECODE ANALYSIS")
    print("Do prompt (prefill) positions use more recurrence than continuation (decode) positions?")
    print(f"{'='*70}")

    autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if device != 'cpu' else torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16)

    all_prefill_gates = []
    all_decode_gates = []

    for prompt, continuation in PROMPTS:
        full_text = prompt + continuation
        prompt_tokens = tokenizer.encode(prompt)
        full_tokens = tokenizer.encode(full_text)
        n_prompt = len(prompt_tokens)

        if len(full_tokens) > MAX_SEQ_LEN - 1:
            full_tokens = full_tokens[:MAX_SEQ_LEN - 1]

        x = torch.tensor([full_tokens], dtype=torch.long, device=device)
        y = torch.full_like(x, -1)
        y[0, :-1] = x[0, 1:]

        with torch.no_grad(), autocast_ctx:
            _, _, _, _, g_prelude = model(x, y, reduction='none', return_gate_values=True)

        if g_prelude is None:
            print("  [no gate values — model not gated]")
            return

        gate_vals = g_prelude[0, :, 0].cpu().float().tolist()
        n_seq = len(gate_vals)
        n_decode_start = min(n_prompt, n_seq)

        prefill_gates = gate_vals[:n_decode_start]
        decode_gates = gate_vals[n_decode_start:]

        if not prefill_gates or not decode_gates:
            continue

        pmean = sum(prefill_gates) / len(prefill_gates)
        dmean = sum(decode_gates) / len(decode_gates)
        phigh = sum(g > 0.5 for g in prefill_gates) / len(prefill_gates)
        dhigh = sum(g > 0.5 for g in decode_gates) / len(decode_gates)

        all_prefill_gates.extend(prefill_gates)
        all_decode_gates.extend(decode_gates)

        print(f"\n  Prompt: {repr(prompt[:50])}...")
        print(f"    Prefill ({n_decode_start} tokens):     mean={pmean:.3f}  hard_frac={phigh:.1%}")
        print(f"    Decode  ({len(decode_gates)} tokens): mean={dmean:.3f}  hard_frac={dhigh:.1%}")
        print(f"    Delta (prefill - decode): {pmean - dmean:+.3f}")

        # Visual comparison: first 60 gate values as bars
        bar_prefill = "".join("█" if g > 0.5 else "░" for g in gate_vals[:min(60, n_decode_start)])
        bar_decode = "".join("█" if g > 0.5 else "░" for g in gate_vals[n_decode_start:n_decode_start+min(60, len(decode_gates))])
        print(f"    Prefill gates: |{bar_prefill}|")
        print(f"    Decode  gates: |{bar_decode}|")

    if all_prefill_gates and all_decode_gates:
        pmean_all = sum(all_prefill_gates) / len(all_prefill_gates)
        dmean_all = sum(all_decode_gates) / len(all_decode_gates)
        phigh_all = sum(g > 0.5 for g in all_prefill_gates) / len(all_prefill_gates)
        dhigh_all = sum(g > 0.5 for g in all_decode_gates) / len(all_decode_gates)
        delta = pmean_all - dmean_all

        print(f"\n{'='*70}")
        print(f"AGGREGATE ({len(all_prefill_gates)} prefill, {len(all_decode_gates)} decode tokens):")
        print(f"  Prefill mean gate: {pmean_all:.4f}  hard_frac={phigh_all:.1%}")
        print(f"  Decode  mean gate: {dmean_all:.4f}  hard_frac={dhigh_all:.1%}")
        print(f"  Delta (prefill - decode): {delta:+.4f}")

        if abs(delta) < 0.01:
            print(f"\n  FINDING: Gate is similar in prefill and decode positions (Δ={delta:+.4f})")
            print(f"  Interpretation: Recursion need is content-driven, not context-position-driven.")
            print(f"  Both prompts and continuations have similar proportions of hard/easy tokens.")
        elif delta > 0.01:
            print(f"\n  FINDING: Prefill uses MORE recursion than decode (Δ={delta:+.4f})")
            print(f"  Interpretation: Prompt tokens are harder — they establish ambiguous context.")
            print(f"  Generated tokens flow more naturally (more predictable given context).")
        else:
            print(f"\n  FINDING: Decode uses MORE recursion than prefill (Δ={delta:+.4f})")
            print(f"  Interpretation: Generated tokens are harder — exploring novel continuations.")

        # Information-theoretic interpretation
        print(f"\n  INFORMATION-THEORETIC NOTE:")
        print(f"  Gate ≈ uncertainty proxy. High gate → token has high conditional entropy")
        print(f"  (many plausible continuations). Low gate → token nearly deterministic given context.")
        print(f"  Hard tokens (g≈1.0) have high H(x_t | context) and benefit from extra compute.")
        print(f"  Easy tokens (g≈0.1) are low-entropy: their representation is settled after 1 pass.")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run eval_gates.py checkpoint_p4l-gated-checkpoint.pt")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    print(f"Loading checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, k_recurse = load_model(ckpt_path)
    model = model.to(device)
    print(f"Model loaded: K={k_recurse}, gate_from_prelude=True\n")

    tokenizer = Tokenizer.from_directory()

    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"{'='*70}")
        print(f"Sample {i+1}: {text[:60]}{'...' if len(text)>60 else ''}")
        analyze_text(model, tokenizer, text, k_recurse=k_recurse, device=device)
    print(f"\n{'='*70}")
    print("Done. Green=easy (skipped), Red=hard (kept).")

    # Bulk positional analysis over validation data
    positional_analysis(model, tokenizer, device)

    # Prefill vs decode: does prompt context use more recursion than generated continuation?
    prefill_vs_decode_analysis(model, tokenizer, device)

    # Gate vs token loss correlation: does gate track per-token uncertainty?
    gate_loss_correlation(model, tokenizer, device)

    # K-sweep: measure val_bpb at K=1, K=2, and gate-controlled (various thresholds)
    if torch.cuda.is_available():  # requires GPU for evaluate_bpb (uses CUDA token_bytes)
        k_sweep_analysis(model, tokenizer, device, k_recurse)


if __name__ == "__main__":
    main()
