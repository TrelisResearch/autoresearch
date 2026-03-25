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
    gate_from_prelude = use_gate  # all our runs use gate_from_prelude=True

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
        gate_min=0.1,
        step_embed_scale=0.1,
        inject_init="identity",
        lora_rank=0,
        use_grad_ckpt=False,
    )
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


if __name__ == "__main__":
    main()
