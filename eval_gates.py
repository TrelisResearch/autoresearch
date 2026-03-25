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
    # Import RecursiveGPT from train.py
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "train", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
    # We can't exec train.py fully (it runs training), so import just the classes.
    # Instead, replicate the model construction from the checkpoint keys.
    from train import RecursiveGPT, GPTConfig

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Infer config from state dict
    n_embd = sd["transformer.wte.weight"].shape[1]
    vocab_size = sd["transformer.wte.weight"].shape[0]
    n_rec = sum(1 for k in sd if k.startswith("transformer.rec.") and k.endswith(".attn.c_q.weight"))
    n_pre = sum(1 for k in sd if k.startswith("transformer.pre.") and k.endswith(".attn.c_q.weight"))
    n_cod = sum(1 for k in sd if k.startswith("transformer.cod.") and k.endswith(".attn.c_q.weight"))
    n_head = sd["transformer.pre.0.attn.c_q.weight"].shape[0] // (n_embd // sd["transformer.pre.0.attn.c_q.weight"].shape[1])
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
        n_pre=n_pre, n_rec=n_rec, n_cod=n_cod,
        k_recurse=k_recurse,
        use_gate=use_gate,
        gate_from_prelude=gate_from_prelude,
        gate_from_diff=False,
        gate_min=0.1,
        gate_min_init=0.1,
        step_embed_scale=0.1,
        inject_init="identity",
        lora_rank=0,
        use_grad_ckpt=False,   # no grad ckpt at eval
    )
    model.load_state_dict(sd)
    model.eval()
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

    with torch.no_grad():
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

    tokenizer = Tokenizer()

    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"{'='*70}")
        print(f"Sample {i+1}: {text[:60]}{'...' if len(text)>60 else ''}")
        analyze_text(model, tokenizer, text, k_recurse=k_recurse, device=device)
    print(f"\n{'='*70}")
    print("Done. Green=easy (skipped), Red=hard (kept).")


if __name__ == "__main__":
    main()
