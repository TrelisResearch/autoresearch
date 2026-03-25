#!/bin/bash
# deploy_p4s_p4t.sh: Deploy P4s (20-min K=4 gate) then P4t (80-min K=4 gate)
# Run after P4r completes. P4s is a quick ablation; P4t is the long-run if K=4 looks good.
set -euo pipefail

REMOTE="root@64.247.201.46"
PORT="10926"
KEY="$HOME/.ssh/runpod_vibe"
WANDB="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"
P4S_COMMIT="e3c092c"  # P4s: K=4 gated 20-min
P4U_COMMIT="781ddb5"  # P4u: K=2 gate_from_postlude0 20-min ablation
P4T_COMMIT="979bfd7"  # P4t: K=4 gated 80-min

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no "$REMOTE" -p "$PORT" -i "$KEY" "$@"
}

wait_for_run() {
    local log="$1"
    local run_name="$2"
    echo "[$(date)] Waiting for $run_name (checking $log)..."
    while true; do
        if ssh_cmd "test -f $log && grep -q 'val_bpb:' $log" 2>/dev/null; then
            echo "[$(date)] $run_name finished."
            ssh_cmd "grep 'val_bpb:\|peak_vram_mb:\|final_gate_mean:\|final_gate_std:' $log | tail -10"
            return 0
        fi
        if ssh_cmd "test -f $log && test -s $log" 2>/dev/null && ! ssh_cmd "pgrep -f 'train.py' > /dev/null" 2>/dev/null; then
            echo "[$(date)] WARNING: train.py not running. Check $log"
            ssh_cmd "tail -5 $log" 2>/dev/null
            return 1
        fi
        sleep 60
    done
}

# Step 1: Deploy P4s (K=4 gate 20-min)
echo "=== Deploy P4s (K=4 gate 20-min, commit $P4S_COMMIT) ==="
ssh_cmd "cd /workspace/autoresearch && git fetch trelis && git checkout $P4S_COMMIT -- train.py && git checkout trelis/recursive-gate -- eval_gates.py"
ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u train.py > /workspace/run_p4s.log 2>&1 &"
echo "[$(date)] P4s launched."

# Step 2: Wait for P4s
echo ""
echo "=== Wait for P4s ==="
P4S_BPB="N/A"; P4S_GATE_MEAN="N/A"; P4S_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4s.log" "P4s"; then
    P4S_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4s.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4S_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4s.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4S_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4s.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4s val_bpb=$P4S_BPB  gate_mean=$P4S_GATE_MEAN  gate_std=$P4S_GATE_STD"
    # Auto-run eval_gates on P4s checkpoint
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4s-k4-gate-20min.pt && \
        cd /workspace/autoresearch && git checkout trelis/recursive-gate -- eval_gates.py && \
        WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py \
        checkpoint_p4s-k4-gate-20min.pt > /workspace/eval_gates_p4s.log 2>&1 &" 2>/dev/null || true
fi

# Step 3: Deploy P4u (K=2 gate_from_postlude0 20-min ablation)
echo ""
echo "=== Deploy P4u (K=2 gate_from_postlude0 20-min, commit $P4U_COMMIT) ==="
ssh_cmd "cd /workspace/autoresearch && git fetch trelis && git checkout $P4U_COMMIT -- train.py && git checkout trelis/recursive-gate -- eval_gates.py"
ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u train.py > /workspace/run_p4u.log 2>&1 &"
echo "[$(date)] P4u launched."

# Step 4: Wait for P4u
echo ""
echo "=== Wait for P4u ==="
P4U_BPB="N/A"; P4U_GATE_MEAN="N/A"; P4U_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4u.log" "P4u"; then
    P4U_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4u.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4U_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4u.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4U_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4u.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4u val_bpb=$P4U_BPB  gate_mean=$P4U_GATE_MEAN  gate_std=$P4U_GATE_STD"
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4u-k2-postlude0-gate.pt && \
        cd /workspace/autoresearch && git checkout trelis/recursive-gate -- eval_gates.py && \
        WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py \
        checkpoint_p4u-k2-postlude0-gate.pt > /workspace/eval_gates_p4u.log 2>&1 &" 2>/dev/null || true
fi

# Step 5: Deploy P4t (K=4 gate 80-min)
echo ""
echo "=== Deploy P4t (K=4 gate 80-min, commit $P4T_COMMIT) ==="
ssh_cmd "cd /workspace/autoresearch && git fetch trelis && git checkout $P4T_COMMIT -- train.py && git checkout trelis/recursive-gate -- eval_gates.py"
ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u train.py > /workspace/run_p4t.log 2>&1 &"
echo "[$(date)] P4t launched."

# Step 6: Wait for P4t
echo ""
echo "=== Wait for P4t (80-min) ==="
P4T_BPB="N/A"; P4T_GATE_MEAN="N/A"; P4T_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4t.log" "P4t"; then
    P4T_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4t.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4T_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4t.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4T_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4t.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    # Auto-run eval_gates on P4t checkpoint
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4t-k4-gate-80min.pt && \
        cd /workspace/autoresearch && git checkout trelis/recursive-gate -- eval_gates.py && \
        WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py \
        checkpoint_p4t-k4-gate-80min.pt > /workspace/eval_gates_p4t.log 2>&1 &" 2>/dev/null || true
fi

echo ""
echo "=== P4s + P4u + P4t COMPLETE ==="
echo "P4s (K=4 gated 20-min):         val_bpb=$P4S_BPB  gate_mean=$P4S_GATE_MEAN  gate_std=$P4S_GATE_STD"
echo "P4u (K=2 postlude0 20-min):     val_bpb=$P4U_BPB  gate_mean=$P4U_GATE_MEAN  gate_std=$P4U_GATE_STD"
echo "P4t (K=4 gated 80-min):         val_bpb=$P4T_BPB  gate_mean=$P4T_GATE_MEAN  gate_std=$P4T_GATE_STD"
echo ""
echo "Reference:"
echo "  P4n (K=2 no-gate 80-min):         val_bpb=0.9254"
echo "  P4r (K=2 gate 80-min, committed): val_bpb=TBD"
echo "  P4s (K=4 gate 20-min):            val_bpb=$P4S_BPB"
