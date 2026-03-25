#!/bin/bash
# deploy_sequence.sh: Deploy P4o → P4p → P4q in sequence after P4n completes.
# Commits: P4o=a98c960, P4p=2614d38, P4q=e974e95 (origin/recursive-gate tip)
set -euo pipefail

REMOTE="root@64.247.201.46"
PORT="10926"
KEY="$HOME/.ssh/runpod_vibe"
WANDB="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"

ssh_cmd() {
    ssh "$REMOTE" -p "$PORT" -i "$KEY" "$@"
}

wait_for_run() {
    local log="$1"
    local run_name="$2"
    echo "[$(date)] Waiting for $run_name to finish (checking $log)..."
    while true; do
        if ssh_cmd "test -f $log && grep -q 'val_bpb:' $log" 2>/dev/null; then
            echo "[$(date)] $run_name finished."
            ssh_cmd "grep 'val_bpb:\|peak_vram_mb:\|final_gate_mean:\|final_gate_std:' $log | tail -10"
            return 0
        fi
        # Check if train.py died without writing val_bpb
        if ssh_cmd "test -f $log" 2>/dev/null && ! ssh_cmd "pgrep -f 'train.py' > /dev/null" 2>/dev/null; then
            echo "[$(date)] WARNING: train.py not running, may have crashed. Check $log"
            ssh_cmd "tail -5 $log" 2>/dev/null
            return 1
        fi
        sleep 60
    done
}

deploy_run() {
    local commit="$1"
    local log="$2"
    local run_name="$3"
    echo "[$(date)] Deploying $run_name (commit $commit)..."
    ssh_cmd "cd /workspace/autoresearch && git fetch trelis && git checkout $commit -- train.py && git checkout trelis/recursive-gate -- eval_gates.py"
    ssh_cmd "WANDB_API_KEY=$WANDB nohup uv run python -u /workspace/autoresearch/train.py > $log 2>&1 &"
    echo "[$(date)] $run_name launched, logging to $log"
}

# Step 1: Wait for P4n to finish
echo "=== Step 1: Wait for P4n (K=2 RANDOM_K 80-min recursive) ==="
P4N_BPB="N/A"
if wait_for_run "/workspace/run_p4n.log" "P4n"; then
    P4N_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4n.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4n val_bpb: $P4N_BPB"
fi

# Step 2: Deploy P4o (standard GPT 80-min) — commit a98c960
echo ""
echo "=== Step 2: Deploy P4o (standard GPT 80-min, commit a98c960) ==="
deploy_run "a98c960" "/workspace/run_p4o.log" "P4o"

# Step 3: Wait for P4o
echo ""
echo "=== Step 3: Wait for P4o (standard GPT 80-min) ==="
P4O_BPB="N/A"
if wait_for_run "/workspace/run_p4o.log" "P4o"; then
    P4O_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4o.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4o val_bpb: $P4O_BPB"
    echo "80-min gap: P4n=$P4N_BPB vs P4o=$P4O_BPB"
fi

# Step 4: Deploy P4p (VAR=0.2 gate, 20-min) — commit 2614d38
echo ""
echo "=== Step 4: Deploy P4p (VAR=0.2 gate 20-min, commit 2614d38) ==="
deploy_run "2614d38" "/workspace/run_p4p.log" "P4p"

# Step 5: Wait for P4p
echo ""
echo "=== Step 5: Wait for P4p (VAR=0.2 gate 20-min) ==="
P4P_BPB="N/A"; P4P_GATE_MEAN="N/A"; P4P_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4p.log" "P4p"; then
    P4P_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4p.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4P_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4p.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4P_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4p.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4p val_bpb=$P4P_BPB  gate_mean=$P4P_GATE_MEAN  gate_std=$P4P_GATE_STD"
    # Run eval_gates on P4p checkpoint if it exists
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4p-var0.2-gate.pt && \
        WANDB_API_KEY=$WANDB nohup uv run python -u /workspace/autoresearch/eval_gates.py \
        /workspace/autoresearch/checkpoint_p4p-var0.2-gate.pt > /workspace/eval_gates_p4p.log 2>&1 &" 2>/dev/null || true
fi

# Step 6: Deploy P4q (VAR=0.3 + LAMBDA=0.15, target 75% skip) — commit e974e95
echo ""
echo "=== Step 6: Deploy P4q (VAR=0.3 + LAMBDA=0.15 target 75% skip, commit e974e95) ==="
deploy_run "e974e95" "/workspace/run_p4q.log" "P4q"

# Step 7: Wait for P4q
echo ""
echo "=== Step 7: Wait for P4q (VAR=0.3 + LAMBDA=0.15 20-min) ==="
P4Q_BPB="N/A"; P4Q_GATE_MEAN="N/A"; P4Q_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4q.log" "P4q"; then
    P4Q_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4q.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4Q_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4q.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4Q_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4q.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4q val_bpb=$P4Q_BPB  gate_mean=$P4Q_GATE_MEAN  gate_std=$P4Q_GATE_STD"
    # Run eval_gates on P4q checkpoint if it exists
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4q-var0.3-lambda0.15.pt && \
        WANDB_API_KEY=$WANDB nohup uv run python -u /workspace/autoresearch/eval_gates.py \
        /workspace/autoresearch/checkpoint_p4q-var0.3-lambda0.15.pt > /workspace/eval_gates_p4q.log 2>&1 &" 2>/dev/null || true
fi

echo ""
echo "=== All runs complete ==="
echo "P4n (K=2 RANDOM_K 80-min):             val_bpb=$P4N_BPB"
echo "P4o (standard GPT 80-min):             val_bpb=$P4O_BPB"
echo "P4p (VAR=0.2 gate 20-min):             val_bpb=$P4P_BPB  gate_mean=$P4P_GATE_MEAN  gate_std=$P4P_GATE_STD"
echo "P4q (VAR=0.3 + LAMBDA=0.15 20-min):   val_bpb=$P4Q_BPB  gate_mean=$P4Q_GATE_MEAN  gate_std=$P4Q_GATE_STD"
