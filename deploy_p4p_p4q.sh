#!/bin/bash
# deploy_p4p_p4q.sh: Wait for P4o, then deploy P4p → P4q
# P4o is already running. This script picks up from "wait for P4o".
set -euo pipefail

REMOTE="root@64.247.201.46"
PORT="10926"
KEY="$HOME/.ssh/runpod_vibe"
WANDB="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"

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no "$REMOTE" -p "$PORT" -i "$KEY" "$@"
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
        # Check crash: log exists, train.py not running
        if ssh_cmd "test -f $log && test -s $log" 2>/dev/null && ! ssh_cmd "pgrep -f 'train.py' > /dev/null" 2>/dev/null; then
            echo "[$(date)] WARNING: train.py not running, $run_name may have crashed."
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
    ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u train.py > $log 2>&1 &"
    echo "[$(date)] $run_name launched, logging to $log"
}

P4N_BPB="0.925392"  # already completed

# Step 1: Wait for P4o (already running)
echo "=== Wait for P4o (standard GPT 80-min, already running) ==="
P4O_BPB="N/A"
if wait_for_run "/workspace/run_p4o.log" "P4o"; then
    P4O_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4o.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "80-min scaling curve: recursive=$P4N_BPB, standard=$P4O_BPB, gap=$(echo "$P4O_BPB - $P4N_BPB" | bc -l 2>/dev/null || echo 'N/A')"
fi

# Step 2: Deploy P4p (VAR=0.2 gate 20-min) — commit 2614d38
echo ""
echo "=== Deploy P4p (VAR=0.2 gate 20-min, commit 2614d38) ==="
deploy_run "2614d38" "/workspace/run_p4p.log" "P4p"

# Step 3: Wait for P4p
echo ""
echo "=== Wait for P4p ==="
P4P_BPB="N/A"; P4P_GATE_MEAN="N/A"; P4P_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4p.log" "P4p"; then
    P4P_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4p.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4P_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4p.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4P_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4p.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4p val_bpb=$P4P_BPB  gate_mean=$P4P_GATE_MEAN  gate_std=$P4P_GATE_STD"
    # Auto-run eval_gates on P4p checkpoint
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4p-var0.2-gate.pt && \
        cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py \
        checkpoint_p4p-var0.2-gate.pt > /workspace/eval_gates_p4p.log 2>&1 &" 2>/dev/null || true
fi

# Step 4: Deploy P4q (VAR=0.3 + LAMBDA=0.15, target 75% skip) — commit e974e95
echo ""
echo "=== Deploy P4q (VAR=0.3 + LAMBDA=0.15 target 75% skip, commit e974e95) ==="
deploy_run "e974e95" "/workspace/run_p4q.log" "P4q"

# Step 5: Wait for P4q
echo ""
echo "=== Wait for P4q ==="
P4Q_BPB="N/A"; P4Q_GATE_MEAN="N/A"; P4Q_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4q.log" "P4q"; then
    P4Q_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4q.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4Q_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4q.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4Q_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4q.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "P4q val_bpb=$P4Q_BPB  gate_mean=$P4Q_GATE_MEAN  gate_std=$P4Q_GATE_STD"
    # Auto-run eval_gates on P4q checkpoint
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4q-var0.3-lambda0.15.pt && \
        cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py \
        checkpoint_p4q-var0.3-lambda0.15.pt > /workspace/eval_gates_p4q.log 2>&1 &" 2>/dev/null || true
fi

echo ""
echo "=== All runs complete ==="
echo "P4n (recursive 80-min):              val_bpb=$P4N_BPB"
echo "P4o (standard GPT 80-min):           val_bpb=$P4O_BPB"
echo "P4p (VAR=0.2 gate 20-min):          val_bpb=$P4P_BPB  gate_mean=$P4P_GATE_MEAN  gate_std=$P4P_GATE_STD"
echo "P4q (VAR=0.3+LAMBDA=0.15 20-min):  val_bpb=$P4Q_BPB  gate_mean=$P4Q_GATE_MEAN  gate_std=$P4Q_GATE_STD"
