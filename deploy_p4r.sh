#!/bin/bash
# deploy_p4r.sh: Deploy P4r (80-min gated VAR=0.3+LAMBDA=0.15) after verifying P4q worked
# Only run this if P4q confirms ~75% skip rate (gate_mean ≈ 0.25)
set -euo pipefail

REMOTE="root@64.247.201.46"
PORT="10926"
KEY="$HOME/.ssh/runpod_vibe"
WANDB="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"
COMMIT="33b8a1e"  # P4r: 80-min gated VAR=0.3+LAMBDA=0.15 + checkpoint for all recursive models

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
            return 1
        fi
        sleep 60
    done
}

echo "=== Deploy P4r (80-min gated K=2, VAR=0.3+LAMBDA=0.15, commit $COMMIT) ==="
echo "Deploying..."
ssh_cmd "cd /workspace/autoresearch && git fetch trelis && git checkout $COMMIT -- train.py && git checkout trelis/recursive-gate -- eval_gates.py"
ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u train.py > /workspace/run_p4r.log 2>&1 &"
echo "[$(date)] P4r launched."

echo ""
echo "=== Wait for P4r (80-min) ==="
P4R_BPB="N/A"; P4R_GATE_MEAN="N/A"; P4R_GATE_STD="N/A"
if wait_for_run "/workspace/run_p4r.log" "P4r"; then
    P4R_BPB=$(ssh_cmd "grep '^val_bpb:' /workspace/run_p4r.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4R_GATE_MEAN=$(ssh_cmd "grep 'final_gate_mean:' /workspace/run_p4r.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4R_GATE_STD=$(ssh_cmd "grep 'final_gate_std:' /workspace/run_p4r.log | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo ""
    echo "=== P4r COMPLETE ==="
    echo "P4r (80-min gated VAR=0.3+L=0.15):  val_bpb=$P4R_BPB  gate_mean=$P4R_GATE_MEAN  gate_std=$P4R_GATE_STD"
    echo "P4n (80-min recursive no-gate):      val_bpb=0.9254"
    echo "P4o (80-min standard GPT):           val_bpb=P4O_RESULT"
    echo "Gated advantage vs no-gate: $(echo "$P4R_BPB $P4N_BPB" | awk '{printf "%+.4f", $1-$2}' 2>/dev/null || echo 'N/A')"
    # Auto-run eval_gates K-sweep on P4r checkpoint
    ssh_cmd "test -f /workspace/autoresearch/checkpoint_p4r-gate-80min.pt && \
        cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py \
        checkpoint_p4r-gate-80min.pt > /workspace/eval_gates_p4r.log 2>&1 &" 2>/dev/null || true

    # Auto-launch P4s → P4t pipeline
    echo ""
    echo "=== Auto-launching P4s → P4t pipeline ==="
    scp -P "$PORT" -i "$KEY" "$(dirname "$0")/deploy_p4s_p4t.sh" "$REMOTE:/workspace/deploy_p4s_p4t.sh" 2>/dev/null || true
    ssh_cmd "chmod +x /workspace/deploy_p4s_p4t.sh && nohup bash /workspace/deploy_p4s_p4t.sh > /workspace/deploy_p4s_p4t.out 2>&1 &" || true
    echo "[$(date)] deploy_p4s_p4t.sh launched."
fi
