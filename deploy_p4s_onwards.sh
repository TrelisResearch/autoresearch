#!/bin/bash
# deploy_p4s_onwards.sh: P4s already running. Wait for it, then deploy P4u → P4t.
# Run from LOCAL machine.
# Fixes vs previous script:
#   - Uses git show COMMIT:file > file instead of git checkout (avoids index.lock on MFS)
#   - Crash detection uses kill -0 PID (not pgrep which matches itself)
#   - PID stored in /tmp/train_pid on remote for tracking
set -euo pipefail

REMOTE="root@64.247.201.46"
PORT="10926"
KEY="$HOME/.ssh/runpod_vibe"
WANDB="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"

P4U_COMMIT="781ddb5"  # P4u: K=2 gate_from_postlude0 20-min
P4T_COMMIT="979bfd7"  # P4t: K=4 gate 80-min

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no "$REMOTE" -p "$PORT" -i "$KEY" "$@"
}

wait_for_run() {
    local log="$1"
    local run_name="$2"
    echo "[$(date)] Waiting for $run_name (log: $log)..."
    while true; do
        # Check if run completed (val_bpb: printed)
        if ssh_cmd "strings $log 2>/dev/null | grep -q 'val_bpb:'" 2>/dev/null; then
            echo "[$(date)] $run_name FINISHED."
            ssh_cmd "strings $log 2>/dev/null | grep 'val_bpb:\|peak_vram_mb:\|final_gate_mean:\|final_gate_std:' | tail -5" 2>/dev/null
            return 0
        fi
        # Check for crash: log non-empty AND tracked PID no longer alive
        if ssh_cmd "test -s $log" 2>/dev/null; then
            if ! ssh_cmd "test -f /tmp/train_pid && kill -0 \$(cat /tmp/train_pid) 2>/dev/null" 2>/dev/null; then
                echo "[$(date)] WARNING: train.py not running (PID dead) — $run_name may have crashed."
                ssh_cmd "strings $log 2>/dev/null | tail -5" 2>/dev/null
                return 1
            fi
        fi
        sleep 60
    done
}

deploy_run() {
    local commit="$1"; local log="$2"; local name="$3"
    echo "[$(date)] Deploying $name (commit $commit)..."
    # Use git show instead of git checkout to avoid MFS index.lock issues
    ssh_cmd "cd /workspace/autoresearch && git fetch trelis 2>/dev/null; git show ${commit}:train.py > train.py && git show trelis/recursive-gate:eval_gates.py > eval_gates.py && echo 'files updated'"
    # Launch and save PID for crash detection
    ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB setsid sh -c 'nohup uv run python -u train.py </dev/null > $log 2>&1 & echo \$! > /tmp/train_pid' && sleep 1 && echo 'launched PID:'\$(cat /tmp/train_pid)"
    echo "[$(date)] $name launched → $log"
}

run_eval_gates() {
    local ckpt="$1"; local log="$2"
    ssh_cmd "cd /workspace/autoresearch && test -f $ckpt && setsid sh -c 'WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py $ckpt </dev/null > $log 2>&1 &' && echo eval_gates_launched" 2>/dev/null \
        || echo "  [eval_gates: $ckpt not found or launch failed]"
}

P4S_BPB="N/A"; P4U_BPB="N/A"; P4T_BPB="N/A"

# Step 1: Wait for P4s (already running)
echo "=== [1] Wait for P4s (K=4 gate 20-min, already running) ==="
# Record its PID for crash detection
ssh_cmd "pgrep -f 'python.*train\.py' | head -1 > /tmp/train_pid 2>/dev/null || true" 2>/dev/null
if wait_for_run "/workspace/run_p4s.log" "P4s"; then
    P4S_BPB=$(ssh_cmd "strings /workspace/run_p4s.log 2>/dev/null | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4S_GATE=$(ssh_cmd "strings /workspace/run_p4s.log 2>/dev/null | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4s: val_bpb=$P4S_BPB  $P4S_GATE"
    run_eval_gates "checkpoint_p4s-k4-gate-20min.pt" "/workspace/eval_gates_p4s.log"
fi

# Step 2: Deploy P4u (K=2 gate_from_postlude0 20-min)
echo ""
echo "=== [2] Deploy P4u (K=2 gate_from_postlude0 20-min) ==="
deploy_run "$P4U_COMMIT" "/workspace/run_p4u.log" "P4u"

# Step 3: Wait for P4u
echo ""
echo "=== [3] Wait for P4u ==="
if wait_for_run "/workspace/run_p4u.log" "P4u"; then
    P4U_BPB=$(ssh_cmd "strings /workspace/run_p4u.log 2>/dev/null | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4U_GATE=$(ssh_cmd "strings /workspace/run_p4u.log 2>/dev/null | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4u: val_bpb=$P4U_BPB  $P4U_GATE"
    run_eval_gates "checkpoint_p4u-k2-postlude0-gate.pt" "/workspace/eval_gates_p4u.log"
fi

# Step 4: Deploy P4t (K=4 gate 80-min)
echo ""
echo "=== [4] Deploy P4t (K=4 gate 80-min) ==="
deploy_run "$P4T_COMMIT" "/workspace/run_p4t.log" "P4t"

# Step 5: Wait for P4t
echo ""
echo "=== [5] Wait for P4t (80-min) ==="
if wait_for_run "/workspace/run_p4t.log" "P4t"; then
    P4T_BPB=$(ssh_cmd "strings /workspace/run_p4t.log 2>/dev/null | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4T_GATE=$(ssh_cmd "strings /workspace/run_p4t.log 2>/dev/null | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4t: val_bpb=$P4T_BPB  $P4T_GATE"
    run_eval_gates "checkpoint_p4t-k4-gate-80min.pt" "/workspace/eval_gates_p4t.log"
fi

echo ""
echo "==========================================="
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "==========================================="
echo "P4n (K=2 no-gate 80-min):           val_bpb=0.9254"
echo "P4o (standard GPT 80-min):          val_bpb=0.9244"
echo "P4r (K=2 gate VAR=0.3+L=0.15 80m): val_bpb=0.9326  [+0.007 vs P4n]"
echo "P4s (K=4 gate 20-min):              val_bpb=$P4S_BPB"
echo "P4u (K=2 postlude0 gate 20-min):    val_bpb=$P4U_BPB"
echo "P4t (K=4 gate 80-min):              val_bpb=$P4T_BPB"
