#!/bin/bash
# deploy_p4r_onwards.sh: P4q already running. Wait for it, then deploy P4r → P4s → P4u → P4t.
# Run from LOCAL machine. Uses </dev/null to avoid SSH hanging on background processes.
set -euo pipefail

REMOTE="root@64.247.201.46"
PORT="10926"
KEY="$HOME/.ssh/runpod_vibe"
WANDB="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"

P4R_COMMIT="33b8a1e"  # P4r: K=2 gate 80-min
P4S_COMMIT="e3c092c"  # P4s: K=4 gate 20-min
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
        if ssh_cmd "strings $log | grep -q 'val_bpb:'" 2>/dev/null; then
            echo "[$(date)] $run_name FINISHED."
            ssh_cmd "strings $log | grep 'val_bpb:\|peak_vram_mb:\|final_gate_mean:\|final_gate_std:' | tail -5" 2>/dev/null
            return 0
        fi
        if ssh_cmd "test -s $log" 2>/dev/null && ! ssh_cmd "pgrep -f 'train.py' > /dev/null" 2>/dev/null; then
            echo "[$(date)] WARNING: train.py not running — $run_name may have crashed."
            ssh_cmd "strings $log | tail -5" 2>/dev/null
            return 1
        fi
        sleep 60
    done
}

deploy_run() {
    local commit="$1"; local log="$2"; local name="$3"
    echo "[$(date)] Deploying $name (commit $commit)..."
    ssh_cmd "cd /workspace/autoresearch && git remote prune trelis 2>/dev/null; git fetch trelis && git checkout $commit -- train.py && git checkout trelis/recursive-gate -- eval_gates.py"
    ssh_cmd "cd /workspace/autoresearch && WANDB_API_KEY=$WANDB nohup uv run python -u train.py </dev/null > $log 2>&1 & disown; echo launched"
    echo "[$(date)] $name launched → $log"
}

run_eval_gates() {
    local ckpt="$1"; local log="$2"
    # Use setsid to create new process group — fully detaches from SSH session
    ssh_cmd "cd /workspace/autoresearch && test -f $ckpt && setsid sh -c 'WANDB_API_KEY=$WANDB nohup uv run python -u eval_gates.py $ckpt </dev/null >$log 2>&1 &' && echo eval_gates_launched" 2>/dev/null || echo "  [eval_gates: $ckpt not found or setsid failed]"
}

P4Q_BPB="N/A"; P4R_BPB="N/A"; P4S_BPB="N/A"; P4U_BPB="N/A"; P4T_BPB="N/A"
P4P_BPB="0.967333"  # already complete

# Step 1: Wait for P4q (already running)
echo "=== [1] Wait for P4q (VAR=0.3+LAMBDA=0.15 20-min, already running) ==="
if wait_for_run "/workspace/run_p4q.log" "P4q"; then
    P4Q_BPB=$(ssh_cmd "strings /workspace/run_p4q.log | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4Q_GATE=$(ssh_cmd "strings /workspace/run_p4q.log | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4q: val_bpb=$P4Q_BPB  $P4Q_GATE"
    run_eval_gates "checkpoint_p4q-var0.3-lambda0.15.pt" "/workspace/eval_gates_p4q.log"
fi

# Step 2: Deploy P4r (80-min)
echo ""
echo "=== [2] Deploy P4r (K=2 gate 80-min) ==="
deploy_run "$P4R_COMMIT" "/workspace/run_p4r.log" "P4r"

# Step 3: Wait for P4r
echo ""
echo "=== [3] Wait for P4r (80-min) ==="
if wait_for_run "/workspace/run_p4r.log" "P4r"; then
    P4R_BPB=$(ssh_cmd "strings /workspace/run_p4r.log | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4R_GATE=$(ssh_cmd "strings /workspace/run_p4r.log | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4r: val_bpb=$P4R_BPB  $P4R_GATE"
    run_eval_gates "checkpoint_p4r-gate-80min.pt" "/workspace/eval_gates_p4r.log"
fi

# Step 4: Deploy P4s (K=4 gate 20-min)
echo ""
echo "=== [4] Deploy P4s (K=4 gate 20-min) ==="
deploy_run "$P4S_COMMIT" "/workspace/run_p4s.log" "P4s"

# Step 5: Wait for P4s
echo ""
echo "=== [5] Wait for P4s ==="
if wait_for_run "/workspace/run_p4s.log" "P4s"; then
    P4S_BPB=$(ssh_cmd "strings /workspace/run_p4s.log | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    echo "  P4s: val_bpb=$P4S_BPB"
    run_eval_gates "checkpoint_p4s-k4-gate-20min.pt" "/workspace/eval_gates_p4s.log"
fi

# Step 6: Deploy P4u (K=2 gate_from_postlude0 20-min)
echo ""
echo "=== [6] Deploy P4u (K=2 gate_from_postlude0 20-min) ==="
deploy_run "$P4U_COMMIT" "/workspace/run_p4u.log" "P4u"

# Step 7: Wait for P4u
echo ""
echo "=== [7] Wait for P4u ==="
if wait_for_run "/workspace/run_p4u.log" "P4u"; then
    P4U_BPB=$(ssh_cmd "strings /workspace/run_p4u.log | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4U_GATE=$(ssh_cmd "strings /workspace/run_p4u.log | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4u: val_bpb=$P4U_BPB  $P4U_GATE"
    run_eval_gates "checkpoint_p4u-k2-postlude0-gate.pt" "/workspace/eval_gates_p4u.log"
fi

# Step 8: Deploy P4t (K=4 gate 80-min)
echo ""
echo "=== [8] Deploy P4t (K=4 gate 80-min) ==="
deploy_run "$P4T_COMMIT" "/workspace/run_p4t.log" "P4t"

# Step 9: Wait for P4t
echo ""
echo "=== [9] Wait for P4t (80-min) ==="
if wait_for_run "/workspace/run_p4t.log" "P4t"; then
    P4T_BPB=$(ssh_cmd "strings /workspace/run_p4t.log | grep '^val_bpb:' | tail -1 | awk '{print \$2}'" 2>/dev/null || echo "N/A")
    P4T_GATE=$(ssh_cmd "strings /workspace/run_p4t.log | grep 'final_gate_mean:\|final_gate_std:' | tail -2" 2>/dev/null || echo "")
    echo "  P4t: val_bpb=$P4T_BPB  $P4T_GATE"
    run_eval_gates "checkpoint_p4t-k4-gate-80min.pt" "/workspace/eval_gates_p4t.log"
fi

echo ""
echo "==========================================="
echo "=== ALL EXPERIMENTS COMPLETE ==="
echo "==========================================="
echo "P4n (K=2 no-gate 80-min):           val_bpb=0.9254"
echo "P4o (standard GPT 80-min):          val_bpb=0.9244"
echo "P4p (K=2 gate VAR=0.2 20-min):      val_bpb=$P4P_BPB"
echo "P4q (K=2 gate VAR=0.3+L=0.15 20m): val_bpb=$P4Q_BPB"
echo "P4r (K=2 gate VAR=0.3+L=0.15 80m): val_bpb=$P4R_BPB"
echo "P4s (K=4 gate 20-min):              val_bpb=$P4S_BPB"
echo "P4u (K=2 postlude0 gate 20-min):    val_bpb=$P4U_BPB"
echo "P4t (K=4 gate 80-min):              val_bpb=$P4T_BPB"
