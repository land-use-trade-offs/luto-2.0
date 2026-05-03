#!/bin/bash
# =============================================================================
# test_internet_keeper.sh
#
# Minimal end-to-end test for the relay + internet keeper.
# Submits a lightweight PBS job that:
#   1. Starts the relay monitor tmux session
#   2. Verifies internet connectivity through the proxy
#   3. Prints results and exits (no real LUTO task is run)
#
# Run from the repo root:
#   bash luto/tools/create_task_runs/bash_scripts/test_internet_keeper.sh
#
# Check results:
#   cat /tmp/test_internet_keeper_out.txt
#   cat /tmp/test_internet_keeper_err.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_INTERNET="${SCRIPT_DIR}/setup_internet.sh"
PROJECT="${PROJECT:-$(echo "${SCRIPT_DIR}" | grep -oP '(?<=/g/data/)[^/]+')}"

echo "================================================================"
echo " test_internet_keeper.sh"
echo " SCRIPT_DIR = ${SCRIPT_DIR}"
echo " PROJECT    = ${PROJECT}"
echo "================================================================"

# ── Step 1: Setup relay on DM ─────────────────────────────────────────────────
echo ""
echo "[$(date +%T)] Step 1: sourcing setup_internet.sh ..."
source "${SETUP_INTERNET}"
echo "[$(date +%T)] Step 1 done. DM_HOST_FOUND='${DM_HOST_FOUND}' DM_IP_FOUND='${DM_IP_FOUND}'"

# ── Step 2: Build and submit a minimal PBS test job ──────────────────────────
SCRIPT_PBS=$(mktemp /tmp/test_keeper.XXXXXX.sh)

cat << EOF > "${SCRIPT_PBS}"
#!/bin/bash
#PBS -N test_inet_keeper
#PBS -q express
#PBS -l storage=gdata/${PROJECT}
#PBS -l ncpus=1
#PBS -l mem=4GB
#PBS -l walltime=00:10:00
#PBS -l wd
#PBS -o /g/data/jk53/jinzhu/LUTO/luto-2.0/test_internet_keeper_out.txt
#PBS -e /g/data/jk53/jinzhu/LUTO/luto-2.0/test_internet_keeper_err.txt

echo "=== PBS job started on \$(hostname) at \$(date) ==="

# ── A. Start relay monitor tmux session ───────────────────────────────────────
echo ""
echo "[A] Starting relay monitor ..."
if [[ -n "${DM_HOST_FOUND}" ]]; then
    bash "${SETUP_INTERNET}" --monitor \
        "${DM_HOST_FOUND}" "${DM_IP_FOUND}" "${RELAY_PORT}"
    echo "[A] Monitor started (or tmux not available — check below)"
else
    echo "[A] SKIP: no DM found, proxy not configured"
fi

# ── B. Test internet connectivity ─────────────────────────────────────────────
echo ""
echo "[B] Testing internet via proxy socks5h://${DM_IP_FOUND}:${RELAY_PORT} ..."
if [[ -n "${DM_IP_FOUND}" ]]; then
    if curl --proxy "socks5h://${DM_IP_FOUND}:${RELAY_PORT}" \
            --connect-timeout 10 --silent --fail \
            https://www.google.com -o /dev/null; then
        echo "[B] PASS — internet reachable through proxy"
    else
        echo "[B] FAIL — curl through proxy failed"
    fi
else
    echo "[B] SKIP — no proxy configured"
fi

# ── C. Test tmux session is alive ─────────────────────────────────────────────
echo ""
echo "[C] Checking relay_monitor tmux session ..."
if tmux has-session -t relay_monitor 2>/dev/null; then
    echo "[C] PASS — relay_monitor tmux session is running"
    tmux list-sessions
else
    echo "[C] WARN — relay_monitor tmux session not found (tmux may be unavailable on compute nodes)"
fi

# ── D. Dummy task (replace with real task in production) ──────────────────────
echo ""
echo "[D] Running dummy task ..."
sleep 5
echo "[D] Dummy task complete"

# ── E. Cleanup ────────────────────────────────────────────────────────────────
echo ""
echo "[E] Killing relay_monitor tmux session ..."
tmux kill-session -t relay_monitor 2>/dev/null && echo "[E] Killed" || echo "[E] Nothing to kill"

echo ""
echo "=== PBS job finished at \$(date) ==="
EOF

echo ""
echo "[$(date +%T)] Step 2: submitting PBS test job ..."
JOB_ID=$(qsub "${SCRIPT_PBS}")
rm "${SCRIPT_PBS}"
echo "[$(date +%T)] Submitted: ${JOB_ID}"
echo ""
echo "Monitor with:  qstat ${JOB_ID}"
echo "Output at:     /g/data/jk53/jinzhu/LUTO/luto-2.0/test_internet_keeper_out.txt"
echo "               /g/data/jk53/jinzhu/LUTO/luto-2.0/test_internet_keeper_err.txt"
echo ""
echo "task_cmd.sh exits here — submission is non-blocking."
