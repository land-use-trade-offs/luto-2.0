#!/bin/bash
# ============================================================
# start_relay.sh
# Run on the DATA MOVER node (e.g. gadi-dm-02).
#
# Starts a TCP relay inside a persistent tmux session so it
# survives SSH disconnects and VS Code session restarts.
#
# Architecture:
#   compute node → DM_IP:LISTEN_PORT (relay) → localhost:SOCKS5_PORT (SOCKS5) → internet
#
# NOTE: NCI login/compute node sshd blocks outbound TCP forwarding
# ("administratively prohibited"), so the relay MUST run on the
# data mover (the only node with unrestricted internet access).
# tmux keeps it alive across session disconnects.
#
# Usage:
#   bash start_relay.sh [LISTEN_PORT] [SOCKS5_PORT]
#
# Defaults:
#   LISTEN_PORT = 19080   (port compute nodes will connect to)
#   SOCKS5_PORT = 1080    (local SOCKS5 proxy port)
# ============================================================

LISTEN_PORT=${1:-19080}
SOCKS5_PORT=${2:-1080}
TMUX_SESSION="socks_relay"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELAY_SCRIPT="${SCRIPT_DIR}/socks_relay.py"

# ── 1. Check the SOCKS5 proxy is available ──────────────────
echo "==> Checking SOCKS5 proxy at localhost:${SOCKS5_PORT} ..."
if ! lsof -i :${SOCKS5_PORT} 2>/dev/null | grep -q LISTEN; then
    echo "    Not found. Creating one via ssh -D ..."
    ssh -D 127.0.0.1:${SOCKS5_PORT} -N -f localhost -o StrictHostKeyChecking=no
    sleep 1
fi
if ! lsof -i :${SOCKS5_PORT} 2>/dev/null | grep -q LISTEN; then
    echo "    ERROR: Could not create SOCKS5 proxy on :${SOCKS5_PORT}"
    exit 1
fi
echo "    OK — SOCKS5 proxy is listening on :${SOCKS5_PORT}"

# ── 2. Start/verify relay under a file lock ──────────────────────────────────
# Multiple concurrent task submissions (parallel joblib workers) all SSH to the
# same DM simultaneously.  Without a lock they race to kill and restart each
# other's tmux session.  flock serialises access so only ONE instance runs the
# check/kill/start block at a time; all others wait and then find the relay
# already listening, skipping the restart entirely.
LOCK_FILE="/tmp/start_relay_${LISTEN_PORT}.lock"
(
  flock -x 200

  if python3 -c "import socket; socket.create_connection(('127.0.0.1', ${LISTEN_PORT}), 1)" 2>/dev/null; then
      echo "==> Relay already listening on :${LISTEN_PORT} — skipping restart."
  else
      # ── 3. Start the relay inside a persistent tmux session ─────
      # Kill any stale tmux session whose relay process has already died.
      if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
          echo "==> Stale tmux session '${TMUX_SESSION}' found (relay not listening) — replacing ..."
          tmux kill-session -t "${TMUX_SESSION}"
          sleep 1
      fi

      echo "==> Starting relay inside tmux session '${TMUX_SESSION}' ..."
      echo "    0.0.0.0:${LISTEN_PORT} -> localhost:${SOCKS5_PORT}"
      tmux new-session -d -s "${TMUX_SESSION}" \
          "python3 '${RELAY_SCRIPT}' ${LISTEN_PORT} ${SOCKS5_PORT} 2>&1 | tee /tmp/socks_relay.log"

      # Wait up to 5 s for the relay to bind and start accepting
      for _i in 1 2 3 4 5; do
          sleep 1
          if python3 -c "import socket; socket.create_connection(('127.0.0.1', ${LISTEN_PORT}), 1)" 2>/dev/null; then
              break
          fi
      done

      if python3 -c "import socket; socket.create_connection(('127.0.0.1', ${LISTEN_PORT}), 1)" 2>/dev/null; then
          echo "    Relay running in tmux session '${TMUX_SESSION}'. Log: /tmp/socks_relay.log"
          echo "    Attach with:  tmux attach -t ${TMUX_SESSION}"
          echo "    Kill with:    tmux kill-session -t ${TMUX_SESSION}"
      else
          echo "    ERROR: Relay failed to start. Check /tmp/socks_relay.log"
          cat /tmp/socks_relay.log 2>/dev/null
          exit 1
      fi
  fi

) 200>"${LOCK_FILE}"

# ── 4. Show connection info ──────────────────────────────────
DM_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "==> Relay is ready."
echo "    Data mover IP : ${DM_IP}"
echo "    Proxy address : socks5h://${DM_IP}:${LISTEN_PORT}"
echo ""
echo "    On the compute node, run:"
echo "      export http_proxy=socks5h://${DM_IP}:${LISTEN_PORT}"
echo "      export https_proxy=socks5h://${DM_IP}:${LISTEN_PORT}"
echo "      export no_proxy=localhost,127.0.0.1,*.nci.org.au"
echo ""
echo "    Or run:  bash setup_compute.sh <COMPUTE_NODE> ${DM_IP} ${LISTEN_PORT}"
