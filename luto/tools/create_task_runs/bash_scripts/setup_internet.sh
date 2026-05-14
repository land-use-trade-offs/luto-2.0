#!/bin/bash
# =============================================================================
# setup_internet.sh
#
# Dual-mode relay helper.
#
# ── MODE 1: SOURCE on the login node before qsub ─────────────────────────────
#   source /path/to/setup_internet.sh
#
#   Finds a reachable Gadi DM, copies relay scripts to ~/relay/ on it, starts
#   the relay there, and exports four variables into the caller's shell:
#
#     RELAY_PORT     — port the relay listens on
#     DM_HOST_FOUND  — hostname of the DM where relay was started (or "")
#     DM_IP_FOUND    — IP address of that DM (or "")
#     PROXY_VARS     — export lines ready to embed in a PBS heredoc (or "")
#
# ── MODE 2: RUN inside the PBS job on the compute node ───────────────────────
#   bash /path/to/setup_internet.sh --monitor <DM_HOST> <DM_IP> <RELAY_PORT>
#
#   Starts a detached tmux session 'relay_monitor' on this compute node.
#   The session checks internet every 60 s and SSHes to <DM_HOST> to restart
#   the relay if the connection is lost.
#   Returns immediately; the caller is responsible for killing the session
#   (tmux kill-session -t relay_monitor) after the real task finishes.
#
# =============================================================================

# ── MODE 2: relay monitor tmux session (compute node) ────────────────────────
if [[ "${1:-}" == "--monitor" ]]; then
    DM_HOST="${2:?--monitor requires: <DM_HOST> <DM_IP> <RELAY_PORT>}"
    DM_IP="${3:?}"
    RELAY_PORT="${4:?}"
    TMUX_SESSION="relay_monitor"

    # Kill any stale session from a previous run
    tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true

    # Write the monitor loop to a temp script (avoids shell quoting tangles)
    _MON_SCRIPT=$(mktemp /tmp/relay_monitor.XXXXXX.sh)
    cat > "${_MON_SCRIPT}" << MONITOR_EOF
#!/bin/bash
echo "[relay-monitor] Started on \$(hostname) — watching socks5h://${DM_IP}:${RELAY_PORT} via ${DM_HOST}"
while true; do
    sleep 60
    if ! curl --proxy "socks5h://${DM_IP}:${RELAY_PORT}" \
              --connect-timeout 5 --silent --fail \
              https://www.google.com -o /dev/null 2>/dev/null; then
        echo "[relay-monitor] \$(date): Internet lost — restarting relay on ${DM_HOST} ..."
        ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
            "${DM_HOST}" "bash ~/relay/start_relay.sh ${RELAY_PORT}" || true
        sleep 10
    fi
done
MONITOR_EOF
    chmod +x "${_MON_SCRIPT}"

    tmux new-session -d -s "${TMUX_SESSION}" "bash '${_MON_SCRIPT}'"
    echo "[relay] Monitor tmux session '${TMUX_SESSION}' started on $(hostname)"
    exit 0
fi

# ── MODE 1: set up relay on DM (sourced on login node) ────────────────────────
RELAY_PORT=19080
DM_HOST_FOUND=""
DM_IP_FOUND=""
PROXY_VARS=""

_SI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ConnectTimeout=5  : give up quickly if the DM is unreachable
# ServerAliveInterval/CountMax : kill the SSH if the remote side stops responding
#   max hang time per call = 5 × 6 = 30 s
_SI_SSH="ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
             -o ServerAliveInterval=5 -o ServerAliveCountMax=6"
# Hard cap per DM: mkdir+rsync+start_relay+hostname must all finish within 60 s.
_SI_TIMEOUT=60

# ── Cache: skip SSH loop when relay is already confirmed up ──────────────────
# With n_workers=4 parallel submissions, every task_cmd.sh calls this script.
# Without a cache, all 4 workers SSH to the DM and queue through start_relay.sh's
# flock — effectively serialising the entire submission pipeline.
#
# Simple sentinel-file cache (no subshell, no flock needed because writes are
# atomic via mktemp+mv and the worst case is two workers both do the SSH setup
# on a cold start — harmless since start_relay.sh is idempotent).
#
# Cache expires after 30 min or if the relay port stops responding.
RELAY_CACHE_FILE="/tmp/relay_cache_${RELAY_PORT}_${USER}.env"
RELAY_CACHE_TTL=1800   # seconds

# Check whether the cache is fresh and the relay is reachable
_cache_valid=0
if [[ -f "${RELAY_CACHE_FILE}" ]]; then
    _cache_age=$(( $(date +%s) - $(stat -c %Y "${RELAY_CACHE_FILE}") ))
    if [[ ${_cache_age} -lt ${RELAY_CACHE_TTL} ]]; then
        # Peek at cached IP without fully sourcing yet (avoid polluting variables)
        _peek_ip=$(grep '^DM_IP_FOUND=' "${RELAY_CACHE_FILE}" | head -1 | cut -d= -f2 | tr -d '"')
        if [[ -n "${_peek_ip}" ]] && \
           python3 -c "import socket; socket.create_connection(('${_peek_ip}', ${RELAY_PORT}), 2)" 2>/dev/null; then
            _cache_valid=1
        fi
    fi
fi

if [[ ${_cache_valid} -eq 1 ]]; then
    # Fast path — source cache directly, no SSH needed
    source "${RELAY_CACHE_FILE}"
    echo "==> Relay cache hit: ${DM_HOST_FOUND} (${DM_IP_FOUND}:${RELAY_PORT})"
else
    # Slow path — do the full SSH setup then write/update the cache
    echo "==> Setting up relay on a Gadi data mover ..."
    _found_host="" ; _found_ip=""
    for _dm in gadi-dm-01 gadi-dm-02 gadi-dm-03 gadi-dm-04 gadi-dm-05 gadi-dm-06; do
        timeout ${_SI_TIMEOUT} ${_SI_SSH} "$_dm" 'mkdir -p ~/relay' 2>/dev/null || continue
        timeout ${_SI_TIMEOUT} rsync -az --delete \
            -e "ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
                    -o ServerAliveInterval=5 -o ServerAliveCountMax=6" \
            "${_SI_DIR}/Relay_proxy/" \
            "${_dm}:~/relay/" 2>/dev/null || continue
        timeout ${_SI_TIMEOUT} ${_SI_SSH} "$_dm" "bash ~/relay/start_relay.sh ${RELAY_PORT}" 2>/dev/null || continue
        _found_ip=$(timeout 10 ${_SI_SSH} "$_dm" "hostname -I | awk '{print \$1}'" 2>/dev/null) || continue
        [[ -n "$_found_ip" ]] || continue
        _found_host="$_dm"
        echo "    Relay started on ${_found_host} (${_found_ip}:${RELAY_PORT})"
        break
    done

    if [[ -n "${_found_host}" ]]; then
        # Write cache atomically (mktemp + mv so readers never see a partial file)
        _tmp_cache=$(mktemp "${RELAY_CACHE_FILE}.XXXXXX")
        printf 'DM_HOST_FOUND="%s"\nDM_IP_FOUND="%s"\nRELAY_PORT=%s\n' \
               "${_found_host}" "${_found_ip}" "${RELAY_PORT}" > "${_tmp_cache}"
        printf 'PROXY_VARS="export http_proxy=socks5h://%s:%s\nexport https_proxy=socks5h://%s:%s\nexport HTTP_PROXY=socks5h://%s:%s\nexport HTTPS_PROXY=socks5h://%s:%s\nexport no_proxy=localhost,127.0.0.1,*.nci.org.au\nexport NO_PROXY=localhost,127.0.0.1,*.nci.org.au"\n' \
               "${_found_ip}" "${RELAY_PORT}" \
               "${_found_ip}" "${RELAY_PORT}" \
               "${_found_ip}" "${RELAY_PORT}" \
               "${_found_ip}" "${RELAY_PORT}" >> "${_tmp_cache}"
        mv "${_tmp_cache}" "${RELAY_CACHE_FILE}"
        DM_HOST_FOUND="${_found_host}"
        DM_IP_FOUND="${_found_ip}"
        source "${RELAY_CACHE_FILE}"   # pick up PROXY_VARS
    else
        echo "    WARNING: Could not start relay on any DM. Jobs will run without internet proxy."
    fi
fi

if [[ -z "$DM_HOST_FOUND" ]]; then
    echo "    WARNING: No relay available. Jobs will run without internet proxy."
fi
