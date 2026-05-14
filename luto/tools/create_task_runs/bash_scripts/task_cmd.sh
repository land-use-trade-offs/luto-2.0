#!/bin/bash
# =============================================================================
# task_cmd.sh — run from the task root directory (cwd = task root).
#
# Execution model (bulk-submission safe):
#   1. Source setup_internet.sh  — idempotent relay setup on a DM; exports
#                                   RELAY_PORT / DM_HOST_FOUND / DM_IP_FOUND /
#                                   PROXY_VARS.  Safe to call from many tasks
#                                   in parallel: start_relay.sh skips restart
#                                   if the relay port is already listening.
#   2. qsub a PBS job            — proxy env vars baked in; returns immediately.
#   3. Exit                      — helpers.py can move straight to the next task.
#
# Inside the PBS job (on the compute node):
#   A. Start a tmux relay_monitor session — watches internet, restarts relay on
#      the DM if the connection drops during a long run.
#   B. Run the real task via conda.
#   C. Kill the monitor and exit.
# =============================================================================

# Read the settings_bash file ==> JOB_NAME, QUEUE, NCPUS, MEM, TIME
source luto/settings_bash.py

# SCRIPT_DIR = task run root (where this script and python_script.py live).
# The full luto project is a sub-directory, so setup_internet.sh is not at the
# root — it lives inside the copied bash_scripts tree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_INTERNET="${SCRIPT_DIR}/luto/tools/create_task_runs/bash_scripts/setup_internet.sh"

# PROJECT is the NCI project code (e.g. jk53). It is not written to
# settings_bash.py, so we derive it here from $PROJECT (set in the shell
# environment on Gadi) or fall back to parsing the path.
PROJECT="${PROJECT:-$(echo "${SCRIPT_DIR}" | grep -oP '(?<=/g/data/)[^/]+')}"

# ── Step 1: Set up relay on DM ────────────────────────────────────────────────
# Populates: RELAY_PORT, DM_HOST_FOUND, DM_IP_FOUND, PROXY_VARS.
# Idempotent — concurrent calls from parallel submissions are safe.
source "${SETUP_INTERNET}"

# ── Step 2: Submit PBS job and exit ───────────────────────────────────────────
# The PBS job handles relay monitoring and runs the real task independently.
# This script exits as soon as qsub returns so helpers.py can submit the next task.
SCRIPT_PBS=$(mktemp)

cat << EOF > $SCRIPT_PBS
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l storage=scratch/${PROJECT}+gdata/${PROJECT}
#PBS -l ncpus=${NCPUS}
#PBS -l mem=${MEM}
#PBS -l jobfs=100GB
#PBS -l walltime=${TIME}
#PBS -l wd

${PROXY_VARS}
export JOBLIB_TEMP_FOLDER=\$PBS_JOBFS

# ── A. Start relay monitor in a tmux session on this compute node ─────────────
# The monitor wakes every 60 s, checks internet, and SSHes to the DM to restart
# the relay if it has been killed by the DM's process watchdog.
if [[ -n "${DM_HOST_FOUND}" ]]; then
    bash "${SETUP_INTERNET}" --monitor \
        "${DM_HOST_FOUND}" "${DM_IP_FOUND}" "${RELAY_PORT}"
fi

# ── B. Run the real task ──────────────────────────────────────────────────────
conda run -n luto python "${SCRIPT_DIR}/python_script.py"
TASK_EXIT=\$?

# ── C. Cleanup: kill the relay monitor tmux session ───────────────────────────
tmux kill-session -t relay_monitor 2>/dev/null || true
exit \${TASK_EXIT}
EOF

qsub "${SCRIPT_PBS}"
rm "${SCRIPT_PBS}"