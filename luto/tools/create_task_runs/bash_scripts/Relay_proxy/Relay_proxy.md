# Relay Proxy: Internet Access for NCI Compute Nodes via Data Mover

## Background

NCI compute nodes (`gadi-cpu-spr-*`) have **no direct internet access**.  
NCI data mover nodes (`gadi-dm-01` … `gadi-dm-06`) **do have internet access**.

NCI sshd blocks outbound TCP forwarding (`administratively prohibited`) on all node types, so `ssh -L`, `ssh -R`, and `ssh -D` tunnels are not usable. The only working path is a Python TCP relay running as a regular process on the data mover.

---

## Architecture

```
Login node                      Data Mover (gadi-dm-0X)           Internet
──────────────────              ──────────────────────────        ─────────
task_cmd.sh
  └─ setup_internet.sh  ──SSH──▶  start_relay.sh
       (source mode)                  └─ tmux: socks_relay.py (:19080)
       exports PROXY_VARS                        │
                                                 ▼
                                       SOCKS5 proxy (:1080)      ──────▶  pypi.org
                                       (VS Code or ssh -D)                 token.gurobi.com

PBS Compute Node
──────────────────
python_script.py
  (http_proxy=socks5h://DM_IP:19080 baked in by qsub)
```

**Traffic path:**  
`compute node → DM_IP:19080 (socks_relay.py) → localhost:1080 (SOCKS5) → internet`

---

## Files

| File | Location | Purpose |
|------|----------|---------|
| `socks_relay.py` | `Relay_proxy/` | Python TCP relay — listens on `0.0.0.0:19080`, forwards to SOCKS5 `:1080` |
| `start_relay.sh` | `Relay_proxy/` | Starts/verifies relay on the DM inside tmux session `socks_relay`; uses flock for concurrent-safe access |
| `setup_internet.sh` | `bash_scripts/` | Dual-mode helper: (1) sourced on login node to set up relay + cache, (2) `--monitor` mode inside PBS job |
| `task_cmd.sh` | `bash_scripts/` | Per-task PBS submission script; sources `setup_internet.sh` and bakes proxy vars into qsub |

---

## Normal Usage: PBS Job Submission

`task_cmd.sh` is the standard entry point. It is invoked automatically per task by `helpers.py` (via `create_task_runs`):

```bash
# Runs from each task's root directory (copied there by create_task_runs)
bash task_cmd.sh
```

### What it does

**On the login node (fast — exits after qsub):**

1. Sources `luto/settings_bash.py` → reads `JOB_NAME`, `QUEUE`, `NCPUS`, `MEM`, `TIME`
2. Sources `setup_internet.sh` (Mode 1) → exports `RELAY_PORT`, `DM_HOST_FOUND`, `DM_IP_FOUND`, `PROXY_VARS`
3. `qsub` a PBS script with proxy vars baked in → exits immediately

**Inside the PBS job (on compute node):**

A. Starts tmux session `relay_monitor` via `setup_internet.sh --monitor` — wakes every 60 s, restarts relay on DM if internet drops  
B. Runs `conda run -n luto python python_script.py`  
C. Kills `relay_monitor` tmux session; exits with task exit code

---

## setup_internet.sh — Dual-Mode Helper

### Mode 1: Source on login node

```bash
source /path/to/setup_internet.sh
# Exports: RELAY_PORT  DM_HOST_FOUND  DM_IP_FOUND  PROXY_VARS
```

Flow:

1. **Cache check** — reads `/tmp/relay_cache_19080_<USER>.env`
   - If age < 30 min **and** TCP connect to `DM_IP:19080` succeeds → **cache hit**, skip SSH entirely
   - Otherwise → slow path
2. **Slow path** — iterates `gadi-dm-01` … `gadi-dm-06`:
   - `mkdir ~/relay` on DM
   - `rsync` `Relay_proxy/` → `~/relay/` on DM
   - `bash ~/relay/start_relay.sh 19080`
   - Resolves DM IP via `hostname -I`
3. **Writes cache atomically** (mktemp + mv) → subsequent parallel callers get a cache hit

### Mode 2: `--monitor` inside PBS job

```bash
bash setup_internet.sh --monitor <DM_HOST> <DM_IP> <RELAY_PORT>
```

Starts a detached tmux session `relay_monitor` on the compute node. The session loop:

- Sleeps 60 s
- `curl` through the proxy; if it fails → SSH to DM and re-run `start_relay.sh`
- Returns immediately; caller kills it (`tmux kill-session -t relay_monitor`) after the task finishes

---

## Cache Design (Bulk-Submission Safety)

With `n_workers=4` parallel submissions, all workers call `setup_internet.sh` concurrently. Without a cache each would SSH to the DM and queue through `start_relay.sh`'s flock — serialising the pipeline.

```
Worker 1  →  cache MISS  →  SSH setup  →  write /tmp/relay_cache_19080_<USER>.env
Worker 2  →  cache HIT   →  source file  (no SSH)
Worker 3  →  cache HIT   →  source file  (no SSH)
Worker 4  →  cache HIT   →  source file  (no SSH)
```

**Cache invalidation:**

| Condition | Action |
|---|---|
| Cache file missing | Slow path |
| Cache age > 30 min (`RELAY_CACHE_TTL=1800`) | Slow path |
| TCP connect to `DM_IP:19080` fails | Slow path |

**Atomicity:** cache written via `mktemp` + `mv` so readers never see a partial file. Two simultaneous cold-start misses are safe — `start_relay.sh` is idempotent via flock.

---

## start_relay.sh — DM-Side Idempotency

Runs on the data mover. Uses a `flock` file lock so concurrent SSH callers (from multiple parallel workers) serialise safely:

1. Checks SOCKS5 proxy on `:1080` (created by VS Code Remote-SSH); creates one via `ssh -D localhost` if absent
2. Under flock: TCP-tests `:19080` — if relay already listening, returns immediately (skip restart)
3. If not listening: kills any stale `socks_relay` tmux session, starts a fresh one running `socks_relay.py`
4. Waits up to 5 s for the relay to bind, then prints the proxy address

---

## Relay Lifecycle Commands (on the data mover)

```bash
# Check relay status
tmux has-session -t socks_relay && echo running || echo not running

# View relay log
cat /tmp/socks_relay.log

# Attach to relay session
tmux attach -t socks_relay

# Kill relay
tmux kill-session -t socks_relay

# Restart relay
bash ~/relay/start_relay.sh
```

---

## Verify Internet Access from a Compute Node

```bash
# TCP-test the relay port from the login node
python3 -c "import socket; socket.create_connection(('DM_IP', 19080), 2)" && echo ok

# Full HTTP test through the relay
curl --proxy socks5h://DM_IP:19080 --connect-timeout 10 https://pypi.org/ -I 2>&1 | grep HTTP
# Expected: HTTP/2 200
```

---

## Notes

- **DM IP is resolved at submit time** — `setup_internet.sh` resolves `hostname -I` on the DM and bakes the IP into `PROXY_VARS`, which `task_cmd.sh` embeds in the PBS heredoc.
- **SOCKS5 on `:1080`** is created by VS Code Remote-SSH when connected to the DM. If VS Code is not running, `start_relay.sh` creates one automatically via `ssh -D localhost`.
- **Gurobi WLS license** (`~/gurobi.lic`) authenticates over HTTPS to `token.gurobi.com` — requires the proxy to be active.
- **Data mover persistence** — tmux keeps the relay alive across SSH disconnects. If the DM reboots, the next `task_cmd.sh` invocation will miss the cache (TCP check fails) and run the slow path automatically.
- **ACT / NRM constraints** — proxy setup is independent of model settings; it applies to all task runs regardless of scenario.

---

## Why Simpler Approaches Failed

| Approach | Why It Failed |
|---|---|
| Reverse SSH tunnel from DM to compute | `sshd` on compute nodes blocks incoming remote port forwarding |
| Compute SSH to DM with `-L` forward | `sshd` on DM restricts forwarding to external hosts (`administratively prohibited`) |
| Direct SOCKS via compute → DM (`ssh -D`) | Same `administratively prohibited` restriction |
| Expose VS Code SOCKS on `0.0.0.0` | VS Code's `ssh -D` only binds to `localhost` |

The Python relay bypasses all restrictions: it runs on the DM as a plain user process, binds to the network interface, and relays raw TCP bytes to the local SOCKS5 proxy.
