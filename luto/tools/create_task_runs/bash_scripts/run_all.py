"""
Launch python_script.py in every Run_G* subdirectory, up to MAX_CONCURRENT at a time.

Usage:
    python run_all.py                                      # run all, 2 at a time
    python run_all.py --max 4                              # run all, 4 at a time
    python run_all.py --max 1                              # sequential
    python run_all.py Run_G0001 Run_G0003                  # specific runs only
    python run_all.py --input_dir /path/to/input           # override INPUT_DIR in every run
    python run_all.py --max 3 --input_dir /path/to/input   # combine both
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def find_run_dirs(names: list[str] | None) -> list[Path]:
    if names:
        dirs = [HERE / n for n in names]
    else:
        dirs = sorted(HERE.glob("Run_G*"))
    missing = [d for d in dirs if not (d / "python_script.py").exists()]
    if missing:
        print(f"[warn] no python_script.py in: {[str(m) for m in missing]}")
    return [d for d in dirs if (d / "python_script.py").exists()]


def run_all(run_dirs: list[Path], max_concurrent: int, input_dir: str | None):
    python = sys.executable
    queue = list(run_dirs)
    active: dict[str, subprocess.Popen] = {}   # run_name -> process
    done, failed = [], []

    print(f"Launching {len(queue)} run(s) with max_concurrent={max_concurrent}\n")

    while queue or active:
        # Start new processes up to the concurrency limit
        while queue and len(active) < max_concurrent:
            run_dir = queue.pop(0)
            name = run_dir.name
            log_path = run_dir / "run.log"
            log_file = open(log_path, "w", encoding="utf-8")
            cmd = [python, "python_script.py"]
            if input_dir:
                cmd += ["--input_dir", input_dir]
            proc = subprocess.Popen(
                cmd,
                cwd=run_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active[name] = (proc, log_file, log_path)
            print(f"[start]  {name}  (pid {proc.pid})  → {log_path}")

        # Poll for completions
        for name in list(active):
            proc, log_file, log_path = active[name]
            if proc.poll() is not None:
                log_file.close()
                if proc.returncode == 0:
                    done.append(name)
                    print(f"[done]   {name}  ✓")
                else:
                    failed.append(name)
                    print(f"[failed] {name}  ✗  (exit {proc.returncode})  see {log_path}")
                del active[name]

        if active:
            time.sleep(5)

    print(f"\n--- Summary ---")
    print(f"Done   ({len(done)}): {done}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="Specific run dirs (default: all Run_G*)")
    parser.add_argument("--max", type=int, default=2, help="Max concurrent runs (default: 2)")
    parser.add_argument("--input_dir", default=None, help="Override INPUT_DIR in every run's luto/settings.py")
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.runs or None)
    if not run_dirs:
        print("No run directories found.")
        sys.exit(1)

    run_all(run_dirs, args.max, args.input_dir)
