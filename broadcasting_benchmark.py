#!/usr/bin/env python3
"""
Broadcasting benchmark orchestrator.

Runs 6 strategies in parallel using subprocess (avoids multiprocessing deadlocks).
Per-strategy logs and a final comparison table are saved to ./broadcasting_output/.

Usage:
  python3 broadcasting_benchmark.py [duration_seconds] [seed]

  duration_seconds  default 600  (10 minutes per strategy)
  seed              default 42
"""

import subprocess
import sys
import json
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKER     = os.path.join(SCRIPT_DIR, 'broadcasting_worker.py')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'broadcasting_output')

STRATEGIES = [
    ('non_graph_arrays',      'Non-graph + Arrays'),
    ('non_graph_integers',    'Non-graph + Integers'),
    ('graph_arrays',          'Graph-based + Arrays'),
    ('graph_integers',        'Graph-based + Integers'),
    ('fix_one_side_arrays',   'Fix-one-side + Arrays'),
    ('fix_one_side_integers', 'Fix-one-side + Integers'),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _result_path(key): return os.path.join(OUTPUT_DIR, f'result_{key}.json')
def _log_path(key):    return os.path.join(OUTPUT_DIR, f'log_{key}.txt')


def _last_log_line(key):
    """Return the last non-empty line written to a strategy's log file."""
    try:
        with open(_log_path(key)) as f:
            lines = [l.rstrip() for l in f if l.strip()]
        return lines[-1] if lines else '(no output yet)'
    except Exception:
        return '(log unavailable)'


def _build_table(results):
    W = 86
    rows = []
    rows.append('=' * W)
    rows.append(
        f"{'Strategy':<28} {'Models':>8} {'Unique':>8} "
        f"{'SAT%':>7} {'MPD':>8} {'Entropy (bits)':>15}")
    rows.append('-' * W)
    for key, label in STRATEGIES:
        if key not in results:
            rows.append(f"{label:<28}  (no result — check log for errors)")
            continue
        r = results[key]
        rows.append(
            f"{label:<28} {r['n_models']:>8} {r['n_unique']:>8} "
            f"{r['sat_pct']:>6.2f}% {r['mpd']:>8.3f} {r['entropy']:>15.3f}")
    rows.append('=' * W)
    rows.append('')
    rows.append('Columns')
    rows.append('  Models   – total SAT models generated within the time budget')
    rows.append('  Unique   – distinct input tuples (inactive dims zeroed for comparison)')
    rows.append('  SAT%     – fraction of solver calls that returned SAT')
    rows.append('  MPD      – mean pairwise distance, L1 normalized per feature column')
    rows.append('             (higher = inputs more spread across the joint shape space)')
    rows.append('  Entropy  – average Shannon entropy per feature column, in bits')
    rows.append('             (higher = more variety within each individual feature)')
    return '\n'.join(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    seed     = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    if not os.path.exists(WORKER):
        print(f"Error: worker script not found: {WORKER}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    table_path = os.path.join(OUTPUT_DIR, 'results_table.txt')

    print(f"Broadcasting benchmark: {len(STRATEGIES)} strategies × {duration // 60} min each (parallel)")
    print(f"  Seed={seed}  Output: {OUTPUT_DIR}")
    print('=' * 70)

    # ── launch one subprocess per strategy ───────────────────────────────────
    procs   = []   # (key, label, Popen)
    log_fds = {}   # key -> open file handle

    for key, label in STRATEGIES:
        cmd = [
            sys.executable, WORKER,
            key, str(duration), str(seed), _result_path(key),
        ]
        fd = open(_log_path(key), 'w')
        p  = subprocess.Popen(cmd, stdout=fd, stderr=fd)
        procs.append((key, label, p))
        log_fds[key] = fd
        print(f"  Started [{label}]  pid={p.pid}  log={_log_path(key)}")

    print(f"\nWaiting for all strategies to finish ({duration // 60} min)...")
    print("Status updates every 30 s. Ctrl-C terminates all workers.\n")

    # ── poll until all processes finish ──────────────────────────────────────
    start     = time.time()
    done      = {key: False for key, _ in STRATEGIES}
    next_poll = start + 30

    try:
        while not all(done.values()):
            time.sleep(5)

            for key, label, p in procs:
                if not done[key] and p.poll() is not None:
                    done[key] = True
                    code = p.returncode
                    flag = 'done' if code == 0 else f'exited (code {code})'
                    print(f"  [{label}] {flag}", flush=True)

            now = time.time()
            if now >= next_poll:
                print(f"\n[{int(now - start):3d}s elapsed]", flush=True)
                for key, label, _ in procs:
                    state = 'done   ' if done[key] else 'running'
                    print(f"  {state} [{label}]  {_last_log_line(key)}", flush=True)
                next_poll = now + 30

    except KeyboardInterrupt:
        print('\nInterrupted — terminating workers...')
        for _, _, p in procs:
            p.terminate()
        for _, _, p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    finally:
        for fd in log_fds.values():
            fd.close()

    # ── collect JSON results ──────────────────────────────────────────────────
    results = {}
    for key, label in STRATEGIES:
        path = _result_path(key)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    results[key] = json.load(f)
            except json.JSONDecodeError:
                print(f"  Warning: could not parse result for {key}")
        else:
            print(f"  Warning: no result file for {key} (worker may have crashed)")

    # ── print and save table ──────────────────────────────────────────────────
    header = (
        f"Broadcasting benchmark results\n"
        f"Duration per strategy: {duration}s  Seed: {seed}\n"
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    table  = _build_table(results)
    output = header + '\n' + table

    print('\n' + output)

    with open(table_path, 'w') as f:
        f.write(output + '\n')

    print(f"\nSaved: {table_path}")
    print("Logs:")
    for key, label in STRATEGIES:
        print(f"  {label:<28}  {_log_path(key)}")


if __name__ == '__main__':
    main()
