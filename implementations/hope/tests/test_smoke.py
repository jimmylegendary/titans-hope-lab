"""Smoke test: a short training run must track the released loss trajectory.

Reference (the released run, seed 0 — the recipe is deterministic up to GPU
kernel nondeterminism):
    step 100: bpb 5.083      step 200: bpb 3.673

Random data cannot reproduce corpus loss, so this test requires the
FineWeb-Edu shards (scripts/download_data.py). It runs 200 steps (~10 min
on a modern GPU) and asserts the trajectory lands within a tolerance band —
loose enough for hardware nondeterminism, tight enough to catch any of the
silent-failure bugs documented in docs/STABILITY.md (a frozen memory, a
1/T-scaled write, or a missing retention all blow past it).

    pytest tests/test_smoke.py -q          (skipped if data is missing)
"""

import glob
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.skipif(
    not glob.glob(os.path.join(ROOT, "data/fineweb-edu/**/*.parquet"),
                  recursive=True),
    reason="FineWeb-Edu shards not downloaded")
def test_short_training_trajectory(tmp_path):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts/train_byte.py"),
         "--data", os.path.join(ROOT, "data/fineweb-edu"),
         "--out", str(tmp_path), "--max-steps", "200", "--log-every", "100"],
        capture_output=True, text=True, timeout=3600)
    print(out.stdout[-2000:])
    assert out.returncode == 0, out.stderr[-2000:]

    bpbs = {}
    for line in out.stdout.splitlines():
        if line.startswith("step") and "bpb" in line and "VAL" not in line:
            parts = line.split()
            bpbs[int(parts[1])] = float(parts[parts.index("bpb") + 1])
    assert 100 in bpbs and 200 in bpbs, f"missing log lines: {bpbs}"
    # Reference: 5.083 @ 100, 3.673 @ 200. Band: +/- 0.5 bpb.
    assert abs(bpbs[100] - 5.083) < 0.5, f"step-100 bpb {bpbs[100]} off reference 5.083"
    assert abs(bpbs[200] - 3.673) < 0.5, f"step-200 bpb {bpbs[200]} off reference 3.673"
