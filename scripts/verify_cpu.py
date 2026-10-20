#!/usr/bin/env python3
"""Run the bounded, offline CPU verification suites for both snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

TITAN_TESTS = [
    "tests/test_titan_llama_runtime.py",
    "tests/test_titans.py::test_return_surprises",
    "tests/test_titans.py::test_titans_second_order_momentum",
    "tests/test_titans.py::test_neural_mem_chaining_chunks",
    "tests/test_titans.py::test_neural_mem_chaining_with_weight_residual",
    "tests/test_titans.py::test_neural_mem_chaining_with_batch_size",
    "tests/test_titans.py::test_neural_mem_inference",
    "tests/test_titans.py::test_mem_state_detach",
    "tests/test_titans.py::test_titans[True-False-False-1-None-1-False-False-64-True-False-32]",
    "tests/test_titans.py::test_titans[False-False-False-1-None-4-False-False-64-True-False-32]",
    "tests/test_titans.py::test_mac[False-False-None-False-8-False-0-0-1023]",
    "tests/test_titans.py::test_mac[False-True-None-False-8-False-0-0-17]",
    "tests/test_titans.py::test_titans_attn_memory",
    "tests/test_titans.py::test_swiglu_ff_memory",
]

PROJECTS = {
    "titan-llama": {
        "cwd": "implementations/titan-llama",
        "extras": ("research", "test"),
        "tests": TITAN_TESTS,
    },
    "hope": {
        "cwd": "implementations/hope",
        "extras": ("train", "dev"),
        "tests": (),
    },
}

SUMMARY_RE = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|skipped|failed|error)s?\b")
ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?!/)[^\s\"'<>),;]+")


def _public_text(value: str) -> str:
    """Remove machine-local paths from output kept in the portable JSON report."""
    value = value.replace(str(ROOT), "<repo>")
    home = str(Path.home())
    if home and home != "/":
        value = value.replace(home, "<home>")
    value = re.sub(r"file:///[^ \t\r\n\"'<>)]*", "file://<local-path>", value)

    urls: list[str] = []

    def keep_url(match: re.Match[str]) -> str:
        urls.append(match.group(0))
        return f"<url-{len(urls) - 1}>"

    value = re.sub(r"https?://[^\s\"'<>]+", keep_url, value)
    value = ABSOLUTE_PATH_RE.sub("<local-path>", value)
    for index, url in enumerate(urls):
        value = value.replace(f"<url-{index}>", url)
    return value


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except OSError as exc:
        output = f"{type(exc).__name__}: {exc}"
        returncode = 127

    return {
        "command": command,
        "exit_code": returncode,
        "stdout": _public_text(output),
    }


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "skipped": 0, "failed": 0, "errors": 0}
    for match in SUMMARY_RE.finditer(output):
        kind = match.group("kind")
        key = "errors" if kind == "error" else kind
        counts[key] = int(match.group("count"))
    return counts


def _run_project(name: str, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    definition = PROJECTS[name]
    relative_cwd = definition["cwd"]
    cwd = ROOT / relative_cwd
    result: dict[str, object] = {"project": name, "cwd": relative_cwd}
    overall_ok = True

    if args.sync:
        extras = definition["extras"]
        sync_command = ["uv", "sync", "--locked"]
        for extra in extras:
            sync_command.extend(("--extra", extra))
        sync = _run(sync_command, cwd, env)
        result["sync"] = sync
        if sync["exit_code"] != 0:
            result["pytest"] = None
            result["pytest_skipped"] = "uv sync failed"
            return result

    python = ".venv/bin/python"
    test_args = definition["tests"]
    pytest_command = [python, "-m", "pytest", "-q", "-x"]
    if test_args:
        pytest_command.extend(test_args)

    pytest = _run(pytest_command, cwd, env)
    counts = _pytest_counts(str(pytest["stdout"]))
    result["pytest"] = {**pytest, **counts}
    if pytest["exit_code"] != 0:
        overall_ok = False
    result["ok"] = overall_ok
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="run each project's locked uv sync before its tests",
    )
    parser.add_argument(
        "--project",
        choices=("all", "titan-llama", "hope"),
        default="all",
        help="select one project or both (default: both, sequentially)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="write the portable JSON report to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        print("verify_cpu.py supports the locked Linux x86_64 environment only", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("UV_CACHE_DIR", str(ROOT / ".cache" / "uv"))

    selected = tuple(PROJECTS) if args.project == "all" else (args.project,)
    results = [_run_project(name, args, env) for name in selected]
    ok = all(
        result.get("pytest") is not None
        and result["pytest"].get("exit_code") == 0
        and (result.get("sync") is None or result["sync"].get("exit_code") == 0)
        for result in results
    )
    report = {
        "python": sys.version.split()[0],
        "platform": "linux-x86_64",
        "threads": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "projects": results,
        "exit_code": 0 if ok else 1,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
