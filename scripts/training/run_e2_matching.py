"""Run the E2 matching starvation experiment with a mandatory smoke gate.

The baseline and full-pool runs share seed, data, architecture, and optimizer;
only matching preselection and confidence cost differ.  Every invocation gets a
fresh timestamped root so checkpoints and TensorBoard files cannot mix with an
older experiment.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

COMMON_ARGS = [
    "-m", "scripts.training.train_moe",
    "--weights", "yolo11n.pt",
    "--data-root", "datasets/shanghaitech_AB",
    "--crop-size", "640",
    "--batch-size", "8",
    "--epochs", "100",
    "--hidden-channels", "256",
    "--native-references", "1,4,16",
    "--native-warmup-epochs", "5",
    "--backbone-lr", "1e-4",
    "--head-lr", "1e-3",
    "--weight-decay", "1e-4",
    "--grad-clip", "10",
    "--workers", "4",
    "--val-image-interval", "1",
    "--val-image-count", "4",
    "--val-image-conf", "0.5",
    "--seed", "2026",
    "--expert-index", "2",
    "--freeze-epochs", "3",
]

RUNS = [
    (
        "baseline",
        ["--match-top-k", "2000", "--match-confidence-weight", "0.25"],
    ),
    (
        "full_pool",
        ["--match-top-k", "6400", "--match-confidence-weight", "0.0"],
    ),
]


def _run(command: list[str]) -> subprocess.CompletedProcess:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=os.getcwd())


def _run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E2 matching A/B experiment with a mandatory training smoke test"
    )
    parser.add_argument(
        "--output-root",
        default="runs/e2_matching",
        help="timestamped experiment roots are created below this directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root) / _run_id()
    smoke_dir = root / "smoke"
    root.mkdir(parents=True, exist_ok=False)
    print(f"E2_MATCHING_DRIVER_STARTED root={root}", flush=True)

    smoke_command = [
        sys.executable,
        *COMMON_ARGS,
        "--smoke-test",
        "--save-dir",
        str(smoke_dir),
    ]
    print("===== SMOKE GATE =====", flush=True)
    smoke_result = _run(smoke_command)
    smoke_checkpoint = smoke_dir / "last.pt"
    if smoke_result.returncode != 0 or not smoke_checkpoint.exists():
        print(
            "SMOKE GATE FAILED "
            f"rc={smoke_result.returncode} checkpoint={smoke_checkpoint}",
            flush=True,
        )
        return smoke_result.returncode or 1
    print("SMOKE GATE OK", flush=True)

    failures = 0
    for name, extra in RUNS:
        save_dir = root / name
        command = [
            sys.executable,
            *COMMON_ARGS,
            *extra,
            "--save-dir",
            str(save_dir),
        ]
        print(f"\n===== RUN {name} ({save_dir}) =====", flush=True)
        result = _run(command)
        if result.returncode != 0:
            failures += 1
            print(f"RUN {name} FAILED rc={result.returncode}", flush=True)
        else:
            print(f"RUN {name} OK", flush=True)
    print(
        f"E2_MATCHING_DRIVER_DONE root={root} failures={failures}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
