"""Launch all pipeline stages on one machine; no training imports needed for help."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import configs
from lib.numeric import parse_float

REPO = Path(__file__).resolve().parent


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Launch one process per pipeline stage on this machine.",
        epilog=(
            "Example: uv run train.py --method fresco --stages 8 --gpus 4 "
            "--lr 1e-3 --method.proj_rank_ratio 0.5\n"
            "Run this launcher once. Detailed --section.field flags are passed "
            "to every stage; uv run main.py --help lists them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--method", "--method.name", choices=configs.PRESETS, default="fresco"
    )
    parser.add_argument(
        "--stages",
        "--pipeline.n_stages",
        type=int,
        default=os.environ.get("NSTAGES", "32"),
        help="pipeline stages (default: NSTAGES or 32)",
    )
    parser.add_argument(
        "--gpus",
        "--pipeline.num_ranks_in_server",
        type=int,
        default=os.environ.get("NGPUS"),
        help="GPUs to use (default: NGPUS or min(8, stages))",
    )
    parser.add_argument(
        "--lr",
        "--lr.schedule.peak_value",
        type=parse_float,
        help="peak learning rate, decimal or fraction (default: 3e-4)",
    )
    parser.add_argument(
        "--steps", "--train.steps", type=int, help="optimizer steps (default: 20000)"
    )
    parser.add_argument(
        "--batch-size",
        "--train.batch_size",
        type=int,
        help="micro-batch size (default: 8)",
    )
    parser.add_argument(
        "--engine",
        "--runtime.engine",
        choices=("graph", "eager"),
        help="default: graph; eager for a single stage",
    )
    parser.add_argument(
        "--wandb", "--logging.wandb", action="store_true", help="log to W&B"
    )
    parser.add_argument(
        "--model", choices=configs.MODEL_PRESETS, default="ngpt_95m_32layer"
    )
    parser.add_argument(
        "--master-port",
        "--pipeline.master_port",
        type=int,
        help="rendezvous port (default: a free port)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="per-rank log directory (default: logs/<timestamp>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands and GPU mapping without starting training",
    )
    args, extra = parser.parse_known_args(argv)
    if args.gpus is None:
        args.gpus = min(8, args.stages)
    if args.stages < 1 or not 1 <= args.gpus <= args.stages:
        parser.error("require 1 <= gpus <= stages")
    if args.stages % args.gpus:
        parser.error("stages must be divisible by gpus")
    spec = configs.MODEL_PRESETS[args.model]
    layout = (
        REPO
        / f"lib/models/{spec.module.split('.')[-1]}/layers={spec.n_layer}/stage{args.stages}.json"
    )
    if not layout.is_file():
        parser.error(
            f"no stage layout for model {args.model} with {args.stages} stages"
        )
    for name in ("steps", "batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.master_port is not None and not 1 <= args.master_port <= 65535:
        parser.error("master-port must be between 1 and 65535")
    for flag in extra:
        if flag.startswith("--pipeline."):
            parser.error(f"{flag.split('=')[0]} is managed by the launcher")
        if flag.startswith("--") and "." not in flag.split("=")[0]:
            parser.error(f"unrecognized argument: {flag}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    args.devices = (
        [device.strip() for device in visible.split(",") if device.strip()]
        if visible is not None
        else [str(i) for i in range(args.gpus)]
    )
    if len(args.devices) < args.gpus or "-1" in args.devices:
        parser.error("CUDA_VISIBLE_DEVICES must expose at least --gpus devices")
    args.devices = args.devices[: args.gpus]
    return args, extra


def stage_commands(args, extra: Sequence[str], port: int) -> list[list[str]]:
    """Build stage commands with common defaults followed by CLI overrides."""
    common = [
        sys.executable,
        "-u",
        str(REPO / "main.py"),
        "--model",
        args.model,
        "--data.name",
        "openwebtext",
        "--method.name",
        args.method,
        "--lr.schedule.peak_value",
        str(args.lr if args.lr is not None else configs.ScheduleConfig().peak_value),
        "--runtime.engine",
        args.engine or ("eager" if args.stages == 1 else "graph"),
    ]
    for name, flag in (
        ("steps", "--train.steps"),
        ("batch_size", "--train.batch_size"),
    ):
        if getattr(args, name) is not None:
            common += [flag, str(getattr(args, name))]
    if args.wandb:
        common.append("--logging.wandb")
    common += list(extra)
    # Rank/device settings belong to the launcher, never to individual workers.
    common += [
        "--pipeline.n_stages",
        str(args.stages),
        "--pipeline.backend",
        "gloo",
        "--pipeline.num_ranks_in_server",
        str(args.gpus),
        "--pipeline.master_addr",
        "localhost",
        "--pipeline.master_port",
        str(port),
        "--pipeline.local_rank",
        "0",
    ]
    return [common + ["--pipeline.rank", str(rank)] for rank in range(args.stages)]


def stop_workers(workers: list[subprocess.Popen]) -> None:
    """Stop the whole run, including workers that are blocked on communication."""
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    deadline = time.monotonic() + 5
    for worker in workers:
        try:
            worker.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


def launch(args, commands: list[list[str]], log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    try:
        for rank, command in enumerate(commands):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.devices[rank % args.gpus]
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            with (log_dir / f"rank{rank}.log").open("w") as log:
                workers.append(
                    subprocess.Popen(
                        command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                )
        while True:
            statuses = [worker.poll() for worker in workers]
            for rank, status in enumerate(statuses):
                if status is not None and status != 0:
                    print(
                        f"Stage {rank} failed (exit {status}). See {log_dir / f'rank{rank}.log'}",
                        file=sys.stderr,
                    )
                    return 1
            if all(status is not None for status in statuses):
                print(f"Training complete. Logs: {log_dir}", flush=True)
                return 0
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping all stages...", file=sys.stderr)
        return 130
    finally:
        stop_workers(workers)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = parse_args(argv)
    port = args.master_port
    if port is None:
        with socket.socket() as sock:
            sock.bind(("localhost", 0))
            port = sock.getsockname()[1]
    commands = stage_commands(args, extra, port)
    log_dir = (
        args.log_dir or REPO / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    ).resolve()
    print(
        f"{args.method}: {args.stages} stages on {args.gpus} GPUs ({args.stages // args.gpus} stages/GPU)",
        flush=True,
    )
    for device_index, device in enumerate(args.devices):
        ranks = ", ".join(
            str(rank) for rank in range(device_index, args.stages, args.gpus)
        )
        print(f"  GPU {device}: stages {ranks}", flush=True)
    print(
        f"Logs: {log_dir}\nFollow training: tail -f {shlex.quote(str(log_dir / f'rank{args.stages - 1}.log'))}",
        flush=True,
    )
    if args.dry_run:
        for rank, command in enumerate(commands):
            print(
                f"CUDA_VISIBLE_DEVICES={shlex.quote(args.devices[rank % args.gpus])} {shlex.join(command)}"
            )
        return 0

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return launch(args, commands, log_dir)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    sys.exit(main())
