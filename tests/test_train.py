"""Launcher checks that run without CUDA or the training dependencies."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import train


def flag_value(command, flag):
    return command[command.index(flag) + 1]


class LauncherConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_short_options_and_advanced_overrides_reach_every_stage(self):
        args, extra = train.parse_args(
            [
                "--method",
                "fresco",
                "--stages",
                "8",
                "--gpus",
                "4",
                "--lr",
                "1e-3",
                "--steps",
                "2000",
                "--batch-size",
                "4",
                "--method.proj_rank_ratio",
                "0.5",
                "--train.seed=42",
                "--checkpoint.dir",
                "checkpoints/my run",
                "--wandb",
            ]
        )
        commands = train.stage_commands(args, extra, 12345)
        self.assertEqual(len(commands), 8)
        for rank, command in enumerate(commands):
            self.assertEqual(flag_value(command, "--pipeline.rank"), str(rank))
            self.assertEqual(flag_value(command, "--pipeline.master_port"), "12345")
            self.assertEqual(flag_value(command, "--pipeline.local_rank"), "0")
            self.assertEqual(flag_value(command, "--pipeline.num_ranks_in_server"), "4")
            self.assertEqual(flag_value(command, "--lr.schedule.peak_value"), "0.001")
            self.assertEqual(flag_value(command, "--train.steps"), "2000")
            self.assertEqual(flag_value(command, "--train.batch_size"), "4")
            self.assertEqual(flag_value(command, "--method.proj_rank_ratio"), "0.5")
            self.assertEqual(
                flag_value(command, "--checkpoint.dir"), "checkpoints/my run"
            )
            self.assertIn("--train.seed=42", command)
            self.assertIn("--logging.wandb", command)

    def test_legacy_flags_environment_and_explicit_override(self):
        os.environ.update(NSTAGES="8", NGPUS="4")
        args, extra = train.parse_args(
            [
                "--method.name=recovery",
                "--lr.schedule.peak_value=1e-4",
                "--lr",
                "0.002",
                "--train.steps=50",
                "--runtime.engine=eager",
                "--logging.wandb",
                "--logging.wandb_project=fresco",
            ]
        )
        self.assertEqual((args.stages, args.gpus), (8, 4))
        command = train.stage_commands(args, extra, 12345)[0]
        self.assertEqual(flag_value(command, "--method.name"), "recovery")
        self.assertEqual(flag_value(command, "--lr.schedule.peak_value"), "0.002")
        self.assertEqual(flag_value(command, "--train.steps"), "50")
        self.assertEqual(flag_value(command, "--runtime.engine"), "eager")
        self.assertIn("--logging.wandb_project=fresco", command)

    def test_method_defaults_and_single_stage_engine(self):
        for method in train.configs.PRESETS:
            with self.subTest(method=method):
                args, extra = train.parse_args(["--method", method])
                command = train.stage_commands(args, extra, 12345)[0]
                self.assertEqual(
                    flag_value(command, "--lr.schedule.peak_value"),
                    "0.0003",
                )
                self.assertNotIn("--optimizer.momentum", command)
        args, extra = train.parse_args(["--stages", "1"])
        self.assertEqual(args.gpus, 1)
        self.assertEqual(
            flag_value(train.stage_commands(args, extra, 12345)[0], "--runtime.engine"),
            "eager",
        )

    def test_invalid_launch_settings_fail_before_starting_workers(self):
        for argv in (
            ["--stages", "3"],
            ["--gpus", "0"],
            ["--gpus", "3"],
            ["--stages", "4", "--gpus", "8"],
            ["--steps", "0"],
            ["--master-port", "70000"],
            ["--gpu", "4"],
            ["--pipeline.rank", "2"],
            ["--pipeline.backend=nccl"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.parse_args(argv)
                self.assertEqual(error.exception.code, 2)

    def test_device_selection_and_insufficient_visible_gpus(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "2, 5, GPU-uuid"
        args, _ = train.parse_args(["--stages", "8", "--gpus", "2"])
        self.assertEqual(args.devices, ["2", "5"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.parse_args(["--stages", "8", "--gpus", "4"])

    def test_dry_run_reports_mapping_without_starting_workers_or_writing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            output = io.StringIO()
            with (
                mock.patch.object(train.subprocess, "Popen") as popen,
                contextlib.redirect_stdout(output),
            ):
                code = train.main(
                    [
                        "--stages",
                        "8",
                        "--gpus",
                        "4",
                        "--dry-run",
                        "--log-dir",
                        str(logs),
                    ]
                )
            self.assertEqual(code, 0)
            popen.assert_not_called()
            self.assertFalse(logs.exists())
            self.assertIn("GPU 0: stages 0, 4", output.getvalue())
            self.assertIn("rank7.log", output.getvalue())

    def test_shell_entry_points_work_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            shim = Path(tmp) / "uv"
            shim.write_text("#!/bin/bash\nprintf '%s\\0' \"$@\"\n")
            shim.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.defpath)
            scripts = [
                *sorted(
                    path
                    for path in (train.REPO / "scripts").glob("*.sh")
                    if path.name != "sweep.sh"
                ),
            ]
            for script in scripts:
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [
                            "bash",
                            str(script),
                            "--stages",
                            "8",
                            "--gpus",
                            "4",
                            "--lr",
                            "0.002",
                            "--checkpoint.dir",
                            "path with spaces",
                        ],
                        cwd=tmp,
                        env=env,
                        check=True,
                        capture_output=True,
                    )
                    forwarded = result.stdout.decode().rstrip("\0").split("\0")
                    self.assertEqual(
                        forwarded[:5],
                        [
                            "run",
                            "--project",
                            str(train.REPO),
                            "python",
                            str(train.REPO / "train.py"),
                        ],
                    )
                    args, extra = train.parse_args(forwarded[5:])
                    expected = {
                        "train": "fresco",
                        "pipeoptim": "pipeoptim_nostash",
                    }.get(script.stem, script.stem)
                    self.assertEqual(args.method, expected)
                    self.assertEqual((args.stages, args.gpus, args.lr), (8, 4, 0.002))
                    self.assertEqual(
                        flag_value(extra, "--checkpoint.dir"), "path with spaces"
                    )

    def test_sweep_runner_selects_config_and_runs_from_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            shim = Path(tmp) / "uv"
            shim.write_text('#!/bin/bash\nprintf \'%s\\0\' "$PWD" "$@"\n')
            shim.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.defpath)
            script = train.REPO / "scripts/sweep.sh"
            for name in ("fresco", "methods", "recovery", "sync"):
                with self.subTest(name=name):
                    result = subprocess.run(
                        ["bash", str(script), name, "--entity", "test entity"],
                        cwd=tmp,
                        env=env,
                        check=True,
                        capture_output=True,
                    )
                    forwarded = result.stdout.decode().rstrip("\0").split("\0")
                    self.assertEqual(
                        forwarded,
                        [
                            str(train.REPO),
                            "run",
                            "wandb",
                            "sweep",
                            "--project",
                            "fresco",
                            str(train.REPO / f"scripts/sweep_{name}.yaml"),
                            "--entity",
                            "test entity",
                        ],
                    )
            for option, expected_code in (("--help", 0), ("unknown", 2)):
                result = subprocess.run(
                    ["bash", str(script), option],
                    cwd=tmp,
                    env=env,
                    check=False,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, expected_code)


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name) / "logs"
        self.worker = Path(self.tmp.name) / "worker.py"
        self.worker.write_text(
            "import json, os, sys, time\n"
            "rank = int(sys.argv[1])\n"
            "print(json.dumps({'rank': rank, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'cwd': os.getcwd()}), flush=True)\n"
            "if sys.argv[2] == 'fail':\n"
            "    if rank == 1: sys.exit(7)\n"
            "    time.sleep(30)\n"
        )
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,5"}):
            self.args, _ = train.parse_args(["--stages", "4", "--gpus", "2"])
        self.workers = []
        self.real_popen = subprocess.Popen

    def spawn(self, *args, **kwargs):
        worker = self.real_popen(*args, **kwargs)
        self.workers.append(worker)
        return worker

    def run_workers(self, mode):
        commands = [
            [sys.executable, str(self.worker), str(rank), mode] for rank in range(4)
        ]
        with (
            mock.patch.object(train.subprocess, "Popen", side_effect=self.spawn),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return train.launch(self.args, commands, self.log_dir)

    def test_success_writes_each_rank_log_and_respects_gpu_mapping(self):
        self.assertEqual(self.run_workers("success"), 0)
        for rank in range(4):
            result = json.loads((self.log_dir / f"rank{rank}.log").read_text())
            self.assertEqual(
                result,
                {"rank": rank, "gpu": ["2", "5"][rank % 2], "cwd": str(train.REPO)},
            )

    def test_failed_rank_stops_other_workers_and_returns_failure(self):
        start = time.monotonic()
        self.assertEqual(self.run_workers("fail"), 1)
        self.assertLess(time.monotonic() - start, 10)
        self.assertTrue(all(worker.poll() is not None for worker in self.workers))
        self.assertEqual(self.workers[1].returncode, 7)

    def test_interrupt_stops_all_workers(self):
        real_sleep = time.sleep
        interrupted = False

        def interrupt_once(seconds):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            real_sleep(seconds)

        with mock.patch.object(train.time, "sleep", side_effect=interrupt_once):
            self.assertEqual(self.run_workers("fail"), 130)
        self.assertTrue(all(worker.poll() is not None for worker in self.workers))


if __name__ == "__main__":
    unittest.main()
