# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Trains one pipeline stage (one process per stage) with FRESCO or a baseline.

Settings: configs.Config (``--<section>.<field>``; see ``--help``). main() wires data (lib.data), the model (lib.models.build), the
pipeline (lib.pipeline; eager or graph engine), the staleness method
(lib.correctors) and the LR schedule (lib.optim). train() is one continuous
pipeline run (one fill, one drain); validate() scores the held-out loss once.
"""

import dataclasses
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Sequence
from typing import Optional

import psutil
import torch
import wandb

import configs
from lib import correctors
from lib.pipeline import graph_engine, runtime, setup
from lib.optim import base_optimizer, lr_schedule
from lib.models import build
from lib import checkpoint, config, data, utils

LOG_FORMAT = "%(levelname)s:%(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def is_first_stage(r: runtime.StageRuntime) -> bool:
    return r.stage is None or r.stage == 0


def is_last_stage(r: runtime.StageRuntime) -> bool:
    return r.stage is None or r.stage == r.num_stages - 1


def run_name(cfg: configs.Config, num_stages: int) -> str:
    """The wandb run name: logging.exp_name, or stages and method."""
    if cfg.logging.exp_name:
        return cfg.logging.exp_name
    return f"stage{num_stages}_{cfg.method.name}"


def build_optimizer(
    cfg: configs.Config,
    r: runtime.StageRuntime,
    num_versions: int,
) -> correctors.AsyncPipelineOptimizer:
    """Builds the stage's staleness method around its base optimizer.

    Args:
        cfg: The experiment configs.Config.
        r: The stage's runtime.StageRuntime.
        num_versions: The stage's weight versions.

    Returns:
        The staleness method (a lib.correctors.AsyncPipelineOptimizer).
    """
    base = base_optimizer.build_base_optimizer(
        cfg.optimizer, cfg.lr.schedule.peak_value, r.master_parameters
    )
    return correctors.build(
        base,
        r.modules(),
        num_versions,
        **configs.method_slots(cfg.method),
        staleness_lr_steps=cfg.damper.staleness_lr_steps,
        stage=r.stage or 0,
        clip_grad=cfg.optimizer.clip_grad,
        verbose_freq=cfg.logging.verbose_frequency,
    )


def build_graph_engine(
    cfg: configs.Config,
    r: runtime.StageRuntime,
    optimizer: correctors.AsyncPipelineOptimizer,
) -> graph_engine.GraphEngine:
    """The graph engine for this rank (--runtime.engine graph)."""
    return graph_engine.GraphEngine(r, optimizer)


def main(cfg: configs.Config) -> None:
    torch.cuda.set_device(cfg.pipeline.local_rank)
    utils.seed_everything(cfg.train.seed, cfg.train.deterministic)

    # Data and model.
    spec = config.model_spec(cfg)
    train_dataset, val_dataset, vocab_size = data.load_datasets(
        cfg.data.name, os.path.expanduser(cfg.data.dir), spec.block_size
    )
    model, arch = build.build_model(spec, vocab_size)
    nparams = build.count_parameters(model)
    logging.info(f"#Params: {nparams:.2f} M")

    # This process's pipeline stage.
    training_shapes, eval_shapes, dtypes = setup.tensor_shapes(
        model,
        cfg.train.batch_size,
        cfg.train.eval_batch_size,
        spec.block_size,
        spec.n_embd,
        vocab_size,
    )
    stage_maps = setup.load_stage_maps(
        config.stage_layout_path(cfg), spec.n_layer
    )
    r = runtime.StageRuntime(
        model=model,
        distributed_backend=cfg.pipeline.backend,
        training_tensor_shapes=training_shapes,
        eval_tensor_shapes=eval_shapes,
        training_tensor_dtypes=dtypes,
        inputs_module_destinations={"input": 0},
        target_tensor_names={"target"},
        configuration_maps=stage_maps,
        master_addr=cfg.pipeline.master_addr,
        master_port=cfg.pipeline.master_port,
        rank=cfg.pipeline.rank,
        local_rank=cfg.pipeline.local_rank,
        num_ranks_in_server=cfg.pipeline.num_ranks_in_server,
        verbose_freq=cfg.logging.verbose_frequency,
        model_type=runtime.LANGUAGE_MODELING,
    )

    if cfg.logging.wandb and is_last_stage(r):
        wandb.init(
            project=cfg.logging.wandb_project,
            config={**dataclasses.asdict(cfg), "nparams": nparams},
            name=run_name(cfg, r.num_stages),
        )

    num_versions = correctors.num_weight_versions(r.num_warmup_minibatches)
    logging.info(f"## Stage: {r.stage}, Num versions: {num_versions}")

    start_step, resumed = 0, None
    if cfg.checkpoint.resume:
        resumed = checkpoint.load_checkpoint(cfg.checkpoint.resume, r.stage)
        start_step = resumed["step"]
        r.load_state_dict(resumed["state_dict"])

    optimizer = build_optimizer(cfg, r, num_versions)
    if resumed is not None:
        optimizer.load_state_dict(resumed["optimizer"])
    engine = None
    if cfg.runtime.engine == "graph":
        engine = build_graph_engine(cfg, r, optimizer)

    # Data loaders; a replicated first stage reads one shard per replica. The
    # train loader is streamed (DataUtil restarts it when exhausted), so the
    # run has no epochs.
    first_stage_ranks = (
        len(stage_maps["stage_to_rank_map"][0])
        if stage_maps["stage_to_rank_map"] is not None
        else 1
    )
    train_loader, val_loader, _ = data.make_loaders(
        train_dataset,
        val_dataset,
        batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.train.eval_batch_size,
        workers=cfg.data.workers,
        rank=cfg.pipeline.rank,
        first_stage_ranks=first_stage_ranks,
        worker_init_fn=utils.worker_init_fn(cfg.train.seed),
    )
    du = data.DataUtil(train_loader, val_loader)

    # One continuous pipeline run: it fills once and drains once, at the end.
    total_steps = r.num_iterations(loader_size=cfg.train.steps)
    train(cfg, r, optimizer, du, start_step, total_steps, engine)

    # The held-out loss, once, after the final step: the same eval batches at
    # the same step for every method.
    eval_loss = validate(cfg, val_loader, r, du, total_steps)

    should_save = cfg.checkpoint.dir_not_nfs or r.rank_in_stage == 0
    if cfg.checkpoint.dir and should_save:
        checkpoint.save_checkpoint(
            {
                "step": total_steps,
                "arch": arch,
                "state_dict": r.state_dict(),
                "eval_loss": eval_loss if is_last_stage(r) else None,
                "optimizer": optimizer.state_dict(),
            },
            cfg.checkpoint.dir,
            r.stage,
        )


def _log_memory() -> tuple[float, float]:
    return (
        float(torch.cuda.memory_allocated()) / 10**9,
        float(torch.cuda.memory_reserved()) / 10**9,
    )


def train(
    cfg: configs.Config,
    r: runtime.StageRuntime,
    optimizer: correctors.AsyncPipelineOptimizer,
    du: data.DataUtil,
    start_step: int,
    total_steps: int,
    engine: Optional[graph_engine.GraphEngine] = None,
) -> None:
    """Trains one stage for optimizer steps [start_step, total_steps).

    The pipeline fills once (the warm-up forwards), runs steady-state 1F1B, and
    drains once at the end; there is no flush in between. Every optimizer step,
    including the drain's, gets the learning rate of its own step index.

    The last stage logs ``Step: [s/T] ... Loss: <loss> (<ema>)``: the
    micro-batch's training loss and its bias-corrected exponential moving
    average (decay cfg.logging.loss_ema).
    """
    n = total_steps - start_step
    if n <= 0:
        return
    r.train(n)
    r.set_batch_sampler(
        du.get_batch if is_first_stage(r) else None, is_eval=False
    )
    num_warmup_minibatches = min(r.num_warmup_minibatches, n)
    if cfg.logging.verbose_frequency > 0:
        logging.info(
            "Letting in %d warm-up minibatches", num_warmup_minibatches
        )
        logging.info("Running training for %d minibatches", n)

    # The stage's peak LR, adjusted for its replicas.
    base_lr = r.get_adjusted_learning_rate(base_lr=cfg.lr.schedule.peak_value)

    def forward() -> None:
        if engine is not None:  # the method's weights are in the graphs
            engine.run_forward()
            return
        optimizer.before_forward()  # e.g. install the predicted weights
        r.run_forward()
        optimizer.after_forward()

    def backward_and_step(step: int) -> None:
        lr_schedule.adjust_learning_rate(
            optimizer, cfg.lr.schedule, base_lr, step, total_steps
        )
        optimizer.zero_grad()
        if engine is not None:
            engine.run_backward()
            engine.step()
            return
        optimizer.load_old_params()  # load weights for backprop
        r.run_backward()
        optimizer.load_new_params()  # load weights for optimizer step
        optimizer.step()

    batch_time = utils.AverageMeter()
    decay, ema, ema_weight = cfg.logging.loss_ema, 0.0, 0.0
    start = end = time.time()

    for _ in range(num_warmup_minibatches):  # Fill the pipeline.
        forward()

    for i in range(n - num_warmup_minibatches):  # Steady state.
        step = start_step + i
        forward()

        if is_last_stage(r):
            loss = r.loss.item()
            ema, ema_weight = decay * ema + (1 - decay) * loss, (
                decay * ema_weight + (1 - decay)
            )
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg.logging.print_freq == 0:
                memory, cached = _log_memory()
                logging.info(
                    "Step: [%d/%d]\tTime: %.3f (%.3f)\tElapsed [hr]: %.3f\t"
                    "Memory: %.3f (%.3f) (%.3f)\tLoss: %.4f (%.4f)",
                    step,
                    total_steps,
                    batch_time.val,
                    batch_time.avg,
                    (end - start) / 3600.0,
                    memory,
                    cached,
                    psutil.virtual_memory().percent,
                    loss,
                    ema / ema_weight,
                )
                sys.stdout.flush()
                if cfg.logging.wandb:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/loss_ema": ema / ema_weight,
                        },
                        step=step,
                    )
        elif i % cfg.logging.print_freq == 0:
            memory, cached = _log_memory()
            logging.info(
                "Step: [%d/%d]\tMemory: %.3f (%.3f)",
                step,
                total_steps,
                memory,
                cached,
            )
            sys.stdout.flush()

        backward_and_step(step)

    for j in range(num_warmup_minibatches):  # Drain the pipeline, once.
        backward_and_step(start_step + n - num_warmup_minibatches + j)

    r.wait()
    logging.info(
        "Trained steps %d-%d in %.3f seconds",
        start_step,
        total_steps,
        time.time() - start,
    )


def validate(
    cfg: configs.Config,
    val_loader: torch.utils.data.DataLoader,
    r: runtime.StageRuntime,
    du: data.DataUtil,
    step: int,
) -> float:
    """Measures the held-out loss on the first cfg.train.eval_steps batches.

    Runs once, after training (the pipeline is drained), so every method is
    scored on the same eval batches at the same step. The last stage logs
    ``* Eval loss <loss>  ppl <exp(loss)>  (step <step>, <n> batches)``.

    Returns:
        The mean eval loss on the last stage (0 elsewhere).
    """
    losses = utils.AverageMeter()
    n = r.num_iterations(
        loader_size=len(val_loader.dataset) // cfg.train.eval_batch_size
    )
    n = min(n, cfg.train.eval_steps)
    r.eval(n)
    r.set_batch_sampler(
        du.get_batch if is_first_stage(r) else None, is_eval=True
    )
    num_warmup_minibatches = min(r.num_warmup_minibatches, n)
    if cfg.logging.verbose_frequency > 0:
        logging.info(
            "Letting in %d warm-up minibatches", num_warmup_minibatches
        )
        logging.info("Running validation for %d minibatches", n)
    start = time.time()

    with torch.no_grad():
        for _ in range(num_warmup_minibatches):
            r.run_forward()
        for i in range(n - num_warmup_minibatches):
            r.run_forward()
            r.run_ack()
            if is_last_stage(r):
                losses.update(r.loss.item(), r.output.size(0))
                if i % cfg.logging.print_freq == 0:
                    logging.info(
                        "Eval batch %d/%d: loss %.4f (mean %.4f)",
                        i,
                        n,
                        losses.val,
                        losses.avg,
                    )
                    sys.stdout.flush()
        for _ in range(num_warmup_minibatches):
            r.run_ack()
        r.wait()

    if is_last_stage(r):
        logging.info(
            " * Eval loss %.4f  ppl %.2f  (step %d, %d batches)",
            losses.avg,
            math.exp(min(losses.avg, 50.0)),
            step,
            n,
        )
        if cfg.logging.wandb:
            wandb.log(
                {
                    "eval/loss": losses.avg,
                    "eval/ppl": math.exp(min(losses.avg, 50.0)),
                },
                step=step,
            )
    logging.info("Evaluated %d batches in %.3f seconds", n, time.time() - start)
    return losses.avg


def kill_other_ranks(master_port: str) -> None:
    """Terminates this run's other ranks on this node.

    A rank of the run is a process of this script with the same master port;
    other runs (other ports) and other programs are left alone.
    """
    script = os.path.abspath(__file__)
    for proc in psutil.process_iter(["pid", "cmdline", "cwd"]):
        cmd, cwd = proc.info["cmdline"] or [], proc.info["cwd"] or ""
        if proc.info["pid"] == os.getpid() or not any(
            os.path.abspath(os.path.join(cwd, arg)) == script for arg in cmd
        ):
            continue
        port = _flag_value(cmd, "--pipeline.master_port")
        if (port or configs.PipelineConfig().master_port) == str(master_port):
            try:
                proc.terminate()
            except psutil.Error:
                pass


def _flag_value(cmd: Sequence[str], flag: str) -> Optional[str]:
    """The value of ``flag`` in a command line (``--f=v`` or ``--f v``)."""
    for i, arg in enumerate(cmd):
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


if __name__ == "__main__":
    # An invalid command line exits here with a usage error, before any rank
    # starts.
    experiment = config.parse_config()
    try:
        main(experiment)
    except Exception:  # pylint: disable=broad-except
        logging.info(traceback.format_exc())
        logging.info("Killing this run's other ranks...")
        sys.stdout.flush()
        kill_other_ranks(experiment.pipeline.master_port)
        sys.exit(1)
