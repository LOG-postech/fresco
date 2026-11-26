"""Command line of main.py, built from the configs.py dataclasses.

parse_config() parses ``--<section>.<field>`` flags (with tyro) into a nested
configs.Config, which the trainer reads directly (``cfg.lr.schedule.peak_value``,
``cfg.method.name``, ...).
"""

import dataclasses
import os
import sys
import typing
from collections.abc import Sequence
from typing import Optional

import tyro

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import configs  # pylint: disable=wrong-import-position
from lib.numeric import parse_float  # pylint: disable=wrong-import-position


NUMERIC_REGISTRY = tyro.constructors.ConstructorRegistry()


@NUMERIC_REGISTRY.primitive_rule
def _float_constructor(info: tyro.constructors.PrimitiveTypeInfo):
    if info.type is float:
        return tyro.constructors.PrimitiveConstructorSpec(
            nargs=1,
            metavar='FLOAT|FRACTION',
            instance_from_str=lambda values: parse_float(values[0]),
            is_instance=lambda value: isinstance(value, float),
            str_from_instance=lambda value: [str(value)],
        )
    return None

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _description() -> str:
    """Returns the --help description: the model preset table."""
    rows = []
    for name, spec in configs.MODEL_PRESETS.items():
        shape = ', '.join(
            f'{k}={v}'
            for k, v in dataclasses.asdict(spec).items()
            if v is not None
        )
        rows.append(f'  {name}: {shape}')
    return (
        'FRESCO / async pipeline-parallel training.\n\n'
        'Model presets (--model):\n' + '\n'.join(rows)
    )


def parse_config(argv: Optional[Sequence[str]] = None) -> configs.Config:
    """Parses the command line into a Config.

    Exits with a usage error on an invalid command line.

    Args:
        argv: The arguments to parse (default: ``sys.argv[1:]``).

    Returns:
        The Config.
    """
    # tyro returns a Config or exits with a usage error.
    cfg = typing.cast(
        configs.Config,
        tyro.cli(
            configs.Config,
            args=argv,
            description=_description(),
            use_underscores=True,
            registry=NUMERIC_REGISTRY,
        ),
    )
    path = stage_layout_path(cfg)
    if not os.path.exists(os.path.join(REPO, path)):
        layouts = os.path.join(REPO, os.path.dirname(path))
        have = sorted(os.listdir(layouts)) if os.path.isdir(layouts) else []
        sys.exit(
            f'error: no stage layout {path} for --model {cfg.model} '
            f'--pipeline.n_stages {cfg.pipeline.n_stages} (have: {have})'
        )
    return cfg


def stage_layout(spec: configs.ModelConfig, n_stages: int) -> str:
    """Returns the stage-partition JSON path of a model split into n_stages.

    Args:
        spec: The model's ModelConfig.
        n_stages: The number of pipeline stages.

    Returns:
        The path, relative to the repository root.
    """
    family = spec.module.split('.')[-1]
    return f'lib/models/{family}/layers={spec.n_layer}/stage{n_stages}.json'


def model_spec(cfg: configs.Config) -> configs.ModelConfig:
    """Returns the ModelConfig of the config's model preset."""
    return configs.MODEL_PRESETS[cfg.model]


def stage_layout_path(cfg: configs.Config) -> str:
    """Returns the stage layout JSON of a config.

    Args:
        cfg: The Config.

    Returns:
        The path, relative to the repository root, of the model split into
        ``pipeline.n_stages`` stages (default: one layer per stage).
    """
    spec = model_spec(cfg)
    return stage_layout(spec, cfg.pipeline.n_stages or spec.n_layer)
