"""Everything main.py uses (configs.py declares the settings).

    models/      The models, their stage layouts, and build.py.
    correctors/  The weight-consistency correctors (FRESCO and the baselines)
                 composed around a base optimizer.
    linalg/      Randomized subspace iteration, QR / Newton-Schulz
                 orthogonalisation: the numerics behind FRESCO.
    optim/       The base optimizer and the learning-rate schedule.
    pipeline/    Send / recv between stages, and each stage's compute (eager
                 runtime or graph engine).
    graph/       The graph stage: traced forward / backward graphs, CUDA graphs,
                 correctors described as primitives.
    config.py, data.py, checkpoint.py, utils.py: command line, data,
                 checkpoints, seeding.
"""
