"""Traces a stage into forward / backward graphs and rewrites them (FX).

    trace.py   AOTAutograd tracing; the save policy is the partition.
    passes.py  FX passes: Linear sites, taps, saved-parameter aliases, and
               writing saved tensors straight into slot buffers.
"""
