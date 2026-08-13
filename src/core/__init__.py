"""Shared domain logic for the Sale Deed AI Processing System.

Independent of the inference runtime and of the UI. Everything here is pure
Python with no GPU, no model and no I/O beyond what is passed in, so it is fully
unit-testable on any machine.

    validation   INFERENCE_PIPELINE layers 2-7, flag codes, confidence
"""

__all__ = ["validation"]
