"""Offline weight-only INT8 quantization for SmallLM artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file, save_file

INT8_MAX = 127.0
SCALE_SUFFIX = ".scale"
PROJECTION_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def is_quantized_projection(name: str) -> bool:
    """Return whether one exported tensor belongs to an INT8 projection."""

    return name.endswith(PROJECTION_SUFFIXES)


def quantize_per_output_channel(
    weight: NDArray[np.float32],
) -> tuple[NDArray[np.int8], NDArray[np.float32]]:
    """Symmetrically quantize one `[out_features, in_features]` matrix."""

    if weight.ndim != 2:
        raise ValueError(f"projection weight must be rank-2, received shape {weight.shape}")
    if weight.dtype != np.float32:
        raise ValueError(f"projection weight must be float32, received {weight.dtype}")
    if not np.isfinite(weight).all():
        raise ValueError("projection weight contains non-finite values")

    row_max = np.max(np.abs(weight), axis=1)
    scales = np.where(row_max > 0.0, row_max / INT8_MAX, 1.0).astype(np.float32)
    quantized = np.clip(
        np.rint(weight / scales[:, None]),
        -INT8_MAX,
        INT8_MAX,
    ).astype(np.int8)
    return np.ascontiguousarray(quantized), np.ascontiguousarray(scales)


def quantize_tensors(
    tensors: Mapping[str, NDArray[np.generic]],
) -> dict[str, NDArray[np.generic]]:
    """Create the mixed FP32/INT8 tensor map used by the Rust runtime."""

    quantized: dict[str, NDArray[np.generic]] = {}
    for name in sorted(tensors):
        tensor = np.asarray(tensors[name])
        if tensor.dtype != np.float32:
            raise ValueError(f"{name} must be float32 before quantization, found {tensor.dtype}")

        if is_quantized_projection(name):
            values, scales = quantize_per_output_channel(tensor)
            quantized[name] = values
            quantized[f"{name}{SCALE_SUFFIX}"] = scales
        else:
            quantized[name] = np.ascontiguousarray(tensor, dtype=np.float32)

    return quantized


def quantize_file(input_path: Path, output_path: Path) -> None:
    """Quantize one FP32 SafeTensors artifact and write the INT8 artifact."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")

    tensors = load_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        quantize_tensors(tensors),
        output_path,
        metadata={
            "format": "smalllm-int8-v1",
            "quantization": "symmetric-per-output-channel-weight-only-int8",
            "scale_dtype": "float32",
            "zero_point": "0",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for offline INT8 quantization."""

    parser = argparse.ArgumentParser(description="Quantize SmallLM projection weights to INT8.")
    parser.add_argument("input", type=Path, help="FP32 model.safetensors path")
    parser.add_argument("output", type=Path, help="destination INT8 SafeTensors path")
    return parser


def main() -> None:
    """Command-line entry point."""

    args = build_parser().parse_args()
    quantize_file(args.input, args.output)


if __name__ == "__main__":
    main()
