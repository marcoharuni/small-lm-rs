# mypy: ignore-errors
"""FP32 SafeTensors export compatible with the verified Rust engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import save_file as save_safetensors

from smalllm.checkpoint import restore_parameters
from smalllm.config import MODEL, artifact_dir, checkpoint_dir, load_sft_profile, workspace_root
from smalllm.generation import chat_prompt
from smalllm.model import initialize_model
from smalllm.tokenizer import load_frozen_tokenizer, tokenizer_sha256


def export_tensors(parameter_tree) -> dict[str, np.ndarray]:
    """Map NNX parameter names/layouts to the Rust export contract."""

    pure = nnx.to_pure_dict(parameter_tree)
    tensors: dict[str, np.ndarray] = {}

    def fp32(value) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)

    def linear(name: str, node: Mapping[str, Any]) -> None:
        tensors[name] = fp32(node["kernel"]).T.copy()

    tensors["token_embedding.weight"] = fp32(pure["token_embedding"])
    for layer in range(MODEL.num_layers):
        block = pure[f"block_{layer}"]
        prefix = f"layers.{layer}"
        tensors[f"{prefix}.attention_norm.weight"] = fp32(block["attention_norm"]["weight"])
        attention = block["attention"]
        linear(f"{prefix}.q_proj.weight", attention["q_proj"])
        linear(f"{prefix}.k_proj.weight", attention["k_proj"])
        linear(f"{prefix}.v_proj.weight", attention["v_proj"])
        linear(f"{prefix}.o_proj.weight", attention["o_proj"])
        tensors[f"{prefix}.ffn_norm.weight"] = fp32(block["ffn_norm"]["weight"])
        ffn = block["ffn"]
        linear(f"{prefix}.gate_proj.weight", ffn["gate_proj"])
        linear(f"{prefix}.up_proj.weight", ffn["up_proj"])
        linear(f"{prefix}.down_proj.weight", ffn["down_proj"])
    tensors["final_norm.weight"] = fp32(pure["final_norm"]["weight"])
    count = sum(tensor.size for tensor in tensors.values())
    if count != MODEL.expected_parameter_count:
        raise RuntimeError(f"Exported parameter count {count} != {MODEL.expected_parameter_count}")
    return tensors


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_model(params_path: Path, *, root: Path, output_dir: Path | None = None) -> Path:
    """Restore final params and produce the complete Rust-consumable artifact package."""

    graphdef, params = initialize_model(42)
    params = restore_parameters(params_path, params)
    tokenizer = load_frozen_tokenizer(root)
    source_tokenizer = artifact_dir(root) / "tokenizer.json"
    output = output_dir or (root / "exports" / MODEL.model_name)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    weights = output / "model.safetensors"
    save_safetensors(
        export_tensors(params),
        str(weights),
        metadata={"format": "smallLM-v01", "dtype": "float32", "tied_embeddings": "true"},
    )
    shutil.copy2(source_tokenizer, output / "tokenizer.json")
    config = asdict(MODEL) | {
        "normalization": "RMSNorm",
        "position_encoding": "RoPE",
        "activation": "SiTU-GLU",
        "weight_layout": "out_features,in_features",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    generation_config = {
        "context_length": MODEL.context_length,
        "pad_token_id": tokenizer.token_to_id("<|pad|>"),
        "eos_token_id": tokenizer.token_to_id("<|eos|>"),
    }
    (output / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n",
        encoding="utf-8",
    )

    reference_ids = np.asarray(
        chat_prompt(tokenizer, [{"role": "user", "content": "Hello"}]),
        dtype=np.int32,
    )[None, :]
    model = nnx.merge(graphdef, params)
    reference_logits = np.asarray(model(jnp.asarray(reference_ids)), dtype=np.float32)
    (output / "reference_inputs.json").write_text(
        json.dumps({"token_ids": reference_ids.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )
    save_safetensors(
        {"logits": reference_logits},
        str(output / "reference_outputs.safetensors"),
    )

    artifact_files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    checksums = {path.name: file_sha256(path) for path in artifact_files}
    manifest = {
        "model_name": MODEL.model_name,
        "parameter_count": MODEL.expected_parameter_count,
        "tokenizer_sha256": tokenizer_sha256(output / "tokenizer.json"),
        "files": checksums,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(
        "\n".join(f"{checksum}  {name}" for name, checksum in sorted(checksums.items())) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export SmallLM parameters for Rust inference")
    parser.add_argument("--profile", type=Path, default=Path("configs/training/sft_l4.json"))
    parser.add_argument("--params", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = workspace_root()
    if args.params is None:
        profile = load_sft_profile(args.profile)
        params = checkpoint_dir(root) / "sft" / profile.name / "final-params"
    else:
        params = args.params
    print(export_model(params, root=root, output_dir=args.output))


if __name__ == "__main__":
    main()
