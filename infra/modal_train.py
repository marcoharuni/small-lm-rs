"""Modal L4 orchestration for NileMini smoke, pilot, full pretraining, SFT, and export."""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "nilemini-training"
VOLUME_NAME = "nilemini-training"
VOLUME_ROOT = Path("/vol/nilemini")
CODE_ROOT = Path("/root/nilemini-rs")
FROZEN_TOKENIZER = CODE_ROOT / "artifacts" / "nilemini-8m-situ" / "tokenizer.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "jax[cuda12]==0.11.0",
        "flax==0.12.8",
        "optax==0.2.8",
        "orbax-checkpoint==0.12.2",
        "datasets==5.0.1",
        "tokenizers==0.23.1",
        "safetensors==0.8.0",
        "huggingface-hub==1.26.1",
        "numpy==2.5.1",
        "scipy==1.18.0",
    )
    .add_local_dir("src", remote_path=str(CODE_ROOT / "src"), copy=True)
    .add_local_dir("configs/training", remote_path=str(CODE_ROOT / "configs/training"), copy=True)
    .add_local_file(
        "artifacts/nilemini-8m-situ/tokenizer.json",
        remote_path=str(FROZEN_TOKENIZER),
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": str(CODE_ROOT / "src"),
            "NILEMINI_WORKSPACE": str(VOLUME_ROOT),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
training_retries = modal.Retries(
    max_retries=10,
    initial_delay=1.0,
    backoff_coefficient=2.0,
)


def profile_path(name: str) -> Path:
    return CODE_ROOT / "configs" / "training" / name


@app.function(
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=2.0),
    single_use_containers=True,
)
def prepare(profile: str = "pilot_l4.json", force: bool = False) -> str:
    """Copy the frozen tokenizer and materialize the requested FineWeb-Edu token budget."""

    from nilemini.config import load_profile
    from nilemini.pretrain import prepare_profile_data

    volume.reload()
    selected = load_profile(profile_path(profile))
    train_path, validation_path = prepare_profile_data(
        selected,
        root=VOLUME_ROOT,
        force_data=force,
        frozen_tokenizer_source=FROZEN_TOKENIZER,
    )
    volume.commit()
    return f"prepared {selected.name}: {train_path} | {validation_path}"


@app.function(
    volumes={str(VOLUME_ROOT): volume},
    gpu="L4",
    timeout=24 * 60 * 60,
    retries=training_retries,
    single_use_containers=True,
)
def pretrain(profile: str = "pilot_l4.json") -> str:
    """Run/resume one L4 pretraining profile from its newest durable Orbax checkpoint."""

    from nilemini.config import load_profile
    from nilemini.pretrain import run_pretraining

    volume.reload()
    selected = load_profile(profile_path(profile))
    final_params = run_pretraining(
        selected,
        root=VOLUME_ROOT,
        checkpoint_hook=volume.commit,
        prepare_if_missing=False,
        frozen_tokenizer_source=FROZEN_TOKENIZER,
    )
    volume.commit()
    return str(final_params)


@app.function(
    volumes={str(VOLUME_ROOT): volume},
    cpu=8,
    memory=32768,
    timeout=12 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=2.0, backoff_coefficient=2.0),
    single_use_containers=True,
)
def prepare_sft(profile: str = "sft_l4.json", force: bool = False) -> str:
    """Materialize the pinned 70k SmolTalk subset without paying for a GPU."""

    from nilemini.config import artifact_dir, load_sft_profile
    from nilemini.sft import prepare_sft_data
    from nilemini.tokenizer import ensure_tokenizer

    volume.reload()
    ensure_tokenizer(30_000_000, root=VOLUME_ROOT, frozen_source=FROZEN_TOKENIZER)
    selected = load_sft_profile(profile_path(profile))
    prepare_sft_data(selected, root=VOLUME_ROOT, force=force)
    volume.commit()
    return f"prepared {selected.name} under {artifact_dir(VOLUME_ROOT).parent.parent}"


@app.function(
    volumes={str(VOLUME_ROOT): volume},
    gpu="L4",
    timeout=24 * 60 * 60,
    retries=training_retries,
    single_use_containers=True,
)
def sft(base_profile: str = "full_l4.json", profile: str = "sft_l4.json") -> str:
    """Run/resume SFT from the completed full-pretraining parameter checkpoint."""

    from nilemini.config import checkpoint_dir, load_profile, load_sft_profile
    from nilemini.sft import run_sft

    volume.reload()
    base = load_profile(profile_path(base_profile))
    selected = load_sft_profile(profile_path(profile))
    base_params = checkpoint_dir(VOLUME_ROOT) / "pretrain" / base.name / "final-params"
    final_params = run_sft(
        selected,
        root=VOLUME_ROOT,
        base_params_path=base_params,
        checkpoint_hook=volume.commit,
        prepare_if_missing=False,
    )
    volume.commit()
    return str(final_params)


@app.function(
    volumes={str(VOLUME_ROOT): volume},
    gpu="L4",
    timeout=2 * 60 * 60,
    single_use_containers=True,
)
def export(profile: str = "sft_l4.json") -> str:
    """Export final SFT parameters to the SafeTensors/Rust artifact contract."""

    from nilemini.config import checkpoint_dir, load_sft_profile
    from nilemini.export import export_model

    volume.reload()
    selected = load_sft_profile(profile_path(profile))
    params = checkpoint_dir(VOLUME_ROOT) / "sft" / selected.name / "final-params"
    output = export_model(params, root=VOLUME_ROOT)
    volume.commit()
    return str(output)


@app.local_entrypoint()
def main(stage: str = "smoke", force: bool = False) -> None:
    """Run one complete stage; data preparation happens on CPU before L4 allocation."""

    if stage == "prepare-smoke":
        print(prepare.remote("smoke.json", force))
        return
    if stage == "smoke":
        print(prepare.remote("smoke.json", force))
        print(pretrain.remote("smoke.json"))
        return
    if stage == "prepare-pilot":
        print(prepare.remote("pilot_l4.json", force))
        return
    if stage == "pilot":
        print(prepare.remote("pilot_l4.json", force))
        call = pretrain.spawn("pilot_l4.json")
        print(f"Pilot training spawned in background: {call.object_id}")
        return
    if stage == "prepare-full":
        print(prepare.remote("full_l4.json", force))
        return
    if stage == "full":
        print(prepare.remote("full_l4.json", force))
        print(pretrain.remote("full_l4.json"))
        return
    if stage == "prepare-onehour-probe":
        print(prepare.remote("onehour_probe.json", force))
        return
    if stage == "onehour-probe":
        call = pretrain.spawn("onehour_probe.json")
        print(f"One-hour 8M probe spawned in background: {call.object_id}")
        return
    if stage == "prepare-onehour-final":
        print(prepare.remote("onehour_final.json", force))
        return
    if stage == "onehour-final":
        call = pretrain.spawn("onehour_final.json")
        print(f"Final 8M pretraining spawned in background: {call.object_id}")
        return
    if stage == "prepare-onehour-sft":
        print(prepare_sft.remote("onehour_sft.json", force))
        return
    if stage == "onehour-sft":
        call = sft.spawn("onehour_final.json", "onehour_sft.json")
        print(f"Final 8M SFT spawned in background: {call.object_id}")
        return
    if stage == "prepare-sft":
        print(prepare_sft.remote("sft_l4.json", force))
        return
    if stage == "sft":
        print(prepare_sft.remote("sft_l4.json", force))
        print(sft.remote("full_l4.json", "sft_l4.json"))
        return
    if stage == "onehour-export":
        print(export.remote("onehour_sft.json"))
        return
    if stage == "export":
        print(export.remote("sft_l4.json"))
        return
    raise ValueError(
        "stage must be one of: prepare-smoke, smoke, prepare-pilot, pilot, "
        "prepare-full, full, prepare-onehour-probe, onehour-probe, "
        "prepare-onehour-final, onehour-final, prepare-onehour-sft, onehour-sft, "
        "prepare-sft, sft, onehour-export, export"
    )
