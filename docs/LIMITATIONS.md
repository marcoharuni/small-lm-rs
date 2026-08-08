# Limitations

NileMini now has a working training/export/inference implementation, but it is **not yet a completed trained-model release**.

## Remaining release work

- The planned 1.6B-token FineWeb-Edu pretraining run has not been completed.
- The planned 70k-example SmolTalk SFT run has not been completed.
- Final model-quality, safety, memorization, bias, and robustness evaluation is therefore unavailable.
- The final trained SafeTensors artifact has not yet been published.
- Training is currently a single-GPU JAX path; it does not implement multi-GPU sharding, FSDP, pipeline parallelism, or activation rematerialization.
- The Rust server's SSE endpoint is wire-compatible but currently frames output after synchronous generation rather than delivering each token as it is decoded.
- The local Rust engine is correctness-oriented CPU inference; further SIMD/quantization/threading optimization remains possible.
- Authentication, TLS, quotas, and multi-tenant server hardening are outside the internship demo scope.

## What is already real

The repository does contain and test the frozen tokenizer, exact JAX architecture, optimizer partition, data preparation, Orbax checkpoint/resume, SafeTensors export, Rust model execution, JAX↔Rust smoke-artifact parity, KV-cached generation, sampling, and OpenAI-shaped completion endpoints.

The checked-in artifact is for smoke/parity engineering only. It must not be presented as the final trained NileMini model.
