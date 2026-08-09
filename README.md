# small-lm-rs

**Train in JAX. Export to SafeTensors. Run entirely in Rust.**

`small-lm-rs` is an end-to-end small-language-model systems project: a decoder-only model is trained in JAX/Flax NNX, exported as framework-independent artifacts, and executed by an independent Rust CPU inference engine with KV-cached generation and an OpenAI-compatible server. Python and JAX are not required at inference time.

## At a glance

| | |
| ---