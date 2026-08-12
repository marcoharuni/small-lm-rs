# Limitations

The bundled model is intentionally small. Its 8M parameter count, 512-token context, and limited instruction-tuning set constrain language quality, factuality, and instruction following.

The post-submission Rust runtime now includes continuous batching, weight-only INT8 loading, AVX2-assisted INT8 execution, request-local paged KV storage, shared prefix caching, and a consolidated benchmark suite. Those additions improve the systems path, but important engineering limits remain:

- inference is CPU-only;
- FP32 remains the baseline model path, while the quantized path is symmetric per-output-channel weight-only INT8 rather than W8A8 integer-only compute;
- activations, normalization state, embeddings/tied LM-head source weights, and KV values remain FP32;
- there is no FlashAttention-style tiled attention kernel;
- paged KV is request-local lazy paging, not a global vLLM-style physical block allocator/free-list or shared-prefix page manager;
- the shared prefix cache is bounded to 16 entries and uses a simple correctness-first eviction policy;
- prompt prefill is still request-local rather than chunked/batched prefill;
- SSE responses are framed after generation completes rather than emitted token-by-token;
- speculative decoding, INT4/FP8 execution, GPU kernels, and distributed inference are not implemented;
- authentication, TLS, quotas, and multi-tenant controls are not included;
- the training path is single-device JAX rather than a distributed training system.

These are explicit implementation boundaries, not hidden claims. The repository focuses on a small, inspectable end-to-end path from training through independent Rust inference and measured systems optimizations.

See [`FINAL_SYSTEMS_REPORT.md`](FINAL_SYSTEMS_REPORT.md) for the completed milestone summary and [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for machine-specific measurements.
