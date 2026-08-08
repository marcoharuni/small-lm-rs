# Limitations

The bundled model is intentionally small. Its 8M parameter count and limited instruction-tuning set constrain language quality, factuality, and instruction following.

The current runtime also has a few engineering limits:

- inference is CPU-only;
- weights and KV cache are FP32;
- quantized model loading is not implemented yet;
- requests share one loaded generation service and CPU inference is serialized;
- SSE responses are framed after synchronous generation rather than emitted token-by-token;
- authentication, TLS, quotas, and multi-tenant controls are not included;
- the training path is single-device JAX rather than a distributed training system.

These are implementation boundaries, not hidden claims. The repository focuses on a small, inspectable end-to-end path from training through independent Rust inference.
