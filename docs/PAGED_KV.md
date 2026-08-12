# Paged KV cache

Milestone 4 replaces the engine's dense per-layer KV reservation with lazy fixed-size pages while preserving the existing autoregressive cache semantics.

## Storage contract

- page size: 16 token positions
- storage dtype: FP32 keys and values
- layout inside each page: token-major `[page_tokens, kv_heads * head_dim]`
- one independent page list per transformer layer
- pages are allocated only when an append crosses into them
- cached attention reads key/value heads directly from the page that owns the requested logical token position
- no permanent dense mirror is kept
- the legacy contiguous prefix accessors materialize owned diagnostic snapshots only when explicitly requested
- truncation drops trailing pages that are no longer needed
- `clear()` releases all pages

The cache remains request-local. Paging changes physical allocation only; it does not change RoPE positions, attention ordering, model logits, or sampler state.

## Bundled model geometry

For the bundled 8-layer model:

```text
layers                 8
maximum sequence       512 tokens
KV heads               2
head dimension         64
KV width/layer/token   128 floats
K + V payload          1,024 bytes/layer/token
all-layer payload      8,192 bytes/token = 8 KiB/token
page quantum           16 tokens
all-layer page quantum 131,072 bytes = 128 KiB
full dense reservation 4,194,304 bytes = 4 MiB
```

## Deterministic allocation behavior

The table below describes KV payload reservation only; allocator metadata and `Vec` bookkeeping are intentionally excluded.

| Logical sequence length | Pages across 8 layers | Allocated token capacity across layers | KV payload reserved | Reduction vs dense 512-token reservation |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 B | 100.000% |
| 1 | 8 | 128 | 128 KiB | 96.875% |
| 16 | 8 | 128 | 128 KiB | 96.875% |
| 17 | 16 | 256 | 256 KiB | 93.750% |
| 32 | 16 | 256 | 256 KiB | 93.750% |
| 128 | 64 | 1,024 | 1 MiB | 75.000% |
| 512 | 256 | 4,096 | 4 MiB | 0.000% |

Paging therefore reduces unused-capacity reservation for short and medium sequences. It is not KV compression: once all 512 context positions are populated, the KV payload equals the dense representation.

## Reproduce

```bash
cargo run --release -p smalllm-engine --example paged_kv_benchmark -- \
  artifacts/small-lm-8m
```

The example reports model geometry, the equivalent dense KV payload, page counts, allocated token capacity, paged payload bytes, and the percentage of dense reservation avoided at each checkpoint.

## Correctness and lifecycle

`KvCache` keeps the same logical sequence-length and head-access contracts used by FP32, quantized, and batched decoding. Failed cached-attention/model operations can truncate back to the prior logical length; truncation releases pages beyond the retained prefix. Completed operations leave all layers synchronized.

Milestone 4 deliberately does not claim lower end-to-end process RSS from the theoretical payload table alone. The deterministic table proves reduced KV reservation; process-level RSS depends on model weights, allocator behavior, temporary activations, thread stacks, and workload and must be measured separately when relevant.
