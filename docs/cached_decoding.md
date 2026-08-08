# KV-Cached Decoding

The Rust engine now separates autoregressive execution into two paths:

1. **Prompt prefill** computes the full causal prompt and stores rotated K/V
   rows for every transformer layer.
2. **Token decode** embeds one new token, applies RoPE at the absolute cached
   position, attends over the stored history, appends one K/V row per layer,
   and returns one vocabulary-logit row.

Cache mutation is transactional at model level: a failed multi-layer decode
restores every layer to its previous synchronized length.

## Correctness checks

- cached prompt logits versus uncached prompt logits
- cached one-token logits versus uncached final-position logits
- cached greedy generation versus uncached greedy generation
- cache capacity, synchronization, clear, and rollback behavior
- deterministic temperature, top-k, and top-p sampling

## Benchmark

```bash
cargo run --release -p nilemini-engine --example cached_benchmark -- \
  artifacts/nilemini-8m-situ
```

The command reports prompt-prefill latency, one-token decode latency, and the
final synchronized cache length as JSON.
