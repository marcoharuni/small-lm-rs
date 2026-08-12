# Prefix caching

Milestone 5 adds reusable prompt-prefix state on top of the paged KV cache introduced in Milestone 4.

## Design

Each `GenerationScheduler` owns a bounded `PrefixCache`. The default capacity is 16 entries.

Each entry stores:

- the prompt token IDs
- a cloned paged `KvCache` snapshot after that prefix
- the final vocabulary-logit row for the last cached token

Lookup chooses the longest retained token prefix that matches the incoming prompt.

## Exact prompt hit

If the full prompt already exists in the cache, generation clones the cached paged KV state and final logit row. Prompt prefill is skipped entirely before sampling the first generated token.

## Partial prefix hit

If only a leading portion of the prompt matches, generation clones that prefix state and evaluates only the unmatched suffix through cached-token decode. The completed prompt is then inserted as a new reusable entry.

This is token-prefix reuse, not string memoization: cache keys are token ID sequences, and the reused object is model KV state.

## Eviction and statistics

The implementation is intentionally small and correctness-first:

- maximum 16 retained entries by default
- oldest entry is evicted when full
- cumulative hit and miss counters are exposed
- cache contents can be cleared explicitly

More sophisticated LRU/admission policies are left for future work.

## Correctness

Exact and partial reuse preserve generation semantics by cloning the same paged KV representation consumed by cached attention. The benchmark also checks that greedy next-token selection matches the corresponding cold-prefill path for both exact and partial hits.

## Benchmark

A repeatable benchmark is included:

```bash
cargo run --release -p smalllm-engine --example prefix_cache_benchmark -- \
  artifacts/small-lm-8m 10
```

It compares:

- cold 64-token prompt prefill
- exact 64-token prefix-cache hit
- cold 72-token prompt prefill
- a 72-token prompt reusing its first 64 cached tokens

The benchmark reports mean latency, derived speedup, and token-equivalence checks. Machine-specific timing results belong in the full benchmark milestone rather than being treated as portable Milestone-5 guarantees.

## Milestone boundary

Milestone 5 covers correctness-first shared prompt-prefix reuse in the generation scheduler. Full cross-workload performance characterization and benchmark tables are handled by the following benchmark-suite milestone.
