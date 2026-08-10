//! CPU inference building blocks for SmallLM.
//!
//! The crate provides validated configuration, tokenization, SafeTensors
//! loading, foundational tensor operations, embedding lookup, linear
//! projection, RMSNorm, SiTU-GLU, stable softmax, interleaved RoPE, complete
//! causal grouped-query attention, transformer blocks, and complete uncached
//! full-model prefill logits, JAX reference parity, uncached greedy decoding,
//! dense per-layer KV storage, cache-populating prompt prefill, full-model
//! cached decoding, deterministic greedy generation, temperature/top-k/top-p
//! sampling, and token-stepped generation sessions for scheduler-driven
//! decoding.

#![forbid(unsafe_code)]

pub mod attention;
pub mod chat;
pub mod config;
pub mod embedding;
pub mod error;
pub mod generation;
pub mod generation_config;
pub mod greedy;
pub mod kv_cache;
pub mod linear;
pub mod model;
pub mod parity;
pub mod rmsnorm;
pub mod rope;
pub mod sampler;
pub mod service;
pub mod session;
pub mod situ_glu;
pub mod softmax;
mod tensor;
pub mod tokenizer;
pub mod transformer;
pub mod weights;

pub use config::ModelConfig;
pub use error::{EngineError, Result};
pub use generation_config::GenerationConfig;
pub use model::SmallLMModel;
pub use service::{ArtifactPaths, GenerationService, ServiceGenerationOutput};
pub use session::{GenerationFinishReason, GenerationSession};
