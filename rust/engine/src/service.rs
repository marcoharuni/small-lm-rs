//! Reusable model/tokenizer loading and request-isolated generation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::chat::{format_chat_prompt, ChatMessage};
use crate::config::ModelConfig;
use crate::error::Result;
use crate::generation::{generate_token_ids, GenerationRequest};
use crate::generation_config::GenerationConfig;
use crate::model::NileMiniModel;
use crate::sampler::SamplingConfig;
use crate::tokenizer::{NileTokenizer, SpecialTokenIds};

/// Paths expected inside one exported NileMini artifact directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactPaths {
    root: PathBuf,
}

impl ArtifactPaths {
    /// Resolve all artifact paths relative to a directory.
    #[must_use]
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Return the artifact directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the model configuration path.
    #[must_use]
    pub fn model_config(&self) -> PathBuf {
        self.root.join("config.json")
    }

    /// Return the generation configuration path.
    #[must_use]
    pub fn generation_config(&self) -> PathBuf {
        self.root.join("generation_config.json")
    }

    /// Return the tokenizer path.
    #[must_use]
    pub fn tokenizer(&self) -> PathBuf {
        self.root.join("tokenizer.json")
    }

    /// Return the model weight path.
    #[must_use]
    pub fn model_weights(&self) -> PathBuf {
        self.root.join("model.safetensors")
    }
}

/// Completed synchronous generation result prepared for an API layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ServiceGenerationOutput {
    /// Number of tokens consumed by prompt prefill.
    pub prompt_token_count: usize,
    /// Decoded continuation only, excluding the prompt.
    pub completion: String,
    /// Newly generated token identifiers.
    pub generated_token_ids: Vec<u32>,
    /// Stop reason reported by the generation engine.
    pub finish_reason: String,
}

/// Model and tokenizer loaded once for repeated independent requests.
#[derive(Debug)]
pub struct GenerationService {
    model: NileMiniModel,
    tokenizer: NileTokenizer,
    generation_config: GenerationConfig,
    special_tokens: SpecialTokenIds,
}

impl GenerationService {
    /// Load and cross-validate every runtime artifact in one directory.
    ///
    /// # Errors
    ///
    /// Returns configuration, tokenizer, weight, or cross-artifact validation
    /// errors.
    pub fn from_artifact_dir(root: impl AsRef<Path>) -> Result<Self> {
        let paths = ArtifactPaths::new(root);
        let model_config = ModelConfig::from_json_path(paths.model_config())?;
        let generation_config =
            GenerationConfig::from_json_path(paths.generation_config(), &model_config)?;
        let tokenizer = NileTokenizer::from_file(paths.tokenizer())?;
        let special_tokens = tokenizer.validate_contract(&model_config, &generation_config)?;

        let mut model = NileMiniModel::from_config(model_config)?;
        model.load_weights(paths.model_weights())?;

        Ok(Self {
            model,
            tokenizer,
            generation_config,
            special_tokens,
        })
    }

    /// Return the loaded model configuration.
    #[must_use]
    pub const fn model_config(&self) -> &ModelConfig {
        self.model.config()
    }

    /// Return the loaded generation configuration.
    #[must_use]
    pub const fn generation_config(&self) -> &GenerationConfig {
        &self.generation_config
    }

    /// Return validated reserved token identifiers.
    #[must_use]
    pub const fn special_tokens(&self) -> SpecialTokenIds {
        self.special_tokens
    }

    /// Execute a plain-text generation request.
    ///
    /// Each invocation allocates a new KV cache inside the generation engine,
    /// so no request can observe or mutate another request's decoding state.
    ///
    /// # Errors
    ///
    /// Returns request, tokenizer, context, model, cache, or sampling errors.
    pub fn generate_text(&self, request: &GenerationRequest) -> Result<ServiceGenerationOutput> {
        request.validate()?;
        self.execute_prompt(
            &request.prompt,
            request.max_new_tokens,
            request.sampling.clone(),
            true,
        )
    }

    /// Format and execute one chat generation request.
    ///
    /// Each invocation owns an independent KV cache.
    ///
    /// # Errors
    ///
    /// Returns chat-template, request, tokenizer, context, model, cache, or
    /// sampling errors.
    pub fn generate_chat(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        sampling: SamplingConfig,
    ) -> Result<ServiceGenerationOutput> {
        let prompt = format_chat_prompt(messages)?;
        self.execute_prompt(&prompt, max_new_tokens, sampling, false)
    }

    fn execute_prompt(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: SamplingConfig,
        add_special_tokens: bool,
    ) -> Result<ServiceGenerationOutput> {
        let prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens)?;
        let generated = generate_token_ids(
            &self.model,
            &prompt_token_ids,
            max_new_tokens,
            Some(self.generation_config.eos_token_id),
            sampling,
        )?;
        let completion = self
            .tokenizer
            .decode(&generated.generated_token_ids, true)?;

        Ok(ServiceGenerationOutput {
            prompt_token_count: prompt_token_ids.len(),
            completion,
            generated_token_ids: generated.generated_token_ids,
            finish_reason: generated.finish_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ArtifactPaths, GenerationService};

    #[test]
    fn artifact_paths_are_resolved_predictably() {
        let paths = ArtifactPaths::new("/models/nilemini");
        assert_eq!(paths.model_config(), paths.root().join("config.json"));
        assert_eq!(
            paths.generation_config(),
            paths.root().join("generation_config.json")
        );
        assert_eq!(paths.tokenizer(), paths.root().join("tokenizer.json"));
        assert_eq!(
            paths.model_weights(),
            paths.root().join("model.safetensors")
        );
    }

    #[test]
    fn missing_artifact_directory_fails_explicitly() {
        let error = GenerationService::from_artifact_dir(
            "/definitely-not-a-real-nilemini-artifact-directory",
        )
        .expect_err("missing artifacts must fail");
        assert!(error.to_string().contains("model configuration"));
    }
}
