//! Dedicated CPU batching thread for concurrent HTTP generation requests.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{mpsc, Arc};
use std::thread;

use smalllm_engine::sampler::SamplingConfig;
use smalllm_engine::scheduler::{GenerationScheduler, SchedulerConfig, SequenceId};
use smalllm_engine::tokenizer::SmallLMTokenizer;
use smalllm_engine::{
    ArtifactPaths, BatchedDecodeModel, EngineError, GenerationConfig, ModelConfig,
};
use tokio::sync::oneshot;

use crate::errors::ApiError;

/// Completed token-level output returned by the batching runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchGenerationOutput {
    /// Newly generated token identifiers.
    pub generated_token_ids: Vec<u32>,
    /// Engine finish reason: `eos` or `length`.
    pub finish_reason: String,
}

struct GenerationCommand {
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
    sampling: SamplingConfig,
    response: oneshot::Sender<Result<BatchGenerationOutput, ApiError>>,
}

struct PendingRequest {
    generated_token_ids: Vec<u32>,
    response: oneshot::Sender<Result<BatchGenerationOutput, ApiError>>,
}

/// Cloneable async handle to one long-lived CPU batching thread.
#[derive(Clone, Debug)]
pub struct BatchRuntime {
    sender: mpsc::Sender<GenerationCommand>,
    tokenizer: Arc<SmallLMTokenizer>,
    model_name: String,
}

impl BatchRuntime {
    /// Load artifacts and start the dedicated batching thread.
    ///
    /// # Errors
    ///
    /// Returns configuration, tokenizer, model-weight, or scheduler errors.
    pub fn from_artifact_dir(
        root: impl AsRef<Path>,
        max_active_sequences: usize,
    ) -> Result<Self, EngineError> {
        let paths = ArtifactPaths::new(root);
        let model_config = ModelConfig::from_json_path(paths.model_config())?;
        let generation_config =
            GenerationConfig::from_json_path(paths.generation_config(), &model_config)?;
        let tokenizer = SmallLMTokenizer::from_file(paths.tokenizer())?;
        tokenizer.validate_contract(&model_config, &generation_config)?;

        let mut model = BatchedDecodeModel::from_config(model_config.clone())?;
        model.load_weights(paths.model_weights())?;
        let scheduler_config = SchedulerConfig {
            max_active_sequences,
        };
        scheduler_config.validate()?;

        let (sender, receiver) = mpsc::channel();
        let eos_token_id = generation_config.eos_token_id;
        thread::Builder::new()
            .name("smalllm-batch-worker".to_owned())
            .spawn(move || run_scheduler(model, receiver, scheduler_config, eos_token_id))
            .map_err(|source| {
                EngineError::invalid_input(
                    "batch worker",
                    format!("failed to spawn scheduler thread: {source}"),
                )
            })?;

        Ok(Self {
            sender,
            tokenizer: Arc::new(tokenizer),
            model_name: model_config.model_name,
        })
    }

    /// Return the served model identifier.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Encode prompt text with the bundled tokenizer.
    ///
    /// # Errors
    ///
    /// Returns a tokenizer error if encoding fails.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, EngineError> {
        self.tokenizer.encode(text, add_special_tokens)
    }

    /// Decode generated token identifiers with the bundled tokenizer.
    ///
    /// # Errors
    ///
    /// Returns a tokenizer error if decoding fails.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, EngineError> {
        self.tokenizer.decode(token_ids, true)
    }

    /// Submit one request and wait asynchronously for its terminal token.
    ///
    /// # Errors
    ///
    /// Returns request/model errors from admission or a worker error if the
    /// dedicated scheduler thread is unavailable.
    pub async fn generate(
        &self,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        sampling: SamplingConfig,
    ) -> Result<BatchGenerationOutput, ApiError> {
        let (response, receiver) = oneshot::channel();
        self.sender
            .send(GenerationCommand {
                prompt_token_ids,
                max_new_tokens,
                sampling,
                response,
            })
            .map_err(|_| ApiError::Worker {
                message: "batch scheduler thread is not available".to_owned(),
            })?;

        receiver.await.map_err(|_| ApiError::Worker {
            message: "batch scheduler stopped before completing the request".to_owned(),
        })?
    }
}

fn run_scheduler(
    model: BatchedDecodeModel,
    receiver: mpsc::Receiver<GenerationCommand>,
    config: SchedulerConfig,
    eos_token_id: u32,
) {
    let mut scheduler = GenerationScheduler::new(config)
        .expect("batch worker starts only after scheduler configuration validation");
    let mut pending = HashMap::<SequenceId, PendingRequest>::new();
    let mut disconnected = false;

    loop {
        if scheduler.active_sequence_count() == 0 && !disconnected {
            match receiver.recv() {
                Ok(command) => {
                    admit_command(&model, &mut scheduler, &mut pending, command, eos_token_id)
                }
                Err(_) => disconnected = true,
            }
        }

        while scheduler.has_capacity() && !disconnected {
            match receiver.try_recv() {
                Ok(command) => {
                    admit_command(&model, &mut scheduler, &mut pending, command, eos_token_id)
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if scheduler.active_sequence_count() == 0 {
            if disconnected {
                break;
            }
            continue;
        }

        match scheduler.step(&model) {
            Ok(events) => {
                for event in events {
                    let Some(request) = pending.get_mut(&event.sequence_id) else {
                        fail_pending(
                            &mut pending,
                            "scheduler emitted an event for an unknown request",
                        );
                        scheduler = GenerationScheduler::new(config)
                            .expect("validated scheduler config remains valid");
                        break;
                    };
                    request.generated_token_ids.push(event.token_id);

                    if let Some(reason) = event.finish_reason {
                        let request = pending
                            .remove(&event.sequence_id)
                            .expect("terminal request exists after mutable lookup");
                        let _ = request.response.send(Ok(BatchGenerationOutput {
                            generated_token_ids: request.generated_token_ids,
                            finish_reason: reason.as_str().to_owned(),
                        }));
                    }
                }
            }
            Err(error) => {
                fail_pending(&mut pending, format!("batched decode failed: {error}"));
                scheduler = GenerationScheduler::new(config)
                    .expect("validated scheduler config remains valid");
            }
        }
    }

    fail_pending(&mut pending, "batch scheduler shut down before completion");
}

fn admit_command(
    model: &BatchedDecodeModel,
    scheduler: &mut GenerationScheduler,
    pending: &mut HashMap<SequenceId, PendingRequest>,
    command: GenerationCommand,
    eos_token_id: u32,
) {
    let GenerationCommand {
        prompt_token_ids,
        max_new_tokens,
        sampling,
        response,
    } = command;

    match scheduler.admit(
        model,
        &prompt_token_ids,
        max_new_tokens,
        Some(eos_token_id),
        sampling,
    ) {
        Ok(sequence_id) => {
            pending.insert(
                sequence_id,
                PendingRequest {
                    generated_token_ids: Vec::with_capacity(max_new_tokens),
                    response,
                },
            );
        }
        Err(error) => {
            let _ = response.send(Err(ApiError::from(error)));
        }
    }
}

fn fail_pending(pending: &mut HashMap<SequenceId, PendingRequest>, message: impl Into<String>) {
    let message = message.into();
    for (_, request) in pending.drain() {
        let _ = request.response.send(Err(ApiError::Worker {
            message: message.clone(),
        }));
    }
}
