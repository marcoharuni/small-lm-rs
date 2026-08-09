//! Deterministic next-token sampling with temperature, top-k, and top-p.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// User-selectable controls for next-token sampling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingConfig {
    /// Logit temperature; zero selects greedy decoding.
    pub temperature: f32,
    /// Nucleus-sampling cumulative probability threshold.
    pub top_p: f32,
    /// Maximum number of candidates retained before sampling; zero disables it.
    pub top_k: usize,
    /// Seed for deterministic pseudorandom sampling.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 0,
            seed: 0,
        }
    }
}

impl SamplingConfig {
    /// Validate all sampling controls.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for a negative/non-finite
    /// temperature or a `top_p` value outside `(0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(EngineError::invalid_configuration(
                "sampling temperature must be finite and non-negative",
            ));
        }
        if !(self.top_p.is_finite() && 0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(EngineError::invalid_configuration(
                "sampling top_p must be finite and in the range (0, 1]",
            ));
        }
        Ok(())
    }
}

/// Stateful deterministic next-token sampler.
#[derive(Clone, Debug)]
pub struct Sampler {
    config: SamplingConfig,
    state: u64,
}

impl Sampler {
    /// Create a sampler after validating its controls.
    ///
    /// # Errors
    ///
    /// Returns an invalid-configuration error for invalid controls.
    pub fn new(config: SamplingConfig) -> Result<Self> {
        config.validate()?;
        let state = config.seed;
        Ok(Self { config, state })
    }

    /// Return the active sampling controls.
    #[must_use]
    pub const fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Choose one token from a full-vocabulary logit vector.
    ///
    /// Temperature zero performs deterministic argmax. Positive temperature
    /// applies top-k first, then nucleus filtering, then seeded sampling.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error for empty/non-finite logits or a token
    /// index that cannot fit in `u32`.
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        validate_logits(logits)?;
        if matches!(
            self.config.temperature.classify(),
            std::num::FpCategory::Zero
        ) {
            return index_to_token(argmax(logits));
        }

        let temperature = f64::from(self.config.temperature);
        let mut candidates = (0..logits.len()).collect::<Vec<_>>();
        candidates.sort_by(|&left, &right| {
            logits[right]
                .total_cmp(&logits[left])
                .then_with(|| left.cmp(&right))
        });
        if self.config.top_k > 0 && candidates.len() > self.config.top_k {
            candidates.truncate(self.config.top_k);
        }

        let maximum = f64::from(logits[candidates[0]]) / temperature;
        let mut weighted = candidates
            .into_iter()
            .map(|index| {
                let scaled = f64::from(logits[index]) / temperature;
                (index, (scaled - maximum).exp())
            })
            .collect::<Vec<_>>();
        let total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
        let mut cumulative = 0.0_f64;
        let mut retained = weighted.len();
        for (position, (_, weight)) in weighted.iter().enumerate() {
            cumulative += *weight / total;
            if cumulative >= f64::from(self.config.top_p) {
                retained = position + 1;
                break;
            }
        }
        weighted.truncate(retained.max(1));

        let retained_total = weighted.iter().map(|(_, weight)| *weight).sum::<f64>();
        let target = self.next_unit_f64() * retained_total;
        let mut running = 0.0_f64;
        for (index, weight) in &weighted {
            running += *weight;
            if target < running {
                return index_to_token(*index);
            }
        }
        index_to_token(weighted.last().expect("at least one candidate").0)
    }

    fn next_unit_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        let mantissa = value >> 11;
        mantissa as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn validate_logits(logits: &[f32]) -> Result<()> {
    if logits.is_empty() {
        return Err(EngineError::invalid_input(
            "sampler",
            "the logits vector must not be empty",
        ));
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(EngineError::invalid_input(
            "sampler",
            "all logits must be finite",
        ));
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_index = 0_usize;
    let mut best_value = logits[0];
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index
}

fn index_to_token(index: usize) -> Result<u32> {
    u32::try_from(index)
        .map_err(|_| EngineError::invalid_input("sampler", "selected token index exceeds u32"))
}

#[cfg(test)]
mod tests {
    use super::{Sampler, SamplingConfig};

    #[test]
    fn zero_temperature_selects_first_maximum() {
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        })
        .expect("valid sampler");
        assert_eq!(sampler.sample(&[1.0, 4.0, 4.0]).expect("sample"), 1);
    }

    #[test]
    fn top_k_one_always_selects_the_largest_logit() {
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 7,
        })
        .expect("valid sampler");
        for _ in 0..8 {
            assert_eq!(sampler.sample(&[-2.0, 3.0, 1.0]).expect("sample"), 1);
        }
    }

    #[test]
    fn top_p_can_retain_only_the_leading_candidate() {
        let mut sampler = Sampler::new(SamplingConfig {
            temperature: 1.0,
            top_p: 0.5,
            top_k: 0,
            seed: 19,
        })
        .expect("valid sampler");
        for _ in 0..8 {
            assert_eq!(sampler.sample(&[8.0, 0.0, -1.0]).expect("sample"), 0);
        }
    }

    #[test]
    fn equal_seeds_produce_equal_sample_sequences() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
        };
        let mut left = Sampler::new(config.clone()).expect("valid sampler");
        let mut right = Sampler::new(config).expect("valid sampler");
        let left_tokens = (0..16)
            .map(|_| left.sample(&[0.0, 0.0, 0.0]).expect("sample"))
            .collect::<Vec<_>>();
        let right_tokens = (0..16)
            .map(|_| right.sample(&[0.0, 0.0, 0.0]).expect("sample"))
            .collect::<Vec<_>>();
        assert_eq!(left_tokens, right_tokens);
    }

    #[test]
    fn invalid_logits_are_rejected() {
        let mut sampler = Sampler::new(SamplingConfig::default()).expect("valid sampler");
        assert!(sampler.sample(&[]).is_err());
        assert!(sampler.sample(&[0.0, f32::NAN]).is_err());
    }
}
