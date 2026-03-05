use crate::EngineConfig;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum ModelProfile {
    Tiny,
    Balanced,
    Wide,
}

impl ModelProfile {
    pub fn build(self, vocab_size: usize, max_seq_len: usize) -> EngineConfig {
        match self {
            Self::Tiny => EngineConfig {
                vocab_size,
                max_seq_len,
                d_model: 64,
                num_heads: 4,
                num_kv_heads: 2,
                num_layers: 4,
                d_ff: 176,
                rope_theta: 10_000.0,
            },
            Self::Balanced => EngineConfig {
                vocab_size,
                max_seq_len,
                d_model: 96,
                num_heads: 6,
                num_kv_heads: 2,
                num_layers: 6,
                d_ff: 288,
                rope_theta: 10_000.0,
            },
            Self::Wide => EngineConfig {
                vocab_size,
                max_seq_len,
                d_model: 128,
                num_heads: 8,
                num_kv_heads: 2,
                num_layers: 6,
                d_ff: 384,
                rope_theta: 10_000.0,
            },
        }
    }
}
