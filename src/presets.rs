use crate::TransformerConfig;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum ModelPreset {
    Tiny,
    Balanced,
    Wide,
}

impl ModelPreset {
    pub fn build(self, vocab_size: usize, max_seq_len: usize) -> TransformerConfig {
        match self {
            Self::Tiny => TransformerConfig {
                vocab_size,
                max_seq_len,
                d_model: 64,
                num_heads: 4,
                num_kv_heads: 2,
                num_layers: 4,
                d_ff: 176,
                rope_theta: 10_000.0,
            },
            Self::Balanced => TransformerConfig {
                vocab_size,
                max_seq_len,
                d_model: 96,
                num_heads: 6,
                num_kv_heads: 2,
                num_layers: 6,
                d_ff: 288,
                rope_theta: 10_000.0,
            },
            Self::Wide => TransformerConfig {
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
