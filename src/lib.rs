pub mod model;
pub mod presets;
pub mod sampling;
pub mod tokenizer;
pub mod training;

pub use model::{SimpleRng, Tensor, Transformer, TransformerConfig};
pub use presets::ModelPreset;
pub use sampling::{GenerateConfig, generate_text, select_next_token};
pub use tokenizer::{BpeTokenizer, build_language_model_dataset};
pub use training::{TrainingConfig, cosine_lr, cross_entropy_with_grad, train_lm_head_only};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_roundtrip() {
        let text = "hello transformers";
        let tok = BpeTokenizer::train(text, 50);
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn cross_entropy_grad_shape() {
        let logits = Tensor::from_fn(&[3, 5], |i| (i as f32) * 0.01);
        let targets = vec![1usize, 2, 3];
        let (_, grad) = cross_entropy_with_grad(&logits, &targets);
        assert_eq!(grad.shape, vec![3, 5]);
    }

    #[test]
    fn model_save_load_keeps_logits() {
        let cfg = TransformerConfig {
            vocab_size: 32,
            max_seq_len: 8,
            d_model: 16,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            d_ff: 48,
            rope_theta: 10_000.0,
        };
        let model = Transformer::new(cfg, 7);
        let input = vec![1usize, 3, 5, 7];
        let logits_a = model.forward_logits(&input);

        model.save("test_model.bin").unwrap();
        let loaded = Transformer::load("test_model.bin").unwrap();
        let logits_b = loaded.forward_logits(&input);

        assert_eq!(logits_a.shape, logits_b.shape);
        for (a, b) in logits_a.data.iter().zip(logits_b.data.iter()) {
            assert!((a - b).abs() < 1e-8);
        }

        std::fs::remove_file("test_model.bin").unwrap();
    }

    #[test]
    fn tokenizer_save_load_roundtrip() {
        let tok = BpeTokenizer::train("abc abc abc", 20);
        tok.save("test_tok.bin").unwrap();
        let loaded = BpeTokenizer::load("test_tok.bin").unwrap();

        let text = "abc abc";
        assert_eq!(
            tok.decode(&tok.encode(text)),
            loaded.decode(&loaded.encode(text))
        );

        std::fs::remove_file("test_tok.bin").unwrap();
    }

    #[test]
    fn cosine_lr_bounds() {
        let cfg = TrainingConfig {
            steps: 100,
            batch_size: 1,
            grad_accum_steps: 1,
            lr: 1e-3,
            min_lr_ratio: 0.2,
            warmup_steps: 10,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
        };
        let first = cosine_lr(0, &cfg);
        let mid = cosine_lr(50, &cfg);
        let last = cosine_lr(99, &cfg);
        assert!(first > 0.0 && first <= cfg.lr);
        assert!(mid <= cfg.lr && mid >= cfg.lr * cfg.min_lr_ratio);
        assert!(last >= cfg.lr * cfg.min_lr_ratio - 1e-8);
    }

    #[test]
    fn deterministic_greedy_selection() {
        let logits = vec![0.1, 2.5, 1.2, 0.8];
        let cfg = GenerateConfig {
            max_new_tokens: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            beam_width: 1,
            beam_top_candidates: 4,
            length_penalty: 1.0,
            eos_token_id: None,
        };
        let mut rng = SimpleRng::new(1);
        let next = select_next_token(&logits, &[], &cfg, &mut rng);
        assert_eq!(next, 1);
    }

    #[test]
    fn beam_search_generation_runs() {
        let corpus = "rust rust rust model text";
        let tok = BpeTokenizer::train(corpus, 20);
        let cfg_model = TransformerConfig {
            vocab_size: tok.vocab_size(),
            max_seq_len: 8,
            d_model: 32,
            num_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            d_ff: 64,
            rope_theta: 10_000.0,
        };
        let model = Transformer::new(cfg_model, 11);
        let gen_cfg = GenerateConfig {
            max_new_tokens: 4,
            temperature: 0.9,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            beam_width: 3,
            beam_top_candidates: 5,
            length_penalty: 0.8,
            eos_token_id: None,
        };
        let text = generate_text(&model, &tok, "rust", &gen_cfg, 1);
        assert!(!text.is_empty());
    }
}
