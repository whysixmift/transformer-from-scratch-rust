use std::io;

use transformer_from_scratch::{
    BpeTokenizer, GenerateConfig, ModelPreset, TrainingConfig, Transformer,
    build_language_model_dataset, generate_text, train_lm_head_only,
};

fn main() -> io::Result<()> {
    let corpus = "
    Rust makes systems programming safe and fast.
    Transformers learn patterns in sequences.
    A decoder-only language model predicts the next token.
    This tiny project demonstrates tokenizer training, model inference, training loops, and checkpoint IO.
    ";

    let tokenizer = BpeTokenizer::train(corpus, 220);
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

    let tokens = tokenizer.encode(corpus);
    let block_size = 24;
    let dataset = build_language_model_dataset(&tokens, block_size, 1);
    println!("Training samples: {}", dataset.len());

    let config = ModelPreset::Balanced.build(tokenizer.vocab_size(), block_size);
    let mut model = Transformer::new(config, 1234);

    let train_cfg = TrainingConfig {
        steps: 80,
        batch_size: 2,
        grad_accum_steps: 2,
        lr: 3e-3,
        min_lr_ratio: 0.1,
        warmup_steps: 8,
        weight_decay: 1e-4,
        max_grad_norm: 1.0,
    };
    train_lm_head_only(&mut model, &dataset, &train_cfg, 99);

    model.save("model.bin")?;
    tokenizer.save("tokenizer.bin")?;
    println!("Saved model.bin and tokenizer.bin");

    let loaded_model = Transformer::load("model.bin")?;
    let loaded_tokenizer = BpeTokenizer::load("tokenizer.bin")?;

    let gen_cfg = GenerateConfig {
        max_new_tokens: 40,
        temperature: 0.9,
        top_k: 20,
        top_p: 0.92,
        repetition_penalty: 1.1,
        beam_width: 1,
        beam_top_candidates: 8,
        length_penalty: 0.8,
        eos_token_id: None,
    };
    let generated = generate_text(
        &loaded_model,
        &loaded_tokenizer,
        "Rust makes",
        &gen_cfg,
        2026,
    );
    println!("Generated text:\n{}", generated);

    Ok(())
}
