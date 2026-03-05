use std::io;

use transformer_from_scratch::{
    TokenCodec, DecodePlan, ModelProfile, FitConfig, LanguageEngine,
    make_lm_windows, continue_text, fit_head_only,
};

fn main() -> io::Result<()> {
    let corpus = "
    Rust makes systems programming safe and fast.
    Transformers learn patterns in sequences.
    A decoder-only language engine predicts the next token.
    This tiny project demonstrates text_codec fit, engine inference, fit loops, and checkpoint IO.
    ";

    let text_codec = TokenCodec::train(corpus, 220);
    println!("Tokenizer vocab size: {}", text_codec.vocab_size());

    let tokens = text_codec.encode(corpus);
    let block_size = 24;
    let dataset = make_lm_windows(&tokens, block_size, 1);
    println!("Training samples: {}", dataset.len());

    let config = ModelProfile::Balanced.build(text_codec.vocab_size(), block_size);
    let mut engine = LanguageEngine::new(config, 1234);

    let train_cfg = FitConfig {
        steps: 80,
        batch_size: 2,
        grad_accum_steps: 2,
        lr: 3e-3,
        min_lr_ratio: 0.1,
        warmup_steps: 8,
        weight_decay: 1e-4,
        max_grad_norm: 1.0,
    };
    fit_head_only(&mut engine, &dataset, &train_cfg, 99);

    engine.save("engine.bin")?;
    text_codec.save("text_codec.bin")?;
    println!("Saved engine.bin and text_codec.bin");

    let loaded_engine = LanguageEngine::load("engine.bin")?;
    let loaded_codec = TokenCodec::load("text_codec.bin")?;

    let gen_cfg = DecodePlan {
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
    let generated = continue_text(
        &loaded_engine,
        &loaded_codec,
        "Rust makes",
        &gen_cfg,
        2026,
    );
    println!("Generated text:\n{}", generated);

    Ok(())
}
