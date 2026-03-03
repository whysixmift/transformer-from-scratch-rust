# Transformer From Scratch (Rust)

A compact, modular decoder-only Transformer implementation in Rust, built from scratch for learning and experimentation.

## Highlights

- Byte-level BPE tokenizer with train/save/load support
- Decoder-only Transformer with:
  - RMSNorm
  - RoPE positional encoding
  - SwiGLU MLP
  - Grouped-Query Attention (GQA)
- Training utilities:
  - Cross-entropy loss + gradient for LM head
  - AdamW optimizer
  - Cosine LR schedule with warmup
  - Gradient clipping
- Text generation:
  - Greedy decoding
  - Sampling with temperature, top-k, top-p, repetition penalty
  - Beam search with length penalty and optional EOS handling
- Binary checkpointing for model and tokenizer

## Project Structure

- `src/model.rs` - Tensor utilities, layers, Transformer model, checkpoint IO
- `src/tokenizer.rs` - BPE tokenizer and LM dataset builder
- `src/training.rs` - Optimizer, loss/grad helpers, training loop
- `src/sampling.rs` - Sampling and beam-search decoding
- `src/presets.rs` - Model size presets
- `src/lib.rs` - Library exports + tests
- `src/main.rs` - End-to-end demo: train, save, load, generate

## Quick Start

```bash
cargo run
```

This will:

1. Train a tokenizer on a small in-file corpus
2. Build a model from preset config
3. Train LM head parameters
4. Save `model.bin` and `tokenizer.bin`
5. Reload artifacts and generate text

## Tests

```bash
cargo test
```

## Notes

- This code is intentionally educational and CPU-friendly, not production-optimized.
- Tensor operations are implemented manually for clarity.
