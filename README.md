# Rust Language Engine Lab

A compact, modular decoder-only language engine in Rust, built for learning and experimentation.

## Highlights

- Byte-level BPE token codec with train/save/load support
- Decoder-only language engine with:
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
- Binary checkpointing for the engine and token codec

## Project Structure

- `src/engine.rs` - Tensor utilities, layers, language engine, checkpoint IO
- `src/text_codec.rs` - BPE token codec and LM dataset builder
- `src/fit.rs` - Optimizer, loss/grad helpers, training loop
- `src/infer.rs` - Sampling and beam-search decoding
- `src/profiles.rs` - Model size profiles
- `src/lib.rs` - Library exports + tests
- `src/main.rs` - End-to-end demo: train, save, load, generate

## Quick Start

```bash
cargo run
```

This will:

1. Train a token codec on a small in-file corpus
2. Build an engine from a profile config
3. Train LM head parameters
4. Save `engine.bin` and `text_codec.bin`
5. Reload artifacts and generate text

## Tests

```bash
cargo test
```

## Notes

- This code is intentionally educational and CPU-friendly, not production-optimized.
- Tensor operations are implemented manually for clarity.
