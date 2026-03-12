# Transformer From Scratch Written in Rust.

So, now im working on a project thats kinda complex, im making a "decoder-only" language model project written in Rust for learning, experimenting, and also understanding on HOW does modern text generation systems such as Large Language Model especially Transformer work at a lower level.

As if this project are for learning and experimenting, it focuses on clarity and structure implementating rather than a production scale optimization. What ive been implemented was includes a byte-level tokenizer, a decoder only transformer engine, training utilities, checkpoint saving and loading, and a text generation features. 

## What This Project Includes

 - A byte-level BPE Tokenizer with training, saving, and loading features support.
 - A decoder-only transformer language engine
 - RMSNorm
 - RoPE Positional Encoding
 - SwiGLU FFL (Feed Forward Layers)
 - Grouped-Query Attention Layer
 - Training utilities such as.. cross-entropy loss, gradient handling, AdamW, cosine learning rate scheduling with warup, and gradient clipping.
 - Text generation with greedy decoding, temp sampling, top-k, top-p, repetition penalty, and beam search.
 - A binary checkpoint saving and loading for the model and tokenizer.

## Why Does This Project Exists?

This project was built by myself, as a 16 year old AI Enthusiast and Robotics Engineering, it exists because it was built for educational implementation for myself, to better understand how transformer language models work internally using plain low level language such as Rust. The goal is to make the code practical enough to run small experiments without relying on large external deep learning frameworks. 

## Project Structure

- `engine.rs` - it contains a tensor utilities, layers, model logic and also the checkpoint IO
- `text_codec.rs` - this code contains the BPE Tokenizer and the dataset helpers
- `fit.rs` - contains optimizer and the training logic
- `infer.rs` - contains decoding and generation methods
- `profiles.rs` - it contains model configuration presets
- `lib.rs` - for exposing the library modules and testing purpose
- `main.rs` - runs the end-to-end demo

## How to run it?

You can build and run the project in release oor debug mode with Cargo, the program trains  small tokenizer, builds a model, trains language mmodel head parameters, saves the artifacts, reloads them, and gemerates sample text.

https://github.com/whysixmift/transformer-from-scratch-rust/releases/tag/v.1.0.0

Prebuilt binaries are provided in the GitHub Releases page for Linux, Windows, and MacOS.

## NOTE FOR USERS

this project is educational (especially for myself) and CPU-friendlu. but it is NOT intended to be a production ready machine learning system. the implementation emphasizes readability and experimentation over speed and scale. and please, Star this repo if u like it! Thanks You!
