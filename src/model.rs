use std::f32;
use std::fs::File;
use std::io::{self, Read, Write};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn from_fn<F: FnMut(usize) -> f32>(shape: &[usize], mut f: F) -> Self {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push(f(i));
        }
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn idx(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut flat = 0usize;
        let mut stride = 1usize;
        for (i, &dim) in self.shape.iter().enumerate().rev() {
            let ix = indices[i];
            assert!(ix < dim);
            flat += ix * stride;
            stride *= dim;
        }
        flat
    }

    pub fn get(&self, indices: &[usize]) -> f32 {
        let i = self.idx(indices);
        self.data[i]
    }

    pub fn set(&mut self, indices: &[usize], value: f32) {
        let i = self.idx(indices);
        self.data[i] = value;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let mut out = self.clone();
        for i in 0..out.data.len() {
            out.data[i] += other.data[i];
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u32();
        (u as f32) / (u32::MAX as f32)
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    pub fn gen_usize(&mut self, upper_exclusive: usize) -> usize {
        (self.next_u32() as usize) % upper_exclusive
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn softmax(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn xavier_init(rng: &mut SimpleRng, fan_in: usize, fan_out: usize, shape: &[usize]) -> Tensor {
    let limit = (6.0f32 / (fan_in + fan_out) as f32).sqrt();
    Tensor::from_fn(shape, |_| rng.uniform(-limit, limit))
}

#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    in_dim: usize,
    out_dim: usize,
}

impl Linear {
    pub fn new(rng: &mut SimpleRng, in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight: xavier_init(rng, in_dim, out_dim, &[in_dim, out_dim]),
            bias: Tensor::zeros(&[out_dim]),
            in_dim,
            out_dim,
        }
    }

    fn from_tensors(weight: Tensor, bias: Tensor) -> Self {
        assert_eq!(weight.rank(), 2);
        assert_eq!(bias.rank(), 1);
        assert_eq!(weight.shape[1], bias.shape[0]);
        Self {
            in_dim: weight.shape[0],
            out_dim: weight.shape[1],
            weight,
            bias,
        }
    }

    pub fn forward_3d(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 3);
        assert_eq!(x.shape[2], self.in_dim);
        let b = x.shape[0];
        let s = x.shape[1];
        let mut out = Tensor::zeros(&[b, s, self.out_dim]);
        for bi in 0..b {
            for ti in 0..s {
                for o in 0..self.out_dim {
                    let mut sum = self.bias.get(&[o]);
                    for i in 0..self.in_dim {
                        sum += x.get(&[bi, ti, i]) * self.weight.get(&[i, o]);
                    }
                    out.set(&[bi, ti, o], sum);
                }
            }
        }
        out
    }

    pub fn forward_2d(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 2);
        assert_eq!(x.shape[1], self.in_dim);
        let n = x.shape[0];
        let mut out = Tensor::zeros(&[n, self.out_dim]);
        for r in 0..n {
            for o in 0..self.out_dim {
                let mut sum = self.bias.get(&[o]);
                for i in 0..self.in_dim {
                    sum += x.get(&[r, i]) * self.weight.get(&[i, o]);
                }
                out.set(&[r, o], sum);
            }
        }
        out
    }
}

#[derive(Clone, Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f32,
    dim: usize,
}

impl RmsNorm {
    fn new(dim: usize) -> Self {
        Self {
            weight: Tensor::from_fn(&[dim], |_| 1.0),
            eps: 1e-5,
            dim,
        }
    }

    fn from_weight(weight: Tensor, eps: f32) -> Self {
        assert_eq!(weight.rank(), 1);
        Self {
            dim: weight.shape[0],
            weight,
            eps,
        }
    }

    fn forward_3d(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.rank(), 3);
        assert_eq!(x.shape[2], self.dim);
        let b = x.shape[0];
        let s = x.shape[1];
        let d = self.dim;
        let mut out = Tensor::zeros(&[b, s, d]);

        for bi in 0..b {
            for ti in 0..s {
                let mut mean_sq = 0.0;
                for i in 0..d {
                    let v = x.get(&[bi, ti, i]);
                    mean_sq += v * v;
                }
                mean_sq /= d as f32;
                let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();

                for i in 0..d {
                    let y = x.get(&[bi, ti, i]) * inv_rms * self.weight.get(&[i]);
                    out.set(&[bi, ti, i], y);
                }
            }
        }

        out
    }
}

fn apply_rope(q_or_k: &mut Tensor, num_heads: usize, head_dim: usize, rope_theta: f32) {
    assert_eq!(q_or_k.rank(), 3);
    assert_eq!(q_or_k.shape[2], num_heads * head_dim);
    assert!(head_dim % 2 == 0);

    let b = q_or_k.shape[0];
    let s = q_or_k.shape[1];

    for bi in 0..b {
        for pos in 0..s {
            for h in 0..num_heads {
                for i in 0..(head_dim / 2) {
                    let d0 = h * head_dim + 2 * i;
                    let d1 = d0 + 1;

                    let x0 = q_or_k.get(&[bi, pos, d0]);
                    let x1 = q_or_k.get(&[bi, pos, d1]);

                    let freq = rope_theta.powf(-(2.0 * i as f32) / head_dim as f32);
                    let angle = pos as f32 * freq;
                    let (sin_a, cos_a) = angle.sin_cos();

                    let y0 = x0 * cos_a - x1 * sin_a;
                    let y1 = x0 * sin_a + x1 * cos_a;
                    q_or_k.set(&[bi, pos, d0], y0);
                    q_or_k.set(&[bi, pos, d1], y1);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CausalSelfAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    q_per_kv: usize,
    d_model: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl CausalSelfAttention {
    fn new(
        rng: &mut SimpleRng,
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
    ) -> Self {
        assert!(d_model % num_heads == 0);
        assert!(num_heads % num_kv_heads == 0);
        let head_dim = d_model / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        Self {
            w_q: Linear::new(rng, d_model, d_model),
            w_k: Linear::new(rng, d_model, kv_dim),
            w_v: Linear::new(rng, d_model, kv_dim),
            w_o: Linear::new(rng, d_model, d_model),
            num_heads,
            num_kv_heads,
            q_per_kv: num_heads / num_kv_heads,
            d_model,
            head_dim,
            rope_theta,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut q = self.w_q.forward_3d(x);
        let mut k = self.w_k.forward_3d(x);
        let v = self.w_v.forward_3d(x);

        apply_rope(&mut q, self.num_heads, self.head_dim, self.rope_theta);
        apply_rope(&mut k, self.num_kv_heads, self.head_dim, self.rope_theta);

        let b = x.shape[0];
        let s = x.shape[1];
        let h = self.num_heads;
        let hd = self.head_dim;

        let mut context = Tensor::zeros(&[b, s, self.d_model]);
        let scale = 1.0 / (hd as f32).sqrt();

        for bi in 0..b {
            for head in 0..h {
                let mut scores = vec![0.0f32; s * s];
                for i in 0..s {
                    for j in 0..s {
                        let mut dot = 0.0;
                        let kv_head = head / self.q_per_kv;
                        for d in 0..hd {
                            let qv = q.get(&[bi, i, head * hd + d]);
                            let kv = k.get(&[bi, j, kv_head * hd + d]);
                            dot += qv * kv;
                        }
                        let mut score = dot * scale;
                        if j > i {
                            score = -1e9;
                        }
                        scores[i * s + j] = score;
                    }
                }

                for i in 0..s {
                    let row = &scores[i * s..(i + 1) * s];
                    let probs = softmax(row);

                    for d in 0..hd {
                        let mut sum = 0.0;
                        let kv_head = head / self.q_per_kv;
                        for (j, &p) in probs.iter().enumerate() {
                            let vv = v.get(&[bi, j, kv_head * hd + d]);
                            sum += p * vv;
                        }
                        context.set(&[bi, i, head * hd + d], sum);
                    }
                }
            }
        }

        self.w_o.forward_3d(&context)
    }
}

#[derive(Clone, Debug)]
struct SwiGluMlp {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
    d_ff: usize,
}

impl SwiGluMlp {
    fn new(rng: &mut SimpleRng, d_model: usize, d_ff: usize) -> Self {
        Self {
            w_gate: Linear::new(rng, d_model, d_ff),
            w_up: Linear::new(rng, d_model, d_ff),
            w_down: Linear::new(rng, d_ff, d_model),
            d_ff,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.w_gate.forward_3d(x);
        let up = self.w_up.forward_3d(x);
        let b = gate.shape[0];
        let s = gate.shape[1];
        let mut fused = Tensor::zeros(&[b, s, self.d_ff]);

        for bi in 0..b {
            for ti in 0..s {
                for d in 0..self.d_ff {
                    let g = gate.get(&[bi, ti, d]);
                    let u = up.get(&[bi, ti, d]);
                    fused.set(&[bi, ti, d], silu(g) * u);
                }
            }
        }

        self.w_down.forward_3d(&fused)
    }
}

#[derive(Clone, Debug)]
struct DecoderBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: CausalSelfAttention,
    mlp: SwiGluMlp,
}

impl DecoderBlock {
    fn new(
        rng: &mut SimpleRng,
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        d_ff: usize,
        rope_theta: f32,
    ) -> Self {
        Self {
            norm1: RmsNorm::new(d_model),
            norm2: RmsNorm::new(d_model),
            attn: CausalSelfAttention::new(rng, d_model, num_heads, num_kv_heads, rope_theta),
            mlp: SwiGluMlp::new(rng, d_model, d_ff),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let n1 = self.norm1.forward_3d(x);
        let attn_out = self.attn.forward(&n1);
        let x = x.add(&attn_out);

        let n2 = self.norm2.forward_3d(&x);
        let mlp_out = self.mlp.forward(&n2);
        x.add(&mlp_out)
    }
}

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub d_ff: usize,
    pub rope_theta: f32,
}

#[derive(Clone, Debug)]
pub struct Transformer {
    pub config: TransformerConfig,
    token_embedding: Tensor,
    blocks: Vec<DecoderBlock>,
    norm_f: RmsNorm,
    pub lm_head: Linear,
}

impl Transformer {
    pub fn new(config: TransformerConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let token_embedding = xavier_init(
            &mut rng,
            config.vocab_size,
            config.d_model,
            &[config.vocab_size, config.d_model],
        );
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(DecoderBlock::new(
                &mut rng,
                config.d_model,
                config.num_heads,
                config.num_kv_heads,
                config.d_ff,
                config.rope_theta,
            ));
        }
        let norm_f = RmsNorm::new(config.d_model);
        let lm_head = Linear::new(&mut rng, config.d_model, config.vocab_size);

        Self {
            config,
            token_embedding,
            blocks,
            norm_f,
            lm_head,
        }
    }

    fn embed_tokens(&self, input_ids: &[usize]) -> Tensor {
        assert!(input_ids.len() <= self.config.max_seq_len);
        let seq = input_ids.len();
        let mut out = Tensor::zeros(&[1, seq, self.config.d_model]);
        for (t, &token) in input_ids.iter().enumerate() {
            assert!(token < self.config.vocab_size);
            for d in 0..self.config.d_model {
                out.set(&[0, t, d], self.token_embedding.get(&[token, d]));
            }
        }
        out
    }

    pub fn forward_hidden(&self, input_ids: &[usize]) -> Tensor {
        let mut x = self.embed_tokens(input_ids);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        let x = self.norm_f.forward_3d(&x);

        let seq = x.shape[1];
        let mut flat = Tensor::zeros(&[seq, self.config.d_model]);
        for t in 0..seq {
            for d in 0..self.config.d_model {
                flat.set(&[t, d], x.get(&[0, t, d]));
            }
        }
        flat
    }

    pub fn forward_logits(&self, input_ids: &[usize]) -> Tensor {
        let hidden = self.forward_hidden(input_ids);
        self.lm_head.forward_2d(&hidden)
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(b"TRM2")?;
        write_u64(&mut f, self.config.vocab_size as u64)?;
        write_u64(&mut f, self.config.max_seq_len as u64)?;
        write_u64(&mut f, self.config.d_model as u64)?;
        write_u64(&mut f, self.config.num_heads as u64)?;
        write_u64(&mut f, self.config.num_kv_heads as u64)?;
        write_u64(&mut f, self.config.num_layers as u64)?;
        write_u64(&mut f, self.config.d_ff as u64)?;
        write_f32(&mut f, self.config.rope_theta)?;

        write_tensor(&mut f, &self.token_embedding)?;
        for b in &self.blocks {
            write_tensor(&mut f, &b.norm1.weight)?;
            write_tensor(&mut f, &b.norm2.weight)?;
            write_linear(&mut f, &b.attn.w_q)?;
            write_linear(&mut f, &b.attn.w_k)?;
            write_linear(&mut f, &b.attn.w_v)?;
            write_linear(&mut f, &b.attn.w_o)?;
            write_linear(&mut f, &b.mlp.w_gate)?;
            write_linear(&mut f, &b.mlp.w_up)?;
            write_linear(&mut f, &b.mlp.w_down)?;
        }
        write_tensor(&mut f, &self.norm_f.weight)?;
        write_linear(&mut f, &self.lm_head)?;
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        let is_v1 = &magic == b"TRM1";
        let is_v2 = &magic == b"TRM2";
        if !is_v1 && !is_v2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid model magic",
            ));
        }

        let vocab_size = read_u64(&mut f)? as usize;
        let max_seq_len = read_u64(&mut f)? as usize;
        let d_model = read_u64(&mut f)? as usize;
        let num_heads = read_u64(&mut f)? as usize;
        let num_kv_heads = if is_v2 {
            read_u64(&mut f)? as usize
        } else {
            num_heads
        };

        let config = TransformerConfig {
            vocab_size,
            max_seq_len,
            d_model,
            num_heads,
            num_kv_heads,
            num_layers: read_u64(&mut f)? as usize,
            d_ff: read_u64(&mut f)? as usize,
            rope_theta: read_f32(&mut f)?,
        };

        let token_embedding = read_tensor(&mut f)?;
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let norm1 = RmsNorm::from_weight(read_tensor(&mut f)?, 1e-5);
            let norm2 = RmsNorm::from_weight(read_tensor(&mut f)?, 1e-5);
            let w_q = read_linear(&mut f)?;
            let w_k = read_linear(&mut f)?;
            let w_v = read_linear(&mut f)?;
            let w_o = read_linear(&mut f)?;
            let w_gate = read_linear(&mut f)?;
            let w_up = read_linear(&mut f)?;
            let w_down = read_linear(&mut f)?;

            blocks.push(DecoderBlock {
                norm1,
                norm2,
                attn: CausalSelfAttention {
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    q_per_kv: config.num_heads / config.num_kv_heads,
                    d_model: config.d_model,
                    head_dim: config.d_model / config.num_heads,
                    rope_theta: config.rope_theta,
                    w_q,
                    w_k,
                    w_v,
                    w_o,
                },
                mlp: SwiGluMlp {
                    d_ff: config.d_ff,
                    w_gate,
                    w_up,
                    w_down,
                },
            });
        }

        let norm_f = RmsNorm::from_weight(read_tensor(&mut f)?, 1e-5);
        let lm_head = read_linear(&mut f)?;

        Ok(Self {
            config,
            token_embedding,
            blocks,
            norm_f,
            lm_head,
        })
    }
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_tensor<W: Write>(w: &mut W, t: &Tensor) -> io::Result<()> {
    write_u64(w, t.shape.len() as u64)?;
    for &d in &t.shape {
        write_u64(w, d as u64)?;
    }
    write_u64(w, t.data.len() as u64)?;
    for &v in &t.data {
        write_f32(w, v)?;
    }
    Ok(())
}

fn read_tensor<R: Read>(r: &mut R) -> io::Result<Tensor> {
    let rank = read_u64(r)? as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        shape.push(read_u64(r)? as usize);
    }
    let size = read_u64(r)? as usize;
    let expected: usize = shape.iter().product();
    if size != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tensor size mismatch",
        ));
    }
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(read_f32(r)?);
    }
    Ok(Tensor { data, shape })
}

fn write_linear<W: Write>(w: &mut W, l: &Linear) -> io::Result<()> {
    write_tensor(w, &l.weight)?;
    write_tensor(w, &l.bias)
}

fn read_linear<R: Read>(r: &mut R) -> io::Result<Linear> {
    let weight = read_tensor(r)?;
    let bias = read_tensor(r)?;
    Ok(Linear::from_tensors(weight, bias))
}
