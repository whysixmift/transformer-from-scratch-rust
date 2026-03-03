use crate::model::{SimpleRng, Tensor, Transformer, softmax};

#[derive(Clone, Debug)]
struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamW {
    fn new(size: usize, lr: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }

    fn update(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(self.m.len(), grads.len());
        self.t += 1;
        let t = self.t as f32;

        for i in 0..params.len() {
            let g = grads[i] + self.weight_decay * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t));

            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub lr: f32,
    pub min_lr_ratio: f32,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
}

pub fn cosine_lr(step: usize, cfg: &TrainingConfig) -> f32 {
    let min_lr = cfg.lr * cfg.min_lr_ratio;
    if cfg.warmup_steps > 0 && step < cfg.warmup_steps {
        return cfg.lr * (step as f32 + 1.0) / cfg.warmup_steps as f32;
    }
    if cfg.steps <= cfg.warmup_steps {
        return min_lr;
    }
    let decay_steps = (cfg.steps - cfg.warmup_steps) as f32;
    let progress = ((step.saturating_sub(cfg.warmup_steps)) as f32 / decay_steps).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    min_lr + (cfg.lr - min_lr) * cosine
}

fn clip_gradients(grads: &mut [f32], max_norm: f32) -> f32 {
    if max_norm <= 0.0 {
        return 0.0;
    }
    let mut norm_sq = 0.0f32;
    for &g in grads.iter() {
        norm_sq += g * g;
    }
    let norm = norm_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / (norm + 1e-6);
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    norm
}

pub fn cross_entropy_with_grad(logits: &Tensor, targets: &[usize]) -> (f32, Tensor) {
    assert_eq!(logits.rank(), 2);
    let n = logits.shape[0];
    let vocab = logits.shape[1];
    assert_eq!(targets.len(), n);

    let mut loss = 0.0;
    let mut grad = Tensor::zeros(&[n, vocab]);

    for (row, &target) in targets.iter().enumerate() {
        let mut row_vals = vec![0.0f32; vocab];
        for (v, val) in row_vals.iter_mut().enumerate().take(vocab) {
            *val = logits.get(&[row, v]);
        }
        let probs = softmax(&row_vals);
        let p = probs[target].max(1e-12);
        loss += -p.ln();

        for (v, &prob) in probs.iter().enumerate().take(vocab) {
            let mut g = prob;
            if v == target {
                g -= 1.0;
            }
            grad.set(&[row, v], g / n as f32);
        }
    }

    (loss / n as f32, grad)
}

fn grad_linear_from_hidden(hidden: &Tensor, d_logits: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let n = hidden.shape[0];
    let d = hidden.shape[1];
    let v = d_logits.shape[1];
    let mut grad_w = vec![0.0f32; d * v];
    let mut grad_b = vec![0.0f32; v];

    for row in 0..n {
        for token in 0..v {
            let g = d_logits.get(&[row, token]);
            grad_b[token] += g;
            for dim in 0..d {
                grad_w[dim * v + token] += hidden.get(&[row, dim]) * g;
            }
        }
    }

    (grad_w, grad_b)
}

pub fn train_lm_head_only(
    model: &mut Transformer,
    dataset: &[(Vec<usize>, Vec<usize>)],
    cfg: &TrainingConfig,
    seed: u64,
) {
    let mut opt_w = AdamW::new(model.lm_head.weight.data.len(), cfg.lr, cfg.weight_decay);
    let mut opt_b = AdamW::new(model.lm_head.bias.data.len(), cfg.lr, cfg.weight_decay);
    let mut rng = SimpleRng::new(seed);

    let micro_batches = (cfg.batch_size * cfg.grad_accum_steps).max(1);
    for step in 0..cfg.steps {
        let mut acc_w = vec![0.0f32; model.lm_head.weight.data.len()];
        let mut acc_b = vec![0.0f32; model.lm_head.bias.data.len()];
        let mut loss_sum = 0.0f32;

        for _ in 0..micro_batches {
            let idx = rng.gen_usize(dataset.len());
            let (x, y) = &dataset[idx];
            let hidden = model.forward_hidden(x);
            let logits = model.lm_head.forward_2d(&hidden);
            let (loss, d_logits) = cross_entropy_with_grad(&logits, y);
            let (grad_w, grad_b) = grad_linear_from_hidden(&hidden, &d_logits);

            loss_sum += loss;
            for i in 0..acc_w.len() {
                acc_w[i] += grad_w[i];
            }
            for i in 0..acc_b.len() {
                acc_b[i] += grad_b[i];
            }
        }

        let scale = 1.0 / micro_batches as f32;
        for g in &mut acc_w {
            *g *= scale;
        }
        for g in &mut acc_b {
            *g *= scale;
        }

        let grad_norm_w = clip_gradients(&mut acc_w, cfg.max_grad_norm);
        let grad_norm_b = clip_gradients(&mut acc_b, cfg.max_grad_norm);
        let lr_now = cosine_lr(step, cfg);
        opt_w.set_lr(lr_now);
        opt_b.set_lr(lr_now);
        opt_w.update(&mut model.lm_head.weight.data, &acc_w);
        opt_b.update(&mut model.lm_head.bias.data, &acc_b);

        if step % (cfg.steps / 10).max(1) == 0 || step + 1 == cfg.steps {
            println!(
                "step {:4} | lr {:.6} | loss {:.5} | grad_norm ({:.4}, {:.4})",
                step + 1,
                lr_now,
                loss_sum * scale,
                grad_norm_w,
                grad_norm_b
            );
        }
    }
}
