use crate::model::{SimpleRng, Transformer, argmax, softmax};
use crate::tokenizer::BpeTokenizer;

#[derive(Clone, Debug)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub beam_width: usize,
    pub beam_top_candidates: usize,
    pub length_penalty: f32,
    pub eos_token_id: Option<usize>,
}

fn sample_categorical(probs: &[f32], rng: &mut SimpleRng) -> usize {
    let r = rng.next_f32();
    let mut cdf = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r <= cdf {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

fn adjust_logits(
    logits: &[f32],
    generated_ids: &[usize],
    repetition_penalty: f32,
    temperature: f32,
) -> Vec<f32> {
    let mut adjusted = logits.to_vec();
    if repetition_penalty > 1.0 {
        for &id in generated_ids {
            if id < adjusted.len() {
                if adjusted[id] > 0.0 {
                    adjusted[id] /= repetition_penalty;
                } else {
                    adjusted[id] *= repetition_penalty;
                }
            }
        }
    }
    if temperature > 0.0 {
        for v in &mut adjusted {
            *v /= temperature;
        }
    }
    adjusted
}

fn apply_top_k(adjusted: &mut [f32], top_k: usize) {
    if top_k > 0 && top_k < adjusted.len() {
        let mut order: Vec<usize> = (0..adjusted.len()).collect();
        order.sort_by(|&a, &b| adjusted[b].total_cmp(&adjusted[a]));
        let threshold_idx = order[top_k - 1];
        let threshold = adjusted[threshold_idx];
        for x in adjusted {
            if *x < threshold {
                *x = -1e9;
            }
        }
    }
}

fn apply_top_p(probs: &mut [f32], top_p: f32) -> bool {
    if !(top_p > 0.0 && top_p < 1.0) {
        return true;
    }
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    let mut cumulative = 0.0f32;
    for &idx in &order {
        cumulative += probs[idx];
        if cumulative > top_p {
            probs[idx] = 0.0;
        }
    }
    let total: f32 = probs.iter().sum();
    if total <= 0.0 {
        return false;
    }
    for p in probs {
        *p /= total;
    }
    true
}

pub fn select_next_token(
    logits: &[f32],
    generated_ids: &[usize],
    cfg: &GenerateConfig,
    rng: &mut SimpleRng,
) -> usize {
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }

    let mut adjusted = adjust_logits(
        logits,
        generated_ids,
        cfg.repetition_penalty,
        cfg.temperature,
    );
    apply_top_k(&mut adjusted, cfg.top_k);

    let mut probs = softmax(&adjusted);
    if !apply_top_p(&mut probs, cfg.top_p) {
        return argmax(logits);
    }
    sample_categorical(&probs, rng)
}

#[derive(Clone, Debug)]
struct Beam {
    ids: Vec<usize>,
    score: f32,
    ended: bool,
}

fn top_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    idx.truncate(k.min(idx.len()));
    idx
}

fn normalized_beam_score(score: f32, length: usize, prompt_len: usize, length_penalty: f32) -> f32 {
    let gen_len = length.saturating_sub(prompt_len).max(1) as f32;
    if length_penalty <= 0.0 {
        score
    } else {
        score / gen_len.powf(length_penalty)
    }
}

fn beam_search_ids(model: &Transformer, prompt_ids: &[usize], cfg: &GenerateConfig) -> Vec<usize> {
    let mut beams = vec![Beam {
        ids: prompt_ids.to_vec(),
        score: 0.0,
        ended: false,
    }];
    let beam_width = cfg.beam_width.max(1);
    let candidates_per_beam = cfg.beam_top_candidates.max(beam_width);

    for _ in 0..cfg.max_new_tokens {
        let mut all_candidates = Vec::with_capacity(beams.len() * candidates_per_beam);

        for beam in &beams {
            if beam.ended {
                all_candidates.push(beam.clone());
                continue;
            }
            let start = beam.ids.len().saturating_sub(model.config.max_seq_len);
            let context = &beam.ids[start..];
            let logits = model.forward_logits(context);
            let seq = logits.shape[0];
            let vocab = logits.shape[1];
            let row_start = (seq - 1) * vocab;
            let row_end = row_start + vocab;
            let row = &logits.data[row_start..row_end];

            let mut adjusted = adjust_logits(row, &beam.ids, cfg.repetition_penalty, 1.0);
            apply_top_k(&mut adjusted, cfg.top_k);
            let mut probs = softmax(&adjusted);
            if !apply_top_p(&mut probs, cfg.top_p) {
                continue;
            }

            for token in top_indices(&probs, candidates_per_beam) {
                let p = probs[token].max(1e-12);
                let mut ids = beam.ids.clone();
                ids.push(token);
                all_candidates.push(Beam {
                    ended: cfg.eos_token_id == Some(token),
                    ids,
                    score: beam.score + p.ln(),
                });
            }
        }

        if all_candidates.is_empty() {
            break;
        }

        all_candidates.sort_by(|a, b| {
            let a_score =
                normalized_beam_score(a.score, a.ids.len(), prompt_ids.len(), cfg.length_penalty);
            let b_score =
                normalized_beam_score(b.score, b.ids.len(), prompt_ids.len(), cfg.length_penalty);
            b_score.total_cmp(&a_score)
        });

        beams = all_candidates.into_iter().take(beam_width).collect();
        if beams.iter().all(|b| b.ended) {
            break;
        }
    }

    beams
        .into_iter()
        .max_by(|a, b| {
            let a_score =
                normalized_beam_score(a.score, a.ids.len(), prompt_ids.len(), cfg.length_penalty);
            let b_score =
                normalized_beam_score(b.score, b.ids.len(), prompt_ids.len(), cfg.length_penalty);
            a_score.total_cmp(&b_score)
        })
        .map(|b| b.ids)
        .unwrap_or_else(|| prompt_ids.to_vec())
}

pub fn generate_text(
    model: &Transformer,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    cfg: &GenerateConfig,
    seed: u64,
) -> String {
    let mut ids = tokenizer.encode(prompt);
    if ids.is_empty() {
        ids.push(0);
    }

    if cfg.beam_width > 1 {
        let out = beam_search_ids(model, &ids, cfg);
        return tokenizer.decode(&out);
    }

    let mut rng = SimpleRng::new(seed);
    for _ in 0..cfg.max_new_tokens {
        let start = ids.len().saturating_sub(model.config.max_seq_len);
        let context = &ids[start..];
        let logits = model.forward_logits(context);

        let seq = logits.shape[0];
        let vocab = logits.shape[1];
        let row_start = (seq - 1) * vocab;
        let row_end = row_start + vocab;
        let next_id = select_next_token(&logits.data[row_start..row_end], &ids, cfg, &mut rng);
        ids.push(next_id);
        if cfg.eos_token_id == Some(next_id) {
            break;
        }
    }

    tokenizer.decode(&ids)
}
