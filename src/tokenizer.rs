use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};

#[derive(Clone, Debug)]
struct MergeRule {
    left: u32,
    right: u32,
    new_id: u32,
}

#[derive(Clone, Debug)]
pub struct BpeTokenizer {
    id_to_piece: Vec<Vec<u8>>,
    merges: Vec<MergeRule>,
}

impl BpeTokenizer {
    pub fn train(text: &str, num_merges: usize) -> Self {
        let mut id_to_piece: Vec<Vec<u8>> = (0u8..=255u8).map(|b| vec![b]).collect();
        let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        let mut merges = Vec::new();

        for _ in 0..num_merges {
            if ids.len() < 2 {
                break;
            }

            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for i in 0..(ids.len() - 1) {
                *pair_counts.entry((ids[i], ids[i + 1])).or_insert(0) += 1;
            }

            let Some(((left, right), count)) = pair_counts.into_iter().max_by_key(|(_, c)| *c)
            else {
                break;
            };
            if count < 2 {
                break;
            }

            let new_id = id_to_piece.len() as u32;
            let mut new_piece = id_to_piece[left as usize].clone();
            new_piece.extend_from_slice(&id_to_piece[right as usize]);
            id_to_piece.push(new_piece);

            let mut replaced = Vec::with_capacity(ids.len());
            let mut i = 0usize;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == left && ids[i + 1] == right {
                    replaced.push(new_id);
                    i += 2;
                } else {
                    replaced.push(ids[i]);
                    i += 1;
                }
            }
            ids = replaced;
            merges.push(MergeRule {
                left,
                right,
                new_id,
            });
        }

        Self {
            id_to_piece,
            merges,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_piece.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        for m in &self.merges {
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0usize;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == m.left && ids[i + 1] == m.right {
                    out.push(m.new_id);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
        }
        ids.into_iter().map(|v| v as usize).collect()
    }

    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            bytes.extend_from_slice(&self.id_to_piece[id]);
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = File::create(path)?;
        f.write_all(b"BPE1")?;
        write_u64(&mut f, self.id_to_piece.len() as u64)?;
        for p in &self.id_to_piece {
            write_u64(&mut f, p.len() as u64)?;
            f.write_all(p)?;
        }
        write_u64(&mut f, self.merges.len() as u64)?;
        for m in &self.merges {
            write_u32(&mut f, m.left)?;
            write_u32(&mut f, m.right)?;
            write_u32(&mut f, m.new_id)?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"BPE1" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid tokenizer magic",
            ));
        }

        let vocab_size = read_u64(&mut f)? as usize;
        let mut id_to_piece = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let len = read_u64(&mut f)? as usize;
            let mut bytes = vec![0u8; len];
            f.read_exact(&mut bytes)?;
            id_to_piece.push(bytes);
        }

        let n_merges = read_u64(&mut f)? as usize;
        let mut merges = Vec::with_capacity(n_merges);
        for _ in 0..n_merges {
            merges.push(MergeRule {
                left: read_u32(&mut f)?,
                right: read_u32(&mut f)?,
                new_id: read_u32(&mut f)?,
            });
        }

        Ok(Self {
            id_to_piece,
            merges,
        })
    }
}

pub fn build_language_model_dataset(
    tokens: &[usize],
    block_size: usize,
    stride: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert!(tokens.len() > block_size + 1);
    let mut out = Vec::new();
    let last = tokens.len() - block_size - 1;
    let mut i = 0usize;
    while i <= last {
        let x = tokens[i..i + block_size].to_vec();
        let y = tokens[i + 1..i + 1 + block_size].to_vec();
        out.push((x, y));
        i += stride.max(1);
    }
    out
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
