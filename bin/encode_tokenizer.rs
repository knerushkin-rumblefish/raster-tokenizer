use anyhow::{anyhow, bail, Context, Result};
use raster::core::postcard;
use raster::write_raster_files;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizer::{
    GemmaAddedToken, GemmaBpeMerge, GemmaBpeMergeCandidate, GemmaBpeMergeLookupEntry,
    GemmaDecodedToken, GemmaDecoderMetadata, GemmaTokenIdEntry, GemmaTokenizer,
    GemmaTokenizerMetadata,
};

fn load_json_object(path: &Path) -> Result<Map<String, Value>> {
    if !path.exists() {
        return Ok(Map::new());
    }

    let bytes = fs::read(path)?;
    let value: Value = serde_json::from_slice(&bytes)?;
    match value {
        Value::Object(map) => Ok(map),
        _ => anyhow::bail!(
            "Expected '{}' to contain a top-level JSON object",
            path.display()
        ),
    }
}

fn write_json_object(path: &Path, map: Map<String, Value>) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(&Value::Object(map))?)?;
    Ok(())
}

fn load_tokenizer_from_json_path(path: &Path) -> Result<GemmaTokenizer> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let raw: RawTokenizer = serde_json::from_slice(&bytes)
        .context("failed to parse tokenizer.json into raw tokenizer model")?;
    build_tokenizer(raw)
}

fn build_tokenizer(raw: RawTokenizer) -> Result<GemmaTokenizer> {
    validate_raw_tokenizer(&raw)?;

    let metadata = extract_tokenizer_metadata(&raw);
    let decoder = extract_decoder_metadata(&raw)?;
    let RawTokenizer {
        added_tokens,
        model,
        ..
    } = raw;

    let special_tokens = build_special_tokens(added_tokens);
    let token_lookup = build_token_lookup(&model.vocab);
    let tokens_by_id = build_tokens_by_id(&model.vocab, &special_tokens)?;
    let merges = build_merges(&model.vocab, model.merges);
    let merge_lookup = build_merge_lookup(&merges);

    Ok(GemmaTokenizer {
        metadata,
        decoder,
        token_lookup,
        tokens_by_id,
        special_tokens,
        merges,
        merge_lookup,
    })
}

fn validate_raw_tokenizer(raw: &RawTokenizer) -> Result<()> {
    if raw.normalizer.normalizer_type != "Replace" {
        bail!(
            "unsupported normalizer type '{}'",
            raw.normalizer.normalizer_type
        );
    }
    if raw.pre_tokenizer.pre_tokenizer_type != "Split" {
        bail!(
            "unsupported pre_tokenizer type '{}'",
            raw.pre_tokenizer.pre_tokenizer_type
        );
    }
    if raw.decoder.decoder_type != "Sequence" {
        bail!("unsupported decoder type '{}'", raw.decoder.decoder_type);
    }
    if raw.model.model_type != "BPE" {
        bail!("unsupported model type '{}'", raw.model.model_type);
    }
    if raw.normalizer.pattern.string != " " {
        bail!(
            "unsupported normalizer pattern '{}'",
            raw.normalizer.pattern.string
        );
    }

    Ok(())
}

fn extract_decoder_metadata(raw: &RawTokenizer) -> Result<GemmaDecoderMetadata> {
    let replace_decoder = raw
        .decoder
        .decoders
        .iter()
        .find_map(|decoder| match decoder {
            RawDecoderStep::Replace { pattern, content } => {
                Some((pattern.string.clone(), content.clone()))
            }
            _ => None,
        })
        .ok_or_else(|| anyhow!("decoder sequence is missing Replace decoder"))?;
    let byte_fallback = raw
        .decoder
        .decoders
        .iter()
        .any(|decoder| matches!(decoder, RawDecoderStep::ByteFallback));
    let fuse_decoder = raw
        .decoder
        .decoders
        .iter()
        .any(|decoder| matches!(decoder, RawDecoderStep::Fuse));

    Ok(GemmaDecoderMetadata {
        space_replacement: replace_decoder.0,
        byte_fallback,
        fuse_decoder,
    })
}

fn extract_tokenizer_metadata(raw: &RawTokenizer) -> GemmaTokenizerMetadata {
    GemmaTokenizerMetadata {
        space_replacement: raw.normalizer.content.clone(),
        split_delimiter: raw.pre_tokenizer.pattern.string.clone(),
        split_behavior: raw.pre_tokenizer.behavior.clone(),
        invert: raw.pre_tokenizer.invert,
        unk_token: raw.model.unk_token.clone(),
        fuse_unk: raw.model.fuse_unk,
        byte_fallback: raw.model.byte_fallback,
        ignore_merges: raw.model.ignore_merges,
    }
}

fn build_special_tokens(added_tokens: Vec<RawAddedToken>) -> Vec<GemmaAddedToken> {
    let mut special_tokens: Vec<GemmaAddedToken> = added_tokens
        .into_iter()
        .filter(|token| token.special)
        .map(Into::into)
        .collect();
    special_tokens.sort_by(|left, right| {
        right
            .content
            .len()
            .cmp(&left.content.len())
            .then_with(|| left.content.cmp(&right.content))
    });
    special_tokens
}

fn build_token_lookup(vocab: &BTreeMap<String, u32>) -> Vec<GemmaTokenIdEntry> {
    let mut token_lookup: Vec<GemmaTokenIdEntry> = vocab
        .iter()
        .map(|(token, id)| GemmaTokenIdEntry {
            token: token.clone(),
            id: *id,
        })
        .collect();
    token_lookup.sort_by(|left, right| left.token.cmp(&right.token));
    token_lookup
}

fn build_tokens_by_id(
    vocab: &BTreeMap<String, u32>,
    special_tokens: &[GemmaAddedToken],
) -> Result<Vec<GemmaDecodedToken>> {
    let max_id = vocab
        .values()
        .copied()
        .max()
        .ok_or_else(|| anyhow!("tokenizer vocab is empty"))? as usize;
    let mut tokens_by_id = vec![None; max_id + 1];

    for (token, id) in vocab {
        let special = special_tokens.iter().any(|added| added.id == *id);
        tokens_by_id[*id as usize] = Some(GemmaDecodedToken {
            id: *id,
            token: token.clone(),
            special,
        });
    }

    tokens_by_id
        .into_iter()
        .enumerate()
        .map(|(idx, token)| {
            token.ok_or_else(|| anyhow!("tokenizer vocab is missing token id {}", idx))
        })
        .collect()
}

fn build_merges(
    vocab: &BTreeMap<String, u32>,
    raw_merges: Vec<(String, String)>,
) -> Vec<GemmaBpeMerge> {
    raw_merges
        .into_iter()
        .enumerate()
        .map(|(merge_index, merge)| {
            let merged_token = format!("{}{}", merge.0, merge.1);
            let token_id = vocab.get(&merged_token).copied();
            GemmaBpeMerge {
                merge_index: merge_index as u32,
                left: merge.0,
                right: merge.1,
                merged_token,
                has_token_id: token_id.is_some(),
                token_id: token_id.unwrap_or_default(),
            }
        })
        .collect()
}

fn build_merge_lookup(merges: &[GemmaBpeMerge]) -> Vec<GemmaBpeMergeLookupEntry> {
    let mut merge_lookup: Vec<GemmaBpeMergeLookupEntry> = merges
        .iter()
        .map(|merge| GemmaBpeMergeLookupEntry {
            left: merge.left.clone(),
            right: merge.right.clone(),
            candidate: GemmaBpeMergeCandidate {
                merge_index: merge.merge_index,
                merged_token: merge.merged_token.clone(),
                has_token_id: merge.has_token_id,
                token_id: merge.token_id,
            },
        })
        .collect();
    merge_lookup.sort_by(|left, right| {
        left.left
            .cmp(&right.left)
            .then_with(|| left.right.cmp(&right.right))
    });
    merge_lookup
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

fn main() -> Result<()> {
    const DEFAULT_PROMPT: &str = "Over the last couple of months, a new programming paradigm has been rapidly gaining traction within Ethereum's frontier research and development circles, and many other corners of computing: writing code directly either in very low-level languages (eg. EVM bytecode, assembly language) or in Lean, and verifying its correctness with automatically-checkable mathematical proofs written in Lean.";

    let args: Vec<String> = std::env::args().collect();
    let out_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| ".".to_string()));
    let tokenizer_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json"));
    let prompt = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    fs::create_dir_all(&out_dir)?;

    let tokenizer = load_tokenizer_from_json_path(&tokenizer_path)?;
    let rastered_path = out_dir.join("tokenizer.rastered");
    let rindex_path = out_dir.join("tokenizer.rindex");
    let prompt_path = out_dir.join("prompt.bin");
    let input_path = out_dir.join("input.json");
    let manifest_path = out_dir.join("input_manifest.json");

    let commitment = write_raster_files(&tokenizer, &rastered_path, &rindex_path)?;
    let prompt_bytes = postcard::to_allocvec(&prompt).context("failed to serialize prompt")?;
    fs::write(&prompt_path, &prompt_bytes)?;
    let prompt_commitment = sha256_hex(&prompt_bytes);

    let mut input = load_json_object(&input_path)?;
    input.insert(
        "prompt".into(),
        json!({
            "path": "prompt.bin",
            "load_preference": "read"
        }),
    );
    input.insert(
        "tokenizer".into(),
        json!({
            "path": "tokenizer.rastered",
            "index_path": "tokenizer.rindex",
            "load_preference": "mmap"
        }),
    );
    write_json_object(&input_path, input)?;

    let mut manifest = load_json_object(&manifest_path)?;
    manifest.insert(
        "prompt".into(),
        json!({
            "type": "sha256",
            "encoding": "postcard",
            "commitment": prompt_commitment
        }),
    );
    manifest.insert(
        "tokenizer".into(),
        json!({
            "type": "sha256",
            "encoding": "raster",
            "commitment": commitment
        }),
    );
    write_json_object(&manifest_path, manifest)?;

    println!("Loaded {}", tokenizer_path.display());
    println!("Wrote {}", rastered_path.display());
    println!("Wrote {}", rindex_path.display());
    println!("Wrote {}", prompt_path.display());
    println!("Prompt: {:?}", prompt);
    println!("Updated {}", input_path.display());
    println!("Updated {}", manifest_path.display());

    Ok(())
}

#[derive(Debug, Deserialize)]
struct RawTokenizer {
    added_tokens: Vec<RawAddedToken>,
    normalizer: RawNormalizer,
    pre_tokenizer: RawPreTokenizer,
    decoder: RawDecoder,
    model: RawBpeModel,
}

#[derive(Debug, Deserialize)]
struct RawAddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

impl From<RawAddedToken> for GemmaAddedToken {
    fn from(value: RawAddedToken) -> Self {
        Self {
            id: value.id,
            content: value.content,
            single_word: value.single_word,
            lstrip: value.lstrip,
            rstrip: value.rstrip,
            normalized: value.normalized,
            special: value.special,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawNormalizer {
    #[serde(rename = "type")]
    normalizer_type: String,
    pattern: RawStringPattern,
    content: String,
}

#[derive(Debug, Deserialize)]
struct RawPreTokenizer {
    #[serde(rename = "type")]
    pre_tokenizer_type: String,
    pattern: RawStringPattern,
    behavior: String,
    invert: bool,
}

#[derive(Debug, Deserialize)]
struct RawDecoder {
    #[serde(rename = "type")]
    decoder_type: String,
    decoders: Vec<RawDecoderStep>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum RawDecoderStep {
    Replace {
        pattern: RawStringPattern,
        content: String,
    },
    ByteFallback,
    Fuse,
}

#[derive(Debug, Deserialize)]
struct RawStringPattern {
    #[serde(rename = "String")]
    string: String,
}

#[derive(Debug, Deserialize)]
struct RawBpeModel {
    #[serde(rename = "type")]
    model_type: String,
    unk_token: String,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: BTreeMap<String, u32>,
    merges: Vec<(String, String)>,
}
