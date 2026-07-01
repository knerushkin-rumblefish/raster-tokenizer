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

#[derive(Debug, PartialEq, Eq)]
enum ArtifactCommand {
    All {
        out_dir: PathBuf,
        tokenizer_path: PathBuf,
        prompt: String,
    },
    Tokenizer {
        out_dir: PathBuf,
        tokenizer_path: PathBuf,
    },
    Prompt {
        out_dir: PathBuf,
        prompt: String,
    },
}

struct ArtifactPaths {
    rastered_path: PathBuf,
    rindex_path: PathBuf,
    prompt_path: PathBuf,
    input_path: PathBuf,
    manifest_path: PathBuf,
}

impl ArtifactPaths {
    fn new(out_dir: &Path) -> Self {
        Self {
            rastered_path: out_dir.join("tokenizer.rastered"),
            rindex_path: out_dir.join("tokenizer.rindex"),
            prompt_path: out_dir.join("prompt.bin"),
            input_path: out_dir.join("input.json"),
            manifest_path: out_dir.join("input_manifest.json"),
        }
    }
}

#[derive(Debug)]
struct AllArgs {
    out_dir: PathBuf,
    tokenizer_path: PathBuf,
    prompt: String,
}

#[derive(Debug)]
struct TokenizerArgs {
    out_dir: PathBuf,
    tokenizer_path: PathBuf,
}

#[derive(Debug)]
struct PromptArgs {
    out_dir: PathBuf,
    prompt: String,
}

fn default_tokenizer_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tokenizer.json")
}

fn looks_like_tokenizer_path(path: &Path) -> bool {
    path.is_file()
        || path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "tokenizer.json")
        || path.extension().and_then(|ext| ext.to_str()) == Some("json")
}

fn usage() -> &'static str {
    "usage: generate_input_artifacts [all] [OUT_DIR] [TOKENIZER_JSON] [PROMPT]\n       generate_input_artifacts [TOKENIZER_JSON] [PROMPT]\n       generate_input_artifacts prompt [OUT_DIR] [PROMPT]\n       generate_input_artifacts tokenizer [OUT_DIR] [TOKENIZER_JSON]"
}

fn validate_out_dir(out_dir: &Path) -> Result<()> {
    if out_dir.exists() && !out_dir.is_dir() {
        bail!(
            "output path '{}' is not a directory\n{}",
            out_dir.display(),
            usage()
        );
    }

    Ok(())
}

fn parse_all_args(args: &[String]) -> Result<AllArgs> {
    const DEFAULT_PROMPT: &str = "How to create ZKP of LLM inference?";

    let parsed = match args {
        [] => AllArgs {
            out_dir: PathBuf::from("."),
            tokenizer_path: default_tokenizer_path(),
            prompt: DEFAULT_PROMPT.to_string(),
        },
        [out_dir] => AllArgs {
            out_dir: PathBuf::from(out_dir),
            tokenizer_path: default_tokenizer_path(),
            prompt: DEFAULT_PROMPT.to_string(),
        },
        [maybe_tokenizer_path, prompt]
            if looks_like_tokenizer_path(Path::new(maybe_tokenizer_path)) =>
        {
            AllArgs {
                out_dir: PathBuf::from("."),
                tokenizer_path: PathBuf::from(maybe_tokenizer_path),
                prompt: prompt.clone(),
            }
        }
        [out_dir, tokenizer_path] => AllArgs {
            out_dir: PathBuf::from(out_dir),
            tokenizer_path: PathBuf::from(tokenizer_path),
            prompt: DEFAULT_PROMPT.to_string(),
        },
        [out_dir, tokenizer_path, prompt] => AllArgs {
            out_dir: PathBuf::from(out_dir),
            tokenizer_path: PathBuf::from(tokenizer_path),
            prompt: prompt.clone(),
        },
        _ => bail!("{}", usage()),
    };

    validate_out_dir(&parsed.out_dir)?;

    Ok(parsed)
}

fn parse_tokenizer_args(args: &[String]) -> Result<TokenizerArgs> {
    let parsed = match args {
        [] => TokenizerArgs {
            out_dir: PathBuf::from("."),
            tokenizer_path: default_tokenizer_path(),
        },
        [maybe_tokenizer_path] if looks_like_tokenizer_path(Path::new(maybe_tokenizer_path)) => {
            TokenizerArgs {
                out_dir: PathBuf::from("."),
                tokenizer_path: PathBuf::from(maybe_tokenizer_path),
            }
        }
        [out_dir] => TokenizerArgs {
            out_dir: PathBuf::from(out_dir),
            tokenizer_path: default_tokenizer_path(),
        },
        [out_dir, tokenizer_path] => TokenizerArgs {
            out_dir: PathBuf::from(out_dir),
            tokenizer_path: PathBuf::from(tokenizer_path),
        },
        _ => bail!("{}", usage()),
    };

    validate_out_dir(&parsed.out_dir)?;

    Ok(parsed)
}

fn parse_prompt_args(args: &[String]) -> Result<PromptArgs> {
    const DEFAULT_PROMPT: &str = "How to create ZKP of LLM inference?";

    let parsed = match args {
        [] => PromptArgs {
            out_dir: PathBuf::from("."),
            prompt: DEFAULT_PROMPT.to_string(),
        },
        [prompt] => PromptArgs {
            out_dir: PathBuf::from("."),
            prompt: prompt.clone(),
        },
        [out_dir, prompt] => PromptArgs {
            out_dir: PathBuf::from(out_dir),
            prompt: prompt.clone(),
        },
        _ => bail!("{}", usage()),
    };

    validate_out_dir(&parsed.out_dir)?;

    Ok(parsed)
}

fn parse_args(args: &[String]) -> Result<ArtifactCommand> {
    if let Some((command, rest)) = args.split_first() {
        return match command.as_str() {
            "all" => {
                let args = parse_all_args(rest)?;
                Ok(ArtifactCommand::All {
                    out_dir: args.out_dir,
                    tokenizer_path: args.tokenizer_path,
                    prompt: args.prompt,
                })
            }
            "tokenizer" => {
                let args = parse_tokenizer_args(rest)?;
                Ok(ArtifactCommand::Tokenizer {
                    out_dir: args.out_dir,
                    tokenizer_path: args.tokenizer_path,
                })
            }
            "prompt" | "inputs" | "input" => {
                let args = parse_prompt_args(rest)?;
                Ok(ArtifactCommand::Prompt {
                    out_dir: args.out_dir,
                    prompt: args.prompt,
                })
            }
            _ => {
                let args = parse_all_args(args)?;
                Ok(ArtifactCommand::All {
                    out_dir: args.out_dir,
                    tokenizer_path: args.tokenizer_path,
                    prompt: args.prompt,
                })
            }
        };
    }

    let args = parse_all_args(args)?;
    Ok(ArtifactCommand::All {
        out_dir: args.out_dir,
        tokenizer_path: args.tokenizer_path,
        prompt: args.prompt,
    })
}

fn write_tokenizer_input_entry(input_path: &Path) -> Result<()> {
    let mut input = load_json_object(input_path)?;
    input.insert(
        "tokenizer".into(),
        json!({
            "path": "tokenizer.rastered",
            "index_path": "tokenizer.rindex",
            "load_preference": "mmap"
        }),
    );
    write_json_object(input_path, input)
}

fn write_prompt_input_entry(input_path: &Path) -> Result<()> {
    let mut input = load_json_object(input_path)?;
    input.insert(
        "prompt".into(),
        json!({
            "path": "prompt.bin",
            "load_preference": "read"
        }),
    );
    write_json_object(input_path, input)
}

fn write_tokenizer_manifest_entry(manifest_path: &Path, commitment: String) -> Result<()> {
    let mut manifest = load_json_object(manifest_path)?;
    manifest.insert(
        "tokenizer".into(),
        json!({
            "type": "sha256",
            "encoding": "raster",
            "commitment": commitment
        }),
    );
    write_json_object(manifest_path, manifest)
}

fn write_prompt_manifest_entry(manifest_path: &Path, prompt_commitment: String) -> Result<()> {
    let mut manifest = load_json_object(manifest_path)?;
    manifest.insert(
        "prompt".into(),
        json!({
            "type": "sha256",
            "encoding": "postcard",
            "commitment": prompt_commitment
        }),
    );
    write_json_object(manifest_path, manifest)
}

fn write_tokenizer_artifacts(out_dir: &Path, tokenizer_path: &Path) -> Result<()> {
    fs::create_dir_all(out_dir)?;

    let paths = ArtifactPaths::new(out_dir);
    let tokenizer = load_tokenizer_from_json_path(tokenizer_path)?;
    let commitment = write_raster_files(&tokenizer, &paths.rastered_path, &paths.rindex_path)?;

    write_tokenizer_input_entry(&paths.input_path)?;
    write_tokenizer_manifest_entry(&paths.manifest_path, commitment)?;

    println!("Loaded {}", tokenizer_path.display());
    println!("Wrote {}", paths.rastered_path.display());
    println!("Wrote {}", paths.rindex_path.display());
    println!("Updated {}", paths.input_path.display());
    println!("Updated {}", paths.manifest_path.display());

    Ok(())
}

fn write_prompt_artifacts(out_dir: &Path, prompt: &str) -> Result<()> {
    fs::create_dir_all(out_dir)?;

    let paths = ArtifactPaths::new(out_dir);
    let prompt_bytes = postcard::to_allocvec(&prompt).context("failed to serialize prompt")?;
    fs::write(&paths.prompt_path, &prompt_bytes)?;
    let prompt_commitment = sha256_hex(&prompt_bytes);

    write_prompt_input_entry(&paths.input_path)?;
    write_prompt_manifest_entry(&paths.manifest_path, prompt_commitment)?;

    println!("Wrote {}", paths.prompt_path.display());
    println!("Prompt: {:?}", prompt);
    println!("Updated {}", paths.input_path.display());
    println!("Updated {}", paths.manifest_path.display());

    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&args)? {
        ArtifactCommand::All {
            out_dir,
            tokenizer_path,
            prompt,
        } => {
            write_tokenizer_artifacts(&out_dir, &tokenizer_path)?;
            write_prompt_artifacts(&out_dir, &prompt)?;
        }
        ArtifactCommand::Tokenizer {
            out_dir,
            tokenizer_path,
        } => write_tokenizer_artifacts(&out_dir, &tokenizer_path)?,
        ArtifactCommand::Prompt { out_dir, prompt } => write_prompt_artifacts(&out_dir, &prompt)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_arg_tokenizer_prompt_form() {
        let args = vec!["tokenizer.json".to_string(), "hello raster".to_string()];

        let parsed = parse_args(&args).expect("args should parse");

        assert_eq!(
            parsed,
            ArtifactCommand::All {
                out_dir: PathBuf::from("."),
                tokenizer_path: PathBuf::from("tokenizer.json"),
                prompt: "hello raster".to_string(),
            }
        );
    }

    #[test]
    fn parse_three_arg_out_dir_tokenizer_prompt_form() {
        let args = vec![
            "out".to_string(),
            "tokenizer.json".to_string(),
            "hello raster".to_string(),
        ];

        let parsed = parse_args(&args).expect("args should parse");

        assert_eq!(
            parsed,
            ArtifactCommand::All {
                out_dir: PathBuf::from("out"),
                tokenizer_path: PathBuf::from("tokenizer.json"),
                prompt: "hello raster".to_string(),
            }
        );
    }

    #[test]
    fn parse_tokenizer_only_form() {
        let args = vec!["tokenizer".to_string(), "tokenizer.json".to_string()];

        let parsed = parse_args(&args).expect("args should parse");

        assert_eq!(
            parsed,
            ArtifactCommand::Tokenizer {
                out_dir: PathBuf::from("."),
                tokenizer_path: PathBuf::from("tokenizer.json"),
            }
        );
    }

    #[test]
    fn parse_prompt_only_form() {
        let args = vec!["prompt".to_string(), "hello raster".to_string()];

        let parsed = parse_args(&args).expect("args should parse");

        assert_eq!(
            parsed,
            ArtifactCommand::Prompt {
                out_dir: PathBuf::from("."),
                prompt: "hello raster".to_string(),
            }
        );
    }

    #[test]
    fn parse_inputs_alias_form() {
        let args = vec!["inputs".to_string(), "hello raster".to_string()];

        let parsed = parse_args(&args).expect("args should parse");

        assert_eq!(
            parsed,
            ArtifactCommand::Prompt {
                out_dir: PathBuf::from("."),
                prompt: "hello raster".to_string(),
            }
        );
    }

    #[test]
    fn prompt_artifacts_do_not_require_tokenizer_manifest() -> Result<()> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!(
            "raster-tokenizer-prompt-test-{}-{unique}",
            std::process::id()
        ));

        write_prompt_artifacts(&out_dir, "hello raster")?;

        let input = load_json_object(&out_dir.join("input.json"))?;
        let manifest = load_json_object(&out_dir.join("input_manifest.json"))?;
        assert!(input.contains_key("prompt"));
        assert!(!input.contains_key("tokenizer"));
        assert!(manifest.contains_key("prompt"));
        assert!(!manifest.contains_key("tokenizer"));

        fs::remove_dir_all(out_dir)?;
        Ok(())
    }

    #[test]
    fn rejects_existing_file_as_output_directory() {
        let args = vec!["Cargo.toml".to_string()];

        let err = parse_args(&args).expect_err("file output path should fail");

        assert!(err
            .to_string()
            .contains("output path 'Cargo.toml' is not a directory"));
    }
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
