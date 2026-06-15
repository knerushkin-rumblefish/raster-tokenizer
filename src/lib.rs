#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

pub mod tokenizer;

pub use tokenizer::{
    encode_prompt_raster, EncodedPrompt, EncodedToken, GemmaAddedToken, GemmaBpeMerge,
    GemmaBpeMergeCandidate, GemmaBpeMergeLookupEntry, GemmaDecodedToken, GemmaDecoderMetadata,
    GemmaTokenIdEntry, GemmaTokenizer, GemmaTokenizerMetadata, MinimalGemmaTokenizer,
};
