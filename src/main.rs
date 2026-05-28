use raster::prelude::*;

use tokenizer::{encode_prompt_raster, GemmaTokenizer};

#[sequence]
fn main() {
    let tokenizer = select!(GemmaTokenizer, external!(GemmaTokenizer, "tokenizer"));
    let prompt = select!(String, external!(String, "prompt"));

    let encoded =
        call_seq!(encode_prompt_raster, tokenizer, prompt).expect("failed to encode prompt");

    debug!("Prompt: external input 'prompt'");
    debug!("Tokenizer: Raster external input 'tokenizer'");
    debug!("Pieces: {:?}", encoded.token_pieces());
    debug!("Token IDs: {:?}", encoded.token_ids());
    debug!("Detailed tokens:");
    for (index, token) in encoded.tokens.iter().enumerate() {
        let suffix = if token.special { " [special]" } else { "" };
        debug!("  {index}: {} ({}){suffix}", token.piece, token.id);
    }
}
