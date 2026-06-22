use raster::prelude::*;

use tokenizer::tokenizer::__raster_sequence_auth_encode_prompt_raster_sequence;
use tokenizer::GemmaTokenizer;

#[sequence]
fn main() {
    let tokenizer = select!(GemmaTokenizer, external!(GemmaTokenizer, "tokenizer"));
    let prompt = select!(String, external!(String, "prompt"));

    let encoded = call_seq!(encode_prompt_raster_sequence, tokenizer, prompt)
        .expect("failed to encode prompt");

    raster::println!("Encoded: {:?}", encoded);
}
