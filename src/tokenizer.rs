extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::cmp::Ordering;

use raster::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaTokenizerMetadata {
    pub space_replacement: String,
    pub split_delimiter: String,
    pub split_behavior: String,
    pub invert: bool,
    pub unk_token: String,
    pub fuse_unk: bool,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaDecoderMetadata {
    pub space_replacement: String,
    pub byte_fallback: bool,
    pub fuse_decoder: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaAddedToken {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaDecodedToken {
    pub id: u32,
    pub token: String,
    pub special: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaTokenIdEntry {
    pub token: String,
    pub id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaBpeMerge {
    pub merge_index: u32,
    pub left: String,
    pub right: String,
    pub merged_token: String,
    pub has_token_id: bool,
    pub token_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaBpeMergeCandidate {
    pub merge_index: u32,
    pub merged_token: String,
    pub has_token_id: bool,
    pub token_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaBpeMergeLookupEntry {
    pub left: String,
    pub right: String,
    pub candidate: GemmaBpeMergeCandidate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EncodedToken {
    pub piece: String,
    pub id: u32,
    pub special: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct EncodedPrompt {
    pub tokens: Vec<EncodedToken>,
}

impl EncodedPrompt {
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|token| token.id).collect()
    }

    pub fn token_pieces(&self) -> Vec<&str> {
        self.tokens
            .iter()
            .map(|token| token.piece.as_str())
            .collect()
    }
}

pub trait MinimalGemmaTokenizer {
    fn metadata(&self) -> Result<GemmaTokenizerMetadata>;
    fn decoder_metadata(&self) -> Result<GemmaDecoderMetadata>;
    fn token_id(&self, token: &str) -> Result<Option<u32>>;
    fn token_by_id(&self, token_id: u32) -> Result<Option<GemmaDecodedToken>>;
    fn longest_special_token_at(
        &self,
        input: &str,
        byte_idx: usize,
    ) -> Result<Option<GemmaAddedToken>>;
    fn bpe_merge(&self, left: &str, right: &str) -> Result<Option<GemmaBpeMergeCandidate>>;
    fn bpe_merged_token(&self, merge_index: usize) -> Result<Option<String>>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct SegmentTokenizerView {
    special_tokens: Vec<GemmaAddedToken>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct SpecialSegment {
    id: u32,
    piece: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct VocabLookupView {
    token_lookup: Vec<GemmaTokenIdEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct LookupState {
    lo: usize,
    hi: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct LookupResult {
    found: bool,
    id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct MergeLookupView {
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct EncodingTokenizerView {
    metadata: GemmaTokenizerMetadata,
    vocab: VocabLookupView,
    merges: MergeLookupView,
    unk_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct TokenizerViews {
    segmenter: SegmentTokenizerView,
    encoder: EncodingTokenizerView,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum InputSegment {
    Special(SpecialSegment),
    Text(String),
}

impl raster::core::input::Selectable for InputSegment {
    fn schema() -> raster::core::input::SchemaNode {
        raster::core::input::SchemaNode::Leaf {
            type_name: "InputSegment".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct PromptAtom {
    byte_start: usize,
    byte_end: usize,
    text: String,
    window: String,
    is_end: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct PromptChar {
    byte_start: usize,
    byte_end: usize,
    text: String,
    is_end: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct SegmentProbe {
    atom: PromptAtom,
    token_content: String,
    special: SpecialSegment,
    has_special_token: bool,
    is_first_for_atom: bool,
    is_last_for_atom: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct SegmentProbeState {
    cursor: usize,
    pending_text: String,
    matched: bool,
    matched_special: SpecialSegment,
    matched_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct MaxSpecialTokenLenState {
    max_len: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
struct AtomizePromptState {
    pending: Vec<PendingPromptAtom>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PendingPromptAtom {
    byte_start: usize,
    byte_end: usize,
    text: String,
    window: String,
}

//TODO: currently recuce output is required to have those ugly wrappers just to generate schema to
//be able to use atoms.push like operations. This should be part of internal raster Vec type (same
//for String)
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct PromptAtoms {
    atoms: Vec<PromptAtom>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct SegmentedInput {
    segments: Vec<RasterInputSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct RasterInputSegment {
    is_special: bool,
    special: SpecialSegment,
    text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct InputSegments {
    segments: Vec<InputSegment>,
}

#[tile(kind = recur)]
fn raster_input_segment_recur(
    input: RecurInput<RasterInputSegment>,
    output: RecurOutput<InputSegments>,
) -> RecurOutput<InputSegments> {
    let mut output = output;
    let segment = input.into_value();

    if segment.is_special {
        output
            .segments()
            .push(InputSegment::Special(segment.special));
    } else {
        output.segments().push(InputSegment::Text(segment.text));
    }

    output
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct NormalizedChunk {
    value: String,
}

impl NormalizedChunk {
    fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    fn as_str(&self) -> &str {
        self.value.as_str()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct PieceSequence {
    pieces: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum ResolvedTokenFragment {
    Token(EncodedToken),
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
struct BpeMergeSelection {
    piece_idx: usize,
    candidate: GemmaBpeMergeCandidate,
}

#[tile]
fn validate_tokenizer_metadata_impl(metadata: GemmaTokenizerMetadata) -> Result<()> {
    if metadata.invert {
        return Err("tokenizer pre-tokenizer invert mode is not supported".into());
    }
    if metadata.split_delimiter.is_empty() {
        return Err("tokenizer split delimiter must not be empty".into());
    }
    if metadata.split_behavior != "MergedWithPrevious" {
        return Err(alloc::format!(
            "unsupported split behavior '{}'",
            metadata.split_behavior
        ));
    }

    Ok(())
}

#[tile(kind = recur)]
fn lookup_recur(
    input: RecurInput<GemmaTokenIdEntry>,
    state: RecurState<LookupState>,
    output: RecurOutput<LookupResult>,
    entries: Vec<GemmaTokenIdEntry>,
    token: String,
) -> RecurControl<(RecurState<LookupState>, RecurOutput<LookupResult>)> {
    let _ = input.value();
    let mut state = state;
    let mut output = output;

    if input.is_first() {
        state.hi = entries.len();
    }

    if state.lo >= state.hi {
        output.found().set(false);
        output.id().set(0);
        return RecurControl::Break((state, output));
    }

    let mid = state.lo + (state.hi - state.lo) / 2;
    match entries[mid].token.as_str().cmp(&token) {
        Ordering::Equal => {
            output.found().set(true);
            output.id().set(entries[mid].id);
            RecurControl::Break((state, output))
        }
        Ordering::Less => {
            state.lo = mid + 1;
            RecurControl::Continue((state, output))
        }
        Ordering::Greater => {
            state.hi = mid;
            RecurControl::Continue((state, output))
        }
    }
}

#[tile]
fn lookup_result_to_option(result: LookupResult) -> Option<u32> {
    if result.found {
        Some(result.id)
    } else {
        None
    }
}

fn lookup_token_id_native(entries: &[GemmaTokenIdEntry], token: &str) -> Option<u32> {
    let mut lo = 0usize;
    let mut hi = entries.len();

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match entries[mid].token.as_str().cmp(token) {
            Ordering::Equal => return Some(entries[mid].id),
            Ordering::Less => lo = mid + 1,
            Ordering::Greater => hi = mid,
        }
    }

    None
}

#[sequence]
fn lookup_token_id_impl(vocab: VocabLookupView, token: String) -> Option<u32> {
    let token_lookup = select!(Vec<GemmaTokenIdEntry>, vocab.token_lookup);
    let result = call_recur!(
        tile = lookup_recur,
        input = token_lookup.clone(),
        state = LookupState { lo: 0, hi: 0 },
        output = new!(LookupResult),
        args = (token_lookup, token,)
    );

    call!(lookup_result_to_option, result)
}

#[tile]
fn token_lookup_len(token_lookup: Vec<GemmaTokenIdEntry>) -> usize {
    token_lookup.len()
}

#[tile]
fn create_vocab_lookup_view(token_lookup: Vec<GemmaTokenIdEntry>) -> VocabLookupView {
    VocabLookupView { token_lookup }
}

#[sequence]
fn require_token_id_impl(token_lookup: Vec<GemmaTokenIdEntry>, token: String) -> Result<u32> {
    let vocab = call!(create_vocab_lookup_view, token_lookup);
    let token_id = call_seq!(lookup_token_id_impl, vocab, token.clone());
    let required = call!(require_token_id_tile, token_id, token)?;
    Ok(required)
}

fn segment_tokenizer_view_from_parts(special_tokens: Vec<GemmaAddedToken>) -> SegmentTokenizerView {
    SegmentTokenizerView { special_tokens }
}

fn special_segment_from_token(token: &GemmaAddedToken) -> SpecialSegment {
    SpecialSegment {
        id: token.id,
        piece: token.content.clone(),
    }
}

#[tile]
fn segment_probes_tile(
    atoms: Vec<PromptAtom>,
    special_tokens: Vec<GemmaAddedToken>,
) -> Vec<SegmentProbe> {
    let mut probes = Vec::new();

    if special_tokens.is_empty() {
        for atom in atoms {
            probes.push(SegmentProbe {
                atom,
                token_content: String::new(),
                special: SpecialSegment::default(),
                has_special_token: false,
                is_first_for_atom: true,
                is_last_for_atom: true,
            });
        }
        return probes;
    }

    let last_token_idx = special_tokens.len() - 1;
    for atom in atoms {
        for (token_idx, token) in special_tokens.iter().enumerate() {
            probes.push(SegmentProbe {
                atom: atom.clone(),
                token_content: token.content.clone(),
                special: special_segment_from_token(token),
                has_special_token: true,
                is_first_for_atom: token_idx == 0,
                is_last_for_atom: token_idx == last_token_idx,
            });
        }
    }

    probes
}

#[tile(kind = recur)]
fn max_special_token_len_recur(
    input: RecurInput<GemmaAddedToken>,
    state: RecurState<MaxSpecialTokenLenState>,
) -> RecurState<MaxSpecialTokenLenState> {
    let mut state = state;

    state.max_len = state.max_len.max(input.value().content.len());
    state
}

#[tile]
fn create_encoding_tokenizer_view(
    metadata: GemmaTokenizerMetadata,
    token_lookup: Vec<GemmaTokenIdEntry>,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
    unk_id: u32,
) -> EncodingTokenizerView {
    EncodingTokenizerView {
        metadata,
        vocab: VocabLookupView { token_lookup },
        merges: MergeLookupView { merge_lookup },
        unk_id,
    }
}

#[sequence]
fn encoding_tokenizer_view_from_parts(
    metadata: GemmaTokenizerMetadata,
    token_lookup: Vec<GemmaTokenIdEntry>,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
) -> Result<EncodingTokenizerView> {
    call!(validate_tokenizer_metadata_impl, metadata.clone())?;

    let token = select!(String, metadata.clone().unk_token);
    let unk_id = call_seq!(require_token_id_impl, token_lookup.clone(), token)?;

    Ok(call!(
        create_encoding_tokenizer_view,
        metadata,
        token_lookup,
        merge_lookup,
        unk_id
    ))
}

fn encoding_tokenizer_view_from_parts_native(
    metadata: GemmaTokenizerMetadata,
    token_lookup: Vec<GemmaTokenIdEntry>,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
) -> Result<EncodingTokenizerView> {
    validate_tokenizer_metadata_impl(metadata.clone())?;

    let unk_id = lookup_token_id_native(token_lookup.as_slice(), metadata.unk_token.as_str())
        .ok_or_else(|| {
            alloc::format!(
                "tokenizer is missing required token '{}'",
                metadata.unk_token
            )
        })?;

    Ok(create_encoding_tokenizer_view(
        metadata,
        token_lookup,
        merge_lookup,
        unk_id,
    ))
}

fn select_tokenizer_views_impl(tokenizer: GemmaTokenizer) -> Result<TokenizerViews> {
    let GemmaTokenizer {
        metadata,
        token_lookup,
        special_tokens,
        merge_lookup,
        ..
    } = tokenizer;

    Ok(TokenizerViews {
        segmenter: segment_tokenizer_view_from_parts(special_tokens),
        encoder: encoding_tokenizer_view_from_parts_native(metadata, token_lookup, merge_lookup)?,
    })
}

fn longest_special_token_at_impl<'a>(
    tokenizer: &'a SegmentTokenizerView,
    input: &str,
    byte_idx: usize,
) -> Option<&'a GemmaAddedToken> {
    let suffix = input.get(byte_idx..)?;
    tokenizer
        .special_tokens
        .iter()
        .find(|token| suffix.starts_with(token.content.as_str()))
}

fn next_special_token_start_impl(
    tokenizer: &SegmentTokenizerView,
    input: &str,
    byte_idx: usize,
) -> usize {
    let mut cursor = byte_idx;
    while cursor < input.len() {
        if tokenizer
            .special_tokens
            .iter()
            .any(|token| input[cursor..].starts_with(token.content.as_str()))
        {
            return cursor;
        }

        cursor = input
            .get(cursor..)
            .and_then(|suffix| suffix.chars().next())
            .map(|ch| cursor + ch.len_utf8())
            .expect("cursor should always remain on a UTF-8 boundary");
    }

    input.len()
}

fn slice_input_segment(input: &str, start: usize, end: usize) -> String {
    String::from(
        input
            .get(start..end)
            .expect("sequence should only slice valid UTF-8 boundaries"),
    )
}

fn segment_input_impl(tokenizer: &SegmentTokenizerView, input: &str) -> Vec<InputSegment> {
    let mut segments = Vec::new();
    let mut cursor = 0usize;

    while cursor < input.len() {
        if let Some(special) = longest_special_token_at_impl(tokenizer, input, cursor) {
            cursor += special.content.len();
            segments.push(InputSegment::Special(special_segment_from_token(special)));
            continue;
        }

        let next_special = next_special_token_start_impl(tokenizer, input, cursor);
        let chunk = slice_input_segment(input, cursor, next_special);
        if !chunk.is_empty() {
            segments.push(InputSegment::Text(chunk));
        }
        cursor = next_special;
    }

    segments
}

// TODO: current raster do not provide some recur over String because only collection he know is
// Vec, in order to use String as recur argument raster need to provide API for iter over String
#[tile]
fn prompt_chars_tile(input: String) -> Vec<PromptChar> {
    let mut chars = Vec::new();
    for (byte_start, ch) in input.char_indices() {
        chars.push(PromptChar {
            byte_start,
            byte_end: byte_start + ch.len_utf8(),
            text: alloc::format!("{ch}"),
            is_end: false,
        });
    }

    chars.push(PromptChar {
        byte_start: input.len(),
        byte_end: input.len(),
        text: String::new(),
        is_end: true,
    });

    chars
}

fn prompt_atom_from_pending(pending: PendingPromptAtom) -> PromptAtom {
    PromptAtom {
        byte_start: pending.byte_start,
        byte_end: pending.byte_end,
        text: pending.text,
        window: pending.window,
        is_end: false,
    }
}

#[tile(kind = recur)]
fn atomize_prompt_recur(
    input: RecurInput<PromptChar>,
    state: RecurState<AtomizePromptState>,
    output: RecurOutput<PromptAtoms>,
    max_special_len: usize,
) -> RecurControl<(RecurState<AtomizePromptState>, RecurOutput<PromptAtoms>)> {
    let current = input.into_value();
    let mut state = state;
    let mut output = output;

    if current.is_end {
        for pending in state.pending.drain(..) {
            output.atoms().push(prompt_atom_from_pending(pending));
        }
        output.atoms().push(PromptAtom {
            byte_start: current.byte_start,
            byte_end: current.byte_end,
            text: String::new(),
            window: String::new(),
            is_end: true,
        });
        return RecurControl::Break((state, output));
    }

    if max_special_len == 0 {
        output.atoms().push(PromptAtom {
            byte_start: current.byte_start,
            byte_end: current.byte_end,
            text: current.text,
            window: String::new(),
            is_end: false,
        });
        return RecurControl::Continue((state, output));
    }

    let mut next_pending = Vec::new();
    for mut pending in core::mem::take(&mut state.pending) {
        if pending.window.len() < max_special_len {
            pending.window.push_str(current.text.as_str());
        }

        if pending.window.len() >= max_special_len {
            output.atoms().push(prompt_atom_from_pending(pending));
        } else {
            next_pending.push(pending);
        }
    }
    state.pending = next_pending;

    let current_window = current.text.clone();
    if current_window.len() >= max_special_len {
        output.atoms().push(PromptAtom {
            byte_start: current.byte_start,
            byte_end: current.byte_end,
            text: current.text,
            window: current_window,
            is_end: false,
        });
    } else {
        state.pending.push(PendingPromptAtom {
            byte_start: current.byte_start,
            byte_end: current.byte_end,
            text: current.text,
            window: current_window,
        });
    }

    RecurControl::Continue((state, output))
}

#[tile(kind = recur)]
fn segment_probe_recur(
    input: RecurInput<SegmentProbe>,
    state: RecurState<SegmentProbeState>,
    output: RecurOutput<SegmentedInput>,
) -> RecurControl<(RecurState<SegmentProbeState>, RecurOutput<SegmentedInput>)> {
    let probe = input.into_value();
    let atom = probe.atom;
    let mut state = state;
    let mut output = output;

    if atom.byte_start != state.cursor {
        return RecurControl::Continue((state, output));
    }

    if atom.is_end {
        if !state.pending_text.is_empty() {
            output.segments().push(RasterInputSegment {
                is_special: false,
                special: SpecialSegment::default(),
                text: core::mem::take(&mut state.pending_text),
            });
        }
        return RecurControl::Break((state, output));
    }

    if probe.is_first_for_atom {
        state.matched = false;
        state.matched_special = SpecialSegment::default();
        state.matched_len = 0;
    }

    if probe.has_special_token
        && !state.matched
        && atom.window.starts_with(probe.token_content.as_str())
    {
        state.matched = true;
        state.matched_special = probe.special;
        state.matched_len = probe.token_content.len();
    }

    if !probe.is_last_for_atom {
        return RecurControl::Continue((state, output));
    }

    if state.matched {
        if !state.pending_text.is_empty() {
            output.segments().push(RasterInputSegment {
                is_special: false,
                special: SpecialSegment::default(),
                text: core::mem::take(&mut state.pending_text),
            });
        }

        state.cursor = atom.byte_start + state.matched_len;
        output.segments().push(RasterInputSegment {
            is_special: true,
            special: core::mem::take(&mut state.matched_special),
            text: String::new(),
        });
    } else {
        state.pending_text.push_str(atom.text.as_str());
        state.cursor = atom.byte_end;
    }

    state.matched = false;
    state.matched_len = 0;

    RecurControl::Continue((state, output))
}

fn normalize_text_segment_impl(
    chunk: String,
    metadata: &GemmaTokenizerMetadata,
) -> NormalizedChunk {
    let mut normalized = chunk.replace(
        metadata.split_delimiter.as_str(),
        metadata.space_replacement.as_str(),
    );
    if !normalized.is_empty() && !normalized.starts_with(metadata.space_replacement.as_str()) {
        normalized = alloc::format!("{}{}", metadata.space_replacement, normalized);
    }

    NormalizedChunk { value: normalized }
}

fn split_normalized_chunk_impl(normalized: NormalizedChunk) -> PieceSequence {
    PieceSequence {
        pieces: normalized
            .as_str()
            .chars()
            .map(|ch| alloc::format!("{ch}"))
            .collect(),
    }
}

fn scan_best_bpe_merge_impl(
    tokenizer: &MergeLookupView,
    pieces: &PieceSequence,
) -> Option<BpeMergeSelection> {
    let mut best_merge: Option<BpeMergeSelection> = None;

    for idx in 0..pieces.pieces.len().saturating_sub(1) {
        let left = &pieces.pieces[idx];
        let right = &pieces.pieces[idx + 1];
        let candidate = tokenizer
            .merge_lookup
            .binary_search_by(|entry| {
                entry
                    .left
                    .as_str()
                    .cmp(left.as_str())
                    .then_with(|| entry.right.as_str().cmp(right.as_str()))
            })
            .ok()
            .map(|candidate_idx| tokenizer.merge_lookup[candidate_idx].candidate.clone());

        let Some(candidate) = candidate else {
            continue;
        };

        let is_better = best_merge
            .as_ref()
            .map(|best| candidate.merge_index < best.candidate.merge_index)
            .unwrap_or(true);
        if is_better {
            best_merge = Some(BpeMergeSelection {
                piece_idx: idx,
                candidate,
            });
        }
    }

    best_merge
}

fn apply_bpe_merge_selection_impl(
    mut pieces: PieceSequence,
    selection: BpeMergeSelection,
) -> PieceSequence {
    pieces.pieces[selection.piece_idx] = selection.candidate.merged_token;
    pieces.pieces.remove(selection.piece_idx + 1);
    pieces
}

fn apply_bpe_merges_impl(tokenizer: &MergeLookupView, mut pieces: PieceSequence) -> PieceSequence {
    loop {
        let Some(selection) = scan_best_bpe_merge_impl(tokenizer, &pieces) else {
            return pieces;
        };
        pieces = apply_bpe_merge_selection_impl(pieces, selection);
    }
}

fn resolve_piece_impl(
    tokenizer: &VocabLookupView,
    metadata: &GemmaTokenizerMetadata,
    piece: String,
) -> Vec<ResolvedTokenFragment> {
    if let Some(id) = lookup_token_id_native(tokenizer.token_lookup.as_slice(), piece.as_str()) {
        return alloc::vec![ResolvedTokenFragment::Token(EncodedToken {
            piece,
            id,
            special: false,
        })];
    }

    if metadata.byte_fallback {
        let mut fragments = Vec::new();
        for byte in piece.as_bytes() {
            let byte_piece = alloc::format!("<0x{:02X}>", byte);
            if let Some(id) =
                lookup_token_id_native(tokenizer.token_lookup.as_slice(), byte_piece.as_str())
            {
                fragments.push(ResolvedTokenFragment::Token(EncodedToken {
                    piece: byte_piece,
                    id,
                    special: false,
                }));
            } else {
                fragments.push(ResolvedTokenFragment::Unknown);
            }
        }
        return fragments;
    }

    alloc::vec![ResolvedTokenFragment::Unknown]
}

fn resolve_piece_sequence_impl(
    tokenizer: &VocabLookupView,
    metadata: &GemmaTokenizerMetadata,
    pieces: PieceSequence,
) -> Vec<ResolvedTokenFragment> {
    let mut fragments = Vec::new();
    for piece in pieces.pieces {
        fragments.extend(resolve_piece_impl(tokenizer, metadata, piece));
    }
    fragments
}

fn finalize_resolved_fragments_impl(
    unk_token: &str,
    fuse_unk: bool,
    unk_id: u32,
    fragments: Vec<ResolvedTokenFragment>,
) -> Vec<EncodedToken> {
    let mut output = Vec::new();

    for fragment in fragments {
        match fragment {
            ResolvedTokenFragment::Token(token) => output.push(token),
            ResolvedTokenFragment::Unknown => {
                if fuse_unk
                    && output
                        .last()
                        .map(|token| !token.special && token.piece == unk_token)
                        .unwrap_or(false)
                {
                    continue;
                }

                output.push(EncodedToken {
                    piece: String::from(unk_token),
                    id: unk_id,
                    special: false,
                });
            }
        }
    }

    output
}

fn encode_special_segment_impl(special: SpecialSegment) -> EncodedToken {
    EncodedToken {
        piece: special.piece,
        id: special.id,
        special: true,
    }
}

fn encode_text_segment_impl(tokenizer: &EncodingTokenizerView, chunk: String) -> Vec<EncodedToken> {
    if chunk.is_empty() {
        return Vec::new();
    }

    let normalized = normalize_text_segment_impl(chunk, &tokenizer.metadata);
    if normalized.is_empty() {
        return Vec::new();
    }

    let pieces = split_normalized_chunk_impl(normalized);
    let merged_pieces = if tokenizer.metadata.ignore_merges {
        pieces
    } else {
        apply_bpe_merges_impl(&tokenizer.merges, pieces)
    };
    let fragments =
        resolve_piece_sequence_impl(&tokenizer.vocab, &tokenizer.metadata, merged_pieces);

    finalize_resolved_fragments_impl(
        tokenizer.metadata.unk_token.as_str(),
        tokenizer.metadata.fuse_unk,
        tokenizer.unk_id,
        fragments,
    )
}

fn encode_segments_impl(
    tokenizer: &EncodingTokenizerView,
    segments: Vec<InputSegment>,
) -> Result<EncodedPrompt> {
    let mut tokens = Vec::new();

    for segment in segments {
        match segment {
            InputSegment::Special(special) => tokens.push(encode_special_segment_impl(special)),
            InputSegment::Text(chunk) => {
                tokens.extend(encode_text_segment_impl(tokenizer, chunk));
            }
        }
    }

    Ok(EncodedPrompt { tokens })
}

#[tile]
fn require_token_id_tile(token_id: Option<u32>, token: String) -> Result<u32> {
    token_id.ok_or_else(|| alloc::format!("tokenizer is missing required token '{}'", token))
}

#[sequence]
fn build_encoding_tokenizer_view(
    metadata: GemmaTokenizerMetadata,
    token_lookup: Vec<GemmaTokenIdEntry>,
    merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
) -> Result<EncodingTokenizerView> {
    call_seq!(
        encoding_tokenizer_view_from_parts,
        metadata,
        token_lookup,
        merge_lookup
    )
}

#[tile]
fn encode_segments_tile(
    tokenizer: EncodingTokenizerView,
    segments: Vec<InputSegment>,
) -> Result<EncodedPrompt> {
    encode_segments_impl(&tokenizer, segments)
}

#[sequence]
fn select_encoding_view(tokenizer: GemmaTokenizer) -> Result<EncodingTokenizerView> {
    let metadata = select!(GemmaTokenizerMetadata, tokenizer.clone().metadata);
    let token_lookup = select!(Vec<GemmaTokenIdEntry>, tokenizer.clone().token_lookup);
    let merge_lookup = select!(Vec<GemmaBpeMergeLookupEntry>, tokenizer.merge_lookup);

    call_seq!(
        build_encoding_tokenizer_view,
        metadata,
        token_lookup,
        merge_lookup
    )
}

#[sequence]
fn segment_prompt(tokenizer: GemmaTokenizer, input: String) -> Vec<InputSegment> {
    let special_tokens = select!(Vec<GemmaAddedToken>, tokenizer.special_tokens);
    let max_special_len_state = call_recur!(
        tile = max_special_token_len_recur,
        input = special_tokens.clone(),
        state = MaxSpecialTokenLenState { max_len: 0 },
        args = ()
    );
    let max_special_len = select!(usize, max_special_len_state.max_len);
    let prompt_chars = call!(prompt_chars_tile, input);
    let atomized = call_recur!(
        tile = atomize_prompt_recur,
        input = prompt_chars,
        state = AtomizePromptState {
            pending: Vec::new(),
        },
        output = new!(PromptAtoms),
        args = (max_special_len,)
    );
    let atoms = select!(Vec<PromptAtom>, atomized.atoms);
    let probes = call!(segment_probes_tile, atoms, special_tokens);
    let segmented = call_recur!(
        tile = segment_probe_recur,
        input = probes,
        state = SegmentProbeState {
            cursor: 0,
            pending_text: String::new(),
            matched: false,
            matched_special: SpecialSegment::default(),
            matched_len: 0,
        },
        output = new!(SegmentedInput),
        args = ()
    );
    let raster_segments = select!(Vec<RasterInputSegment>, segmented.segments);
    let segments = call_recur!(
        tile = raster_input_segment_recur,
        input = raster_segments,
        output = new!(InputSegments),
        args = ()
    );
    select!(Vec<InputSegment>, segments.segments)
}

#[sequence]
pub fn encode_prompt_raster_sequence(
    tokenizer: GemmaTokenizer,
    input: String,
) -> Result<EncodedPrompt> {
    let segments = call_seq!(segment_prompt, tokenizer.clone(), input);
    let encoder = call_seq!(select_encoding_view, tokenizer)?;
    let prompt = call!(encode_segments_tile, encoder, segments)?;
    Ok(prompt)
}

pub fn encode_prompt_raster(tokenizer: GemmaTokenizer, input: String) -> Result<EncodedPrompt> {
    materialize_auth_result::<EncodedPrompt, _>(
        __raster_sequence_auth_encode_prompt_raster_sequence(tokenizer, input),
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, raster::Selectable)]
pub struct GemmaTokenizer {
    pub metadata: GemmaTokenizerMetadata,
    pub decoder: GemmaDecoderMetadata,
    pub token_lookup: Vec<GemmaTokenIdEntry>,
    pub tokens_by_id: Vec<GemmaDecodedToken>,
    pub special_tokens: Vec<GemmaAddedToken>,
    pub merges: Vec<GemmaBpeMerge>,
    pub merge_lookup: Vec<GemmaBpeMergeLookupEntry>,
}

#[tile]
pub fn tokenizer_metadata(tokenizer: GemmaTokenizer) -> GemmaTokenizerMetadata {
    tokenizer.metadata.clone()
}

impl GemmaTokenizer {
    pub fn encode_prompt(&self, input: &str) -> Result<EncodedPrompt> {
        let views = select_tokenizer_views_impl(self.clone())?;
        let segments = segment_input_impl(&views.segmenter, input);
        encode_segments_impl(&views.encoder, segments)
    }
}

impl MinimalGemmaTokenizer for GemmaTokenizer {
    fn metadata(&self) -> Result<GemmaTokenizerMetadata> {
        Ok(self.metadata.clone())
    }

    fn decoder_metadata(&self) -> Result<GemmaDecoderMetadata> {
        Ok(self.decoder.clone())
    }

    fn token_id(&self, token: &str) -> Result<Option<u32>> {
        Ok(self
            .token_lookup
            .binary_search_by(|entry| entry.token.as_str().cmp(token))
            .ok()
            .map(|idx| self.token_lookup[idx].id))
    }

    fn token_by_id(&self, token_id: u32) -> Result<Option<GemmaDecodedToken>> {
        Ok(self.tokens_by_id.get(token_id as usize).cloned())
    }

    fn longest_special_token_at(
        &self,
        input: &str,
        byte_idx: usize,
    ) -> Result<Option<GemmaAddedToken>> {
        let Some(suffix) = input.get(byte_idx..) else {
            return Err(alloc::format!(
                "byte index {} is not a valid string boundary",
                byte_idx
            ));
        };

        Ok(self
            .special_tokens
            .iter()
            .find(|token| suffix.starts_with(token.content.as_str()))
            .cloned())
    }

    fn bpe_merge(&self, left: &str, right: &str) -> Result<Option<GemmaBpeMergeCandidate>> {
        Ok(self
            .merge_lookup
            .binary_search_by(|entry| {
                entry
                    .left
                    .as_str()
                    .cmp(left)
                    .then_with(|| entry.right.as_str().cmp(right))
            })
            .ok()
            .map(|idx| self.merge_lookup[idx].candidate.clone()))
    }

    fn bpe_merged_token(&self, merge_index: usize) -> Result<Option<String>> {
        Ok(self
            .merges
            .get(merge_index)
            .map(|merge| merge.merged_token.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raster::core::input::SchemaNode;
    use raster::input::verify_selection_proof;
    use raster::{encode_raster_value, Selectable, SelectionProof, SelectorPath, SelectorSegment};

    fn sample_tokenizer() -> GemmaTokenizer {
        GemmaTokenizer {
            metadata: GemmaTokenizerMetadata {
                space_replacement: "▁".into(),
                split_delimiter: " ".into(),
                split_behavior: "MergedWithPrevious".into(),
                invert: false,
                unk_token: "<unk>".into(),
                fuse_unk: true,
                byte_fallback: true,
                ignore_merges: false,
            },
            decoder: GemmaDecoderMetadata {
                space_replacement: "▁".into(),
                byte_fallback: true,
                fuse_decoder: true,
            },
            token_lookup: vec![
                GemmaTokenIdEntry {
                    token: "<bos>".into(),
                    id: 0,
                },
                GemmaTokenIdEntry {
                    token: "hello".into(),
                    id: 1,
                },
                GemmaTokenIdEntry {
                    token: "world".into(),
                    id: 2,
                },
            ],
            tokens_by_id: vec![
                GemmaDecodedToken {
                    id: 0,
                    token: "<bos>".into(),
                    special: true,
                },
                GemmaDecodedToken {
                    id: 1,
                    token: "hello".into(),
                    special: false,
                },
                GemmaDecodedToken {
                    id: 2,
                    token: "world".into(),
                    special: false,
                },
            ],
            special_tokens: vec![GemmaAddedToken {
                id: 0,
                content: "<bos>".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            }],
            merges: vec![GemmaBpeMerge {
                merge_index: 0,
                left: "hello".into(),
                right: "world".into(),
                merged_token: "helloworld".into(),
                has_token_id: false,
                token_id: 0,
            }],
            merge_lookup: vec![GemmaBpeMergeLookupEntry {
                left: "hello".into(),
                right: "world".into(),
                candidate: GemmaBpeMergeCandidate {
                    merge_index: 0,
                    merged_token: "helloworld".into(),
                    has_token_id: false,
                    token_id: 0,
                },
            }],
        }
    }

    fn build_tokenizer(
        vocab: &[(u32, &str, bool)],
        special_tokens: Vec<GemmaAddedToken>,
        merge_specs: &[(&str, &str, &str, bool, u32)],
    ) -> GemmaTokenizer {
        let mut token_lookup: Vec<GemmaTokenIdEntry> = vocab
            .iter()
            .map(|(id, token, _)| GemmaTokenIdEntry {
                token: (*token).into(),
                id: *id,
            })
            .collect();
        token_lookup.sort_by(|left, right| left.token.cmp(&right.token));

        let max_id = vocab.iter().map(|(id, _, _)| *id).max().unwrap() as usize;
        let mut tokens_by_id: Vec<Option<GemmaDecodedToken>> = vec![None; max_id + 1];
        for (id, token, special) in vocab {
            tokens_by_id[*id as usize] = Some(GemmaDecodedToken {
                id: *id,
                token: (*token).into(),
                special: *special,
            });
        }

        let merges: Vec<GemmaBpeMerge> = merge_specs
            .iter()
            .enumerate()
            .map(
                |(merge_index, (left, right, merged_token, has_token_id, token_id))| {
                    GemmaBpeMerge {
                        merge_index: merge_index as u32,
                        left: (*left).into(),
                        right: (*right).into(),
                        merged_token: (*merged_token).into(),
                        has_token_id: *has_token_id,
                        token_id: *token_id,
                    }
                },
            )
            .collect();

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

        GemmaTokenizer {
            metadata: GemmaTokenizerMetadata {
                space_replacement: "▁".into(),
                split_delimiter: " ".into(),
                split_behavior: "MergedWithPrevious".into(),
                invert: false,
                unk_token: "<unk>".into(),
                fuse_unk: true,
                byte_fallback: true,
                ignore_merges: false,
            },
            decoder: GemmaDecoderMetadata {
                space_replacement: "▁".into(),
                byte_fallback: true,
                fuse_decoder: true,
            },
            token_lookup,
            tokens_by_id: tokens_by_id
                .into_iter()
                .map(|token| token.expect("missing token id in test fixture"))
                .collect(),
            special_tokens,
            merges,
            merge_lookup,
        }
    }

    fn sample_encoding_tokenizer() -> GemmaTokenizer {
        build_tokenizer(
            &[
                (0, "<unk>", true),
                (1, "<bos>", true),
                (2, "<0x82>", false),
                (3, "<0x99>", false),
                (4, "<0x9F>", false),
                (5, "<0xF0>", false),
                (6, "d", false),
                (7, "e", false),
                (8, "h", false),
                (9, "l", false),
                (10, "o", false),
                (11, "r", false),
                (12, "w", false),
                (13, "▁", false),
                (14, "▁hello", false),
                (15, "▁world", false),
            ],
            vec![GemmaAddedToken {
                id: 1,
                content: "<bos>".into(),
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            }],
            &[
                ("▁", "h", "▁h", false, 0),
                ("▁h", "e", "▁he", false, 0),
                ("▁he", "l", "▁hel", false, 0),
                ("▁hel", "l", "▁hell", false, 0),
                ("▁hell", "o", "▁hello", true, 14),
                ("▁", "w", "▁w", false, 0),
                ("▁w", "o", "▁wo", false, 0),
                ("▁wo", "r", "▁wor", false, 0),
                ("▁wor", "l", "▁worl", false, 0),
                ("▁worl", "d", "▁world", true, 15),
            ],
        )
    }

    #[test]
    fn provides_minimal_gemma_behavior() {
        let tokenizer = sample_tokenizer();

        assert_eq!(
            MinimalGemmaTokenizer::token_id(&tokenizer, "<bos>").unwrap(),
            Some(0)
        );
        assert_eq!(
            tokenizer
                .token_by_id(0)
                .unwrap()
                .as_ref()
                .map(|token| token.token.as_str()),
            Some("<bos>")
        );
        assert_eq!(
            MinimalGemmaTokenizer::longest_special_token_at(&tokenizer, "say <bos> now", 4)
                .unwrap()
                .as_ref()
                .map(|token| token.content.as_str()),
            Some("<bos>")
        );

        let first_merge = tokenizer.merges.first().unwrap();
        assert_eq!(
            MinimalGemmaTokenizer::bpe_merge(&tokenizer, &first_merge.left, &first_merge.right)
                .unwrap()
                .as_ref()
                .map(|candidate| candidate.merge_index),
            Some(first_merge.merge_index)
        );
        assert_eq!(
            tokenizer
                .bpe_merged_token(first_merge.merge_index as usize)
                .unwrap(),
            Some(first_merge.merged_token.clone())
        );
    }

    #[test]
    fn raster_encoding_round_trips_and_verifies() {
        let tokenizer = sample_tokenizer();
        let (data_bytes, _index_bytes, commitment) = encode_raster_value(&tokenizer).unwrap();
        let proof = SelectionProof {
            path: SelectorPath::default(),
            root_hash: hex_to_bytes(&commitment),
            steps: Vec::new(),
        };

        assert!(verify_selection_proof(&data_bytes, &proof));
    }

    #[test]
    fn selector_path_targets_special_token_content() {
        let selector = SelectorPath::new(vec![
            SelectorSegment::from("special_tokens"),
            SelectorSegment::from(0usize),
            SelectorSegment::from("content"),
        ]);

        let schema = GemmaTokenizer::schema();
        let node = schema_node_at(&schema, &selector).unwrap();
        assert!(matches!(node, SchemaNode::Leaf { .. }));
    }

    #[test]
    fn encodes_known_prompt_pieces() {
        let tokenizer = sample_encoding_tokenizer();

        let encoded = tokenizer.encode_prompt("hello world").unwrap();

        assert_eq!(encoded.token_pieces(), vec!["▁hello", "▁world"]);
        assert_eq!(encoded.token_ids(), vec![14, 15]);
    }

    #[test]
    fn method_uses_native_path() {
        let tokenizer = sample_encoding_tokenizer();

        let encoded = tokenizer.encode_prompt("hello world").unwrap();

        assert_eq!(encoded.token_pieces(), vec!["▁hello", "▁world"]);
        assert_eq!(encoded.token_ids(), vec![14, 15]);
    }

    #[test]
    fn keeps_special_tokens_intact() {
        let tokenizer = sample_encoding_tokenizer();

        let encoded = tokenizer.encode_prompt("hello<bos>world").unwrap();

        assert_eq!(encoded.token_pieces(), vec!["▁hello", "<bos>", "▁world"]);
        assert_eq!(encoded.token_ids(), vec![14, 1, 15]);
        assert!(encoded.tokens[1].special);
    }

    #[test]
    fn falls_back_to_unk_when_byte_fallback_is_disabled() {
        let mut tokenizer = sample_encoding_tokenizer();
        tokenizer.metadata.byte_fallback = false;

        let encoded = tokenizer.encode_prompt("🙂").unwrap();

        assert_eq!(encoded.token_pieces(), vec!["▁", "<unk>"]);
        assert_eq!(encoded.token_ids(), vec![13, 0]);
    }

    #[test]
    fn falls_back_to_bytes_when_piece_is_missing() {
        let tokenizer = sample_encoding_tokenizer();

        let encoded = tokenizer.encode_prompt("🙂").unwrap();

        assert_eq!(
            encoded.token_pieces(),
            vec!["▁", "<0xF0>", "<0x9F>", "<0x99>", "<0x82>"]
        );
        assert_eq!(encoded.token_ids(), vec![13, 5, 4, 3, 2]);
    }

    #[test]
    fn prefers_lowest_rank_bpe_merge_each_iteration() {
        let mut merge_lookup = vec![
            GemmaBpeMergeLookupEntry {
                left: "b".into(),
                right: "c".into(),
                candidate: GemmaBpeMergeCandidate {
                    merge_index: 0,
                    merged_token: "bc".into(),
                    has_token_id: false,
                    token_id: 0,
                },
            },
            GemmaBpeMergeLookupEntry {
                left: "a".into(),
                right: "b".into(),
                candidate: GemmaBpeMergeCandidate {
                    merge_index: 1,
                    merged_token: "ab".into(),
                    has_token_id: false,
                    token_id: 0,
                },
            },
        ];
        merge_lookup.sort_by(|left, right| {
            left.left
                .cmp(&right.left)
                .then_with(|| left.right.cmp(&right.right))
        });
        let merge_lookup = MergeLookupView { merge_lookup };
        let pieces = PieceSequence {
            pieces: vec!["a".into(), "b".into(), "c".into()],
        };

        let merged = apply_bpe_merges_impl(&merge_lookup, pieces);

        assert_eq!(merged.pieces, vec!["a", "bc"]);
    }

    fn hex_to_bytes(input: &str) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(input.len() / 2);
        let chars: Vec<char> = input.chars().collect();
        for pair in chars.chunks(2) {
            let hi = pair[0].to_digit(16).unwrap();
            let lo = pair[1].to_digit(16).unwrap();
            bytes.push(((hi << 4) | lo) as u8);
        }
        bytes
    }

    fn schema_node_at<'a>(
        schema: &'a SchemaNode,
        selector: &SelectorPath,
    ) -> Option<&'a SchemaNode> {
        let mut current = schema;
        for segment in &selector.segments {
            match (segment, current) {
                (SelectorSegment::Field(field_name), SchemaNode::Struct { fields, .. }) => {
                    current = fields
                        .iter()
                        .find(|field| field.name == *field_name)
                        .map(|field| &field.schema)?;
                }
                (SelectorSegment::Index(_), SchemaNode::List { element, .. }) => {
                    current = element;
                }
                _ => return None,
            }
        }
        Some(current)
    }
}
