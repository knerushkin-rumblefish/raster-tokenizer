# Raster Tokenizer Example

This crate rasterizes a Gemma-style `tokenizer.json` and runs the tokenizer sequence with `cargo raster`.

## Prerequisites

- Rust and Cargo
- `cargo raster` installed and available in `PATH`
- A `tokenizer.json` file placed in the repository root

## Prepare Input Files

Generate the rasterized tokenizer plus the input files expected by `cargo raster`:

```sh
cargo run --features encode --bin encode_tokenizer -- . tokenizer.json "Hello from Raster"
```

The command above writes these files into the repository root:

- `tokenizer.json`: source tokenizer model consumed by `bin/encode_tokenizer.rs`
- `tokenizer.rastered`: rasterized tokenizer data
- `tokenizer.rindex`: index file for the rasterized tokenizer
- `prompt.bin`: postcard-encoded prompt string
- `input.json`: maps Raster inputs named `tokenizer` and `prompt` to the files above
- `input_manifest.json`: public commitments for the files referenced by `input.json`

`src/main.rs` reads the logical inputs named `tokenizer` and `prompt`, so the filenames referenced from `input.json` need to stay aligned with those generated outputs.

## Run With Cargo Raster

Once the files above exist in the repo directory, run:

```sh
cargo raster run --backend native --input input.json --input-manifest input_manifest.json  --commit tokenizer_commit.bin --verbose
```

This executes the sequence in `src/main.rs` using:

- `tokenizer` from `tokenizer.rastered` with `tokenizer.rindex`
- `prompt` from `prompt.bin`

## Expected Repo-Root Files

To run this example from the repository root, these files should be present:

```text
Cargo.toml
src/main.rs
tokenizer.json
tokenizer.rastered
tokenizer.rindex
prompt.bin
input.json
input_manifest.json
```

If you change the tokenizer JSON or the prompt text, rerun the `encode_tokenizer` command to regenerate the raster and manifest files before running `cargo raster` again.
