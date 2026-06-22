# Raster Tokenizer Example

This crate rasterizes a Gemma-style `tokenizer.json` and runs the tokenizer sequence with `cargo raster`.

## Prerequisites

- Rust and Cargo
- `cargo raster` installed and available in `PATH`
- A `tokenizer.json` file placed in the repository root

This example depends on a local path checkout of Raster:

```toml
raster = { path = "../../raster/crates/raster", default-features = false }
```

If your local Raster checkout lives somewhere else, update the `raster` path in `Cargo.toml` before running the commands below.

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
cargo raster run --input input.json --input-manifest input_manifest.json --commit tokenizer_commit.bin
```

This executes the sequence in `src/main.rs` using:

- `tokenizer` from `tokenizer.rastered` with `tokenizer.rindex`
- `prompt` from `prompt.bin`

`cargo raster run` defaults to the `native` backend, and the current run command only supports that backend, so `--backend native` can be omitted here.

## Profiling

This crate exposes a `profiling` feature:

```toml
profiling = ["raster/profiling"]
```

To collect a Raster profile, run the same command as above but enable that feature:

```sh
cargo raster run --features profiling --input input.json --input-manifest input_manifest.json --commit tokenizer_commit.bin
```

When profiling is enabled, `cargo raster run` prints a run-artifacts directory and writes profiling files there:

- `profile.json`: finalized profiling summary for the run
- `profile.ndjson`: live profiling stream emitted while the program runs

The paths live under the Raster run output directory for that run, for example:

```text
.../runs/<run-id>/profile.json
.../runs/<run-id>/profile.ndjson
```

After the run completes, analyze the finalized profile with:

```sh
cargo raster analyze <path-to-profile.json>
```

If you want to watch the profile stream live while the run is in progress, use the follow command that `cargo raster run` prints, or run it yourself with:

```sh
cargo raster analyze --follow <path-to-profile.ndjson>
```

If you do not pass `--features profiling`, Raster still runs normally, but the profiling artifacts will not be produced.

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
