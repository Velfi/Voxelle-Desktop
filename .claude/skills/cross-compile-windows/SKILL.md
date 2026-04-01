---
name: cross-compile-windows
description: Cross-compile and test the Tauri/Rust backend for Windows using the `cross` tool (x86_64-pc-windows-gnu). Use when building or testing for Windows from macOS.
argument-hint: [subcommand]
allowed-tools: Bash(cross *), Bash(cargo install *), Read, Grep
---

# Cross-Compile for Windows

Cross-compile the Rust backend for the **x86_64-pc-windows-gnu** target using [cross](https://github.com/cross-rs/cross).

`cross` requires Docker (or Podman) to be running. It provides pre-built toolchain containers so you don't need a local Windows toolchain.

## Subcommand

The default subcommand is `test`. Pass a different subcommand as `$ARGUMENTS` (e.g. `build`, `check`).

## Steps

1. Ensure `cross` is installed:
   ```
   cargo install cross --git https://github.com/cross-rs/cross
   ```
2. Verify Docker is running.
3. Run:
   ```
   cd src-tauri && cross ${ARGUMENTS:-test} --target x86_64-pc-windows-gnu
   ```
4. Report results. If the build fails, diagnose the error — common issues include:
   - Missing system library bindings (check `Cargo.toml` features and conditional compilation)
   - Windows-specific code paths gated behind `#[cfg(target_os = "windows")]`
   - Docker not running or not installed