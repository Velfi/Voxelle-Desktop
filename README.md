# Voxelle Desktop

Native voxel viewer and editor (Tauri + React + Vite).

## Format

- **[Voxelle v4 file layout (writers & readers)](docs/VOXELLE_FORMAT_V4.md)** — VX4 container, inner BSON vs **VX3 dense wire** (`wire_version` 3 = 20-byte records, **4** = 24-byte + `object_id`), `fileMeta`, materials.

## Tooling

This template should help get you started developing with Tauri, React and Typescript in Vite.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
