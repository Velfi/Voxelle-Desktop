use std::path::Path;

fn main() {
    tauri_build::build();
    generate_avatar_module();
}

fn generate_avatar_module() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let avatars_dir = Path::new(&manifest_dir).join("avatars");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("avatars_generated.rs");

    // Re-run if any file in the avatars folder changes.
    println!("cargo:rerun-if-changed={}", avatars_dir.display());

    let mut names: Vec<String> = Vec::new();

    if avatars_dir.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&avatars_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.eq_ignore_ascii_case("voxelle"))
                    .unwrap_or(false)
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let stem = entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .into_owned();
            // Re-run if this specific file changes.
            println!("cargo:rerun-if-changed={}", entry.path().display());
            names.push(stem);
        }
    }

    let mut code = String::new();

    // One static per avatar.
    for name in &names {
        let const_name = avatar_const_name(name);
        code.push_str(&format!(
            "static {}: &[u8] = include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/avatars/{}.voxelle\"));\n",
            const_name, name
        ));
    }

    code.push('\n');

    // embedded_avatar_bytes()
    code.push_str("pub(crate) fn embedded_avatar_bytes(name: &str) -> Option<&'static [u8]> {\n");
    code.push_str("    match name {\n");
    for name in &names {
        let const_name = avatar_const_name(name);
        code.push_str(&format!("        {:?} => Some({}),\n", name, const_name));
    }
    code.push_str("        _ => None,\n");
    code.push_str("    }\n");
    code.push_str("}\n\n");

    // avatar_list_embedded_names()
    code.push_str("fn avatar_list_embedded_names() -> Vec<String> {\n");
    code.push_str("    vec![\n");
    for name in &names {
        code.push_str(&format!("        {:?}.to_string(),\n", name));
    }
    code.push_str("    ]\n");
    code.push_str("}\n");

    std::fs::write(&out_path, code).unwrap();
}

/// Convert an avatar stem like "HeIsRisen" → "AVATAR_HE_IS_RISEN".
fn avatar_const_name(stem: &str) -> String {
    let mut out = String::from("AVATAR_");
    let mut prev_upper = true;
    for ch in stem.chars() {
        if ch.is_uppercase() && !prev_upper {
            out.push('_');
        }
        out.push(ch.to_ascii_uppercase());
        prev_upper = ch.is_uppercase();
    }
    out
}
