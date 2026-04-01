use std::sync::Mutex;

/// Stores the formatted crash report so it survives across `catch_unwind`.
static CRASH_LOG: Mutex<Option<String>> = Mutex::new(None);

/// Install a custom panic hook that captures a full crash report.
///
/// The hook writes the crash log to disk immediately (best-effort) because
/// event-loop panics on macOS may abort at the ObjC FFI boundary before
/// `catch_unwind` can run.
pub fn install() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let bt = std::backtrace::Backtrace::force_capture();
        let report = format!(
            "Voxelle v{ver} crash report\n\
             Time: {time}\n\n\
             {info}\n\n\
             Backtrace:\n{bt}",
            ver = env!("CARGO_PKG_VERSION"),
            time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        );

        // Best-effort: write to file immediately so the log survives even if
        // the process aborts before catch_unwind can handle it.
        let path = crash_log_path();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, &report);

        // Store for catch_unwind to pick up for the dialog.
        if let Ok(mut slot) = CRASH_LOG.lock() {
            *slot = Some(report);
        }

        prev(info);
    }));
}

/// Show a crash dialog after a caught panic.
///
/// Reads the crash log captured by the panic hook, copies it to the system
/// clipboard, and presents a native dialog so the user can retrieve it.
pub fn show_crash_report() {
    let log = CRASH_LOG
        .lock()
        .ok()
        .and_then(|mut g| g.take())
        .unwrap_or_else(|| "Voxelle crashed (no details captured)".into());

    let path = crash_log_path();

    // The hook should have written the file already, but write again in case
    // it was a very early panic before the hook could finish.
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, &log);

    let clipboard_ok = arboard::Clipboard::new()
        .and_then(|mut cb| cb.set_text(&log))
        .is_ok();

    show_crash_dialog(&path, clipboard_ok);
}

fn crash_log_path() -> std::path::PathBuf {
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        return std::path::PathBuf::from(home).join("Library/Logs/Voxelle/crash.log");
    }
    #[cfg(target_os = "windows")]
    if let Some(appdata) = std::env::var_os("LOCALAPPDATA") {
        return std::path::PathBuf::from(appdata).join("Voxelle\\crash.log");
    }
    std::env::temp_dir().join("voxelle-crash.log")
}

// ---------------------------------------------------------------------------
// Platform-specific dialogs
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn show_crash_dialog(path: &std::path::Path, clipboard_ok: bool) {
    let clip_msg = if clipboard_ok {
        "The crash log has been copied to your clipboard."
    } else {
        "Could not copy to clipboard."
    };
    let path_str = path.display().to_string().replace('"', "\\\"");

    // AppleScript: `return` is the newline character inside strings.
    let script = r#"set msg to "Voxelle has crashed unexpectedly." & return & return & "CLIP_MSG" & return & return & "Saved to:" & return & "LOG_PATH"
display alert "Voxelle Crash" message msg as critical buttons {"Open Log", "OK"} default button "OK"
if button returned of result is "Open Log" then
  do shell script "open " & quoted form of "LOG_PATH"
end if"#
        .replace("CLIP_MSG", clip_msg)
        .replace("LOG_PATH", &path_str);

    let _ = std::process::Command::new("osascript")
        .args(["-e", &script])
        .status();
}

#[cfg(target_os = "windows")]
fn show_crash_dialog(path: &std::path::Path, _clipboard_ok: bool) {
    // Open the crash log in the user's default text editor.
    let _ = std::process::Command::new("cmd")
        .args(["/C", "start", "", &path.display().to_string()])
        .status();
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn show_crash_dialog(path: &std::path::Path, clipboard_ok: bool) {
    let clip = if clipboard_ok {
        " (also copied to clipboard)"
    } else {
        ""
    };
    eprintln!("Voxelle crashed. Log saved to: {}{clip}", path.display());
    let _ = std::process::Command::new("xdg-open").arg(path).status();
}
