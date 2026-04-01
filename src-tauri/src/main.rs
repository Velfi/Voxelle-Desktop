// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    voxelle_desktop_lib::crash_guard::install();

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        voxelle_desktop_lib::run();
    }));

    if result.is_err() {
        voxelle_desktop_lib::crash_guard::show_crash_report();
    }
}
