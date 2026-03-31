//! macOS: transparent titlebar with a solid window background so the webview shell reads as one surface.
//! See <https://v2.tauri.app/learn/window-customization/#macos-transparent-titlebar-with-custom-window-background-color>
//!
//! Background RGB matches `App.css` `--app-paper` for dark (`:root`) and light (`:root[data-appearance="light"]`).

use objc2::rc::Retained;
use objc2_app_kit::{NSColor, NSWindow};
use tauri::{TitleBarStyle, WebviewWindow};

/// Dark `--app-paper` base: `rgb(12 12 14)`.
fn ns_color_dark_paper() -> Retained<NSColor> {
    NSColor::colorWithSRGBRed_green_blue_alpha(
        12.0 / 255.0,
        12.0 / 255.0,
        14.0 / 255.0,
        1.0,
    )
}

/// Light `--app-paper`: `#f5f0e6`.
fn ns_color_light_paper() -> Retained<NSColor> {
    NSColor::colorWithSRGBRed_green_blue_alpha(
        245.0 / 255.0,
        240.0 / 255.0,
        230.0 / 255.0,
        1.0,
    )
}

/// Dev-mode accent yellow: `--theme-accent: #fbc02d` / `rgb(251 192 45)`.
fn ns_color_dev_accent() -> Retained<NSColor> {
    NSColor::colorWithSRGBRed_green_blue_alpha(
        251.0 / 255.0,
        192.0 / 255.0,
        45.0 / 255.0,
        1.0,
    )
}

pub fn apply_transparent_titlebar<R: tauri::Runtime>(
    window: &WebviewWindow<R>,
) -> tauri::Result<()> {
    window.set_title_bar_style(TitleBarStyle::Transparent)?;
    set_window_background_resolved_light(window, false)
}

/// Call when [`crate::ViewerState::start_screen_light`] / resolved appearance changes (same moments as `set_start_screen_light`).
pub fn set_window_background_resolved_light<R: tauri::Runtime>(
    window: &WebviewWindow<R>,
    light: bool,
) -> tauri::Result<()> {
    let ptr = window.ns_window()?;
    if ptr.is_null() {
        return Ok(());
    }
    let ns_window: &NSWindow = unsafe { &*ptr.cast() };
    let bg = if cfg!(debug_assertions) {
        ns_color_dev_accent()
    } else if light {
        ns_color_light_paper()
    } else {
        ns_color_dark_paper()
    };
    ns_window.setBackgroundColor(Some(&*bg));
    Ok(())
}
