//! macOS: drive `NSUndoManager` so the system Edit → Undo/Redo stack matches solo voxel edits.
//! Collaboration uses the existing Rust/collab paths only (`macos_undo` is not called).

use std::sync::Once;

use block2::RcBlock;
use dispatch2::DispatchQueue;
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2_app_kit::NSWindow;
use objc2_foundation::{ns_string, NSUndoManager};

use tauri::{AppHandle, Manager, Runtime};

use crate::perform_solo_voxel_redo;
use crate::perform_solo_voxel_undo;
use crate::ViewerState;

static CONFIGURE_UNDO: Once = Once::new();
static UNDO_TARGET_PTR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn undo_target() -> &'static NSObject {
    let p = *UNDO_TARGET_PTR.get_or_init(|| {
        let o = NSObject::new();
        Retained::into_raw(o) as usize
    });
    unsafe { &*(p as *const NSObject) }
}

fn run_on_main<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if objc2::MainThreadMarker::new().is_some() {
        f()
    } else {
        let mut out = None;
        DispatchQueue::main().exec_sync(|| {
            out = Some(f());
        });
        out.expect("main queue sync returned empty")
    }
}

fn undo_manager_for_main_window<R: Runtime>(app: &AppHandle<R>) -> Option<Retained<NSUndoManager>> {
    let window = app.get_webview_window("main")?;
    let ptr = window.ns_window().ok()?;
    if ptr.is_null() {
        return None;
    }
    let window: &NSWindow = unsafe { &*ptr.cast() };
    unsafe { msg_send![window, undoManager] }
}

fn configure_undo_manager(um: &NSUndoManager) {
    CONFIGURE_UNDO.call_once(|| {
        um.setGroupsByEvent(false);
    });
}

/// Register one undo step with `NSUndoManager` after a solo edit has been applied and pushed to
/// `edit_undo`. Must mirror [`perform_solo_voxel_undo`] / [`perform_solo_voxel_redo`] pairing.
pub fn register_solo_edit_completed(app: &AppHandle, state: &std::sync::Arc<ViewerState>) {
    let app = app.clone();
    let state = std::sync::Arc::clone(state);
    run_on_main(move || {
        register_solo_edit_completed_on_main(&app, &state);
    });
}

fn register_solo_edit_completed_on_main(app: &AppHandle, state: &std::sync::Arc<ViewerState>) {
    let Some(um) = undo_manager_for_main_window(app) else {
        return;
    };
    configure_undo_manager(&um);
    let target = undo_target();
    let um_shared = um.clone();
    let state_u = std::sync::Arc::clone(state);
    let app_u = app.clone();
    let undo_block = RcBlock::new(move |_t: std::ptr::NonNull<AnyObject>| {
        match perform_solo_voxel_undo(&state_u, &app_u) {
            Ok(true) => {
                register_redo_on_main(&um_shared, std::sync::Arc::clone(&state_u), app_u.clone());
            }
            Ok(false) => {}
            Err(e) => eprintln!("voxelle: NSUndoManager solo undo failed: {e}"),
        }
    });
    unsafe {
        um.beginUndoGrouping();
        um.setActionName(ns_string!("Voxel Edit"));
        um.registerUndoWithTarget_handler(
            target,
            &*undo_block as &block2::DynBlock<dyn Fn(std::ptr::NonNull<AnyObject>)>,
        );
        um.endUndoGrouping();
    }
}

fn register_redo_on_main(
    um: &Retained<NSUndoManager>,
    state: std::sync::Arc<ViewerState>,
    app: AppHandle,
) {
    let um_shared = um.clone();
    let state_r = std::sync::Arc::clone(&state);
    let app_r = app.clone();
    let redo_block = RcBlock::new(move |_t: std::ptr::NonNull<AnyObject>| {
        match perform_solo_voxel_redo(&state_r, &app_r) {
            Ok(true) => {
                register_undo_only_on_main(
                    &um_shared,
                    std::sync::Arc::clone(&state_r),
                    app_r.clone(),
                );
            }
            Ok(false) => {}
            Err(e) => eprintln!("voxelle: NSUndoManager solo redo failed: {e}"),
        }
    });
    let target = undo_target();
    unsafe {
        um.beginUndoGrouping();
        um.setActionName(ns_string!("Voxel Edit"));
        um.registerUndoWithTarget_handler(
            target,
            &*redo_block as &block2::DynBlock<dyn Fn(std::ptr::NonNull<AnyObject>)>,
        );
        um.endUndoGrouping();
    }
}

fn register_undo_only_on_main(
    um: &Retained<NSUndoManager>,
    state: std::sync::Arc<ViewerState>,
    app: AppHandle,
) {
    let um_shared = um.clone();
    let state_u = std::sync::Arc::clone(&state);
    let app_u = app.clone();
    let undo_block = RcBlock::new(move |_t: std::ptr::NonNull<AnyObject>| {
        match perform_solo_voxel_undo(&state_u, &app_u) {
            Ok(true) => {
                register_redo_on_main(&um_shared, std::sync::Arc::clone(&state_u), app_u.clone());
            }
            Ok(false) => {}
            Err(e) => eprintln!("voxelle: NSUndoManager solo undo failed: {e}"),
        }
    });
    let target = undo_target();
    unsafe {
        um.beginUndoGrouping();
        um.setActionName(ns_string!("Voxel Edit"));
        um.registerUndoWithTarget_handler(
            target,
            &*undo_block as &block2::DynBlock<dyn Fn(std::ptr::NonNull<AnyObject>)>,
        );
        um.endUndoGrouping();
    }
}

/// Clear AppKit undo/redo when Rust stacks are cleared (e.g. new file load).
pub fn clear_all<R: Runtime>(app: &AppHandle<R>) {
    let app = app.clone();
    run_on_main(move || {
        let Some(um) = undo_manager_for_main_window(&app) else {
            return;
        };
        um.removeAllActions();
    });
}

/// Route solo undo through `NSUndoManager` when it has a registered action (`canUndo`).
/// Returns `true` if `undo()` ran (the registered handler applies the Rust inverse).
pub fn solo_undo_via_system(app: &AppHandle) -> bool {
    run_on_main(|| {
        let Some(um) = undo_manager_for_main_window(app) else {
            return false;
        };
        if !um.canUndo() {
            return false;
        }
        um.undo();
        true
    })
}

/// Same for redo.
pub fn solo_redo_via_system(app: &AppHandle) -> bool {
    run_on_main(|| {
        let Some(um) = undo_manager_for_main_window(app) else {
            return false;
        };
        if !um.canRedo() {
            return false;
        }
        um.redo();
        true
    })
}
