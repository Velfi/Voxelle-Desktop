//! Windows-only child window for hosting the wgpu rendering surface.
//!
//! On Windows, WebView2's DirectComposition visuals can conflict with a wgpu
//! swapchain on the same HWND, causing the 3D viewport to render on top of
//! the webview UI on some hardware/driver combinations. Creating a separate
//! child HWND for wgpu gives each renderer its own surface with deterministic
//! z-ordering.

use raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle, Win32WindowHandle,
    WindowsDisplayHandle,
};
use std::num::NonZeroIsize;
use windows::Win32::Foundation::HWND;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::*;

/// Owns a child HWND used as the wgpu render surface.
pub struct ChildRenderWindow {
    hwnd: HWND,
    hinstance: isize,
}

// SAFETY: The HWND is only accessed from the main thread (Tauri setup + event loop).
unsafe impl Send for ChildRenderWindow {}
unsafe impl Sync for ChildRenderWindow {}

impl ChildRenderWindow {
    /// Create a child window parented to `parent`, initially at `(0,0)` with size `(1,1)`.
    /// The caller must call [`reposition`] once the viewport dimensions are known.
    pub fn new(parent: HWND) -> Result<Self, String> {
        unsafe {
            let hinstance =
                GetModuleHandleW(None).map_err(|e| format!("GetModuleHandleW: {e}"))?;

            let class_name = windows::core::w!("VoxelleRenderSurface");

            let wc = WNDCLASSW {
                style: CS_OWNDC,
                lpfnWndProc: Some(DefWindowProcW),
                hInstance: hinstance.into(),
                lpszClassName: class_name,
                ..Default::default()
            };
            // Ignore failure — may already be registered from a previous call.
            let _ = RegisterClassW(&wc);

            let hwnd = CreateWindowExW(
                WINDOW_EX_STYLE(0),
                class_name,
                windows::core::w!(""),
                WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS,
                0,
                0,
                1,
                1,
                Some(parent),
                None,
                Some(hinstance.into()),
                None,
            )
            .map_err(|e| format!("CreateWindowExW child: {e}"))?;

            // Place behind all siblings (WebView2) in z-order.
            let _ = SetWindowPos(
                hwnd,
                Some(HWND_BOTTOM),
                0,
                0,
                1,
                1,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            );

            Ok(Self {
                hwnd,
                hinstance: hinstance.0 as isize,
            })
        }
    }

    /// Reposition and resize the child window to match the viewport region
    /// within the parent window's client area.
    pub fn reposition(&self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                Some(HWND_BOTTOM),
                x,
                y,
                w,
                h,
                SWP_NOACTIVATE,
            );
        }
    }

    /// Return a non-owning handle suitable for passing to
    /// [`wgpu::Instance::create_surface`].
    pub fn surface_handle(&self) -> ChildWindowRef {
        ChildWindowRef {
            hwnd: self.hwnd.0 as isize,
            hinstance: self.hinstance,
        }
    }
}

impl Drop for ChildRenderWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.hwnd);
        }
    }
}

// ---------------------------------------------------------------------------
// Non-owning handle implementing raw-window-handle traits for wgpu.
// ---------------------------------------------------------------------------

/// A lightweight, `Copy`-able handle wrapping a raw HWND. Implements
/// [`HasWindowHandle`] + [`HasDisplayHandle`] so it can be passed to
/// [`wgpu::Instance::create_surface`].
///
/// # Safety
/// The caller must ensure the underlying window ([`ChildRenderWindow`])
/// outlives any surface created from this handle.
#[derive(Clone, Copy)]
pub struct ChildWindowRef {
    hwnd: isize,
    hinstance: isize,
}

// SAFETY: contains only raw integer handles.
unsafe impl Send for ChildWindowRef {}
unsafe impl Sync for ChildWindowRef {}

impl HasWindowHandle for ChildWindowRef {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let mut h =
            Win32WindowHandle::new(NonZeroIsize::new(self.hwnd).expect("HWND must be non-zero"));
        h.hinstance = NonZeroIsize::new(self.hinstance);
        // SAFETY: the HWND is valid as long as ChildRenderWindow is alive.
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(RawWindowHandle::Win32(h)) })
    }
}

impl HasDisplayHandle for ChildWindowRef {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let h = WindowsDisplayHandle::new();
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(RawDisplayHandle::Windows(h)) })
    }
}
