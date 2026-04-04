use std::sync::Arc;
use tauri::{AppHandle, Manager};

use crate::{read_recent_files, RenderingMode, ViewerState};

#[cfg(desktop)]
pub(crate) fn vd_about_metadata(app: &AppHandle) -> tauri::Result<tauri::menu::AboutMetadata<'_>> {
    use tauri::menu::AboutMetadata;
    // Public repo (matches updater endpoint in `tauri.conf.json`).
    const GITHUB_VD: &str = "https://github.com/Velfi/Voxelle-Desktop";
    let pkg = app.package_info();
    let mut m = AboutMetadata {
        name: Some(pkg.name.clone()),
        version: Some(pkg.version.to_string()),
        website: Some(GITHUB_VD.into()),
        website_label: Some("GitHub".into()),
        comments: Some("Voxel art, together on the desktop.".into()),
        copyright: app.config().bundle.copyright.clone(),
        ..Default::default()
    };
    #[cfg(target_os = "macos")]
    {
        // NSAboutPanel only shows a subset of fields; `credits` is the scrollable body with the link.
        m.website = None;
        m.website_label = None;
        m.comments = None;
        m.credits = Some(format!(
            "Voxel art, together on the desktop.\n\n{GITHUB_VD}"
        ));
    }
    Ok(m)
}

/// Native menu handles for [`CheckMenuItem`] sync (match material, debug overlay) and
/// selection/voxel-dependent enable state (mirrors web `MenuBar` disabled rules).
#[cfg(desktop)]
/// Holds the "Open Recent" submenu so it can be rebuilt when the list changes.
pub(crate) struct RecentMenuState {
    pub submenu: tauri::menu::Submenu<tauri::Wry>,
}

/// Rebuild the contents of the "Open Recent" submenu from disk.
#[cfg(desktop)]
pub(crate) fn rebuild_recent_submenu(app: &AppHandle, submenu: &tauri::menu::Submenu<tauri::Wry>) {
    use tauri::menu::{MenuItem, PredefinedMenuItem};
    // Clear existing items.
    while submenu.items().map(|v| v.len()).unwrap_or(0) > 0 {
        let _ = submenu.remove_at(0);
    }
    let recent = read_recent_files(app);
    if recent.is_empty() {
        let empty = MenuItem::with_id(
            app,
            "recent_none",
            "No Recent Projects",
            false,
            None::<&str>,
        );
        if let Ok(item) = empty {
            let _ = submenu.append(&item);
        }
    } else {
        for (i, path) in recent.iter().enumerate() {
            // Show just the filename, with the full path as the menu ID.
            let display = std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            let id = format!("recent_file_{i}");
            let item = MenuItem::with_id(app, &id, &display, true, None::<&str>);
            if let Ok(item) = item {
                let _ = submenu.append(&item);
            }
        }
        let sep = PredefinedMenuItem::separator(app);
        if let Ok(sep) = sep {
            let _ = submenu.append(&sep);
        }
        let clear = MenuItem::with_id(app, "recent_clear", "Clear Recent", true, None::<&str>);
        if let Ok(item) = clear {
            let _ = submenu.append(&item);
        }
    }
}

pub(crate) struct SelectionMenuState {
    pub match_material: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub viewport_cursor_debug: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub logo_light_controls: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub view_show_borders: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub view_hide_ui: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_greedy: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_marching: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_dual: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub render_ray: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub ortho_toggle: tauri::menu::CheckMenuItem<tauri::Wry>,
    pub sel_all: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_by_color: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_connected: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_coplanar: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_coplanar_empty: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_grow: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_shrink: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_invert: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_all: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_inner: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_voxels: tauri::menu::MenuItem<tauri::Wry>,
    pub sel_deselect_empty: tauri::menu::MenuItem<tauri::Wry>,
}

/// Order top-level menus as: … File, Edit, **Selection**, View, Window, **Voxels**, Help, **Debug**
/// (after the app menu on macOS). [`Menu::default`] would leave Help before appended items; we
/// insert Selection / Voxels / Debug at the correct indices instead of appending at the end.
#[cfg(desktop)]
fn place_voxelle_custom_top_level_menus<R: tauri::Runtime>(
    menu: &tauri::menu::Menu<R>,
    selection_submenu: &tauri::menu::Submenu<R>,
    voxels_submenu: &tauri::menu::Submenu<R>,
    debug_menu: &tauri::menu::Submenu<R>,
) -> tauri::Result<()> {
    use tauri::menu::MenuItemKind;

    fn submenu_title<R2: tauri::Runtime>(kind: &MenuItemKind<R2>) -> Option<String> {
        match kind {
            MenuItemKind::Submenu(s) => s.text().ok(),
            _ => None,
        }
    }

    #[cfg(target_os = "macos")]
    {
        let items = menu.items()?;
        let edit_idx = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Edit"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "menubar: Edit submenu not found",
                )
            })?;
        menu.insert(selection_submenu, edit_idx + 1)?;

        let items = menu.items()?;
        let window_idx = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Window"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "menubar: Window submenu not found",
                )
            })?;
        menu.insert(voxels_submenu, window_idx + 1)?;

        let items = menu.items()?;
        let help_idx = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Help"))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "menubar: Help submenu not found",
                )
            })?;
        menu.insert(debug_menu, help_idx + 1)?;
        return Ok(());
    }

    #[cfg(not(target_os = "macos"))]
    {
        let items = menu.items()?;
        if let Some(edit_idx) = items
            .iter()
            .position(|i| submenu_title(i).as_deref() == Some("Edit"))
        {
            menu.insert(selection_submenu, edit_idx + 1)?;
        } else {
            menu.append(selection_submenu)?;
        }
        menu.append(voxels_submenu)?;
        menu.append(debug_menu)?;
        Ok(())
    }
}

#[cfg(desktop)]
pub(crate) fn install_app_menu(app: &AppHandle) -> tauri::Result<(SelectionMenuState, RecentMenuState)> {
    use tauri::menu::{CheckMenuItem, Menu, MenuItem, MenuItemKind, PredefinedMenuItem, Submenu};

    let menu = Menu::default(app)?;
    let about_item = PredefinedMenuItem::about(app, None, Some(vd_about_metadata(app)?))?;
    let new_item = MenuItem::with_id(app, "new_project", "New Project…", true, None::<&str>)?;
    let open_item = MenuItem::with_id(app, "open_voxelle", "Open…", true, Some("CommandOrCtrl+O"))?;
    let save_item = MenuItem::with_id(app, "menu_save", "Save", true, Some("CommandOrCtrl+S"))?;
    let save_as_item = MenuItem::with_id(
        app,
        "menu_save_as",
        "Save As…",
        true,
        Some("CommandOrCtrl+Shift+S"),
    )?;
    let close_project_item =
        MenuItem::with_id(app, "menu_close_project", "Close Project", true, Some("CommandOrCtrl+W"))?;
    let export_glb_item =
        MenuItem::with_id(app, "menu_export_glb", "Export GLB…", true, None::<&str>)?;
    let open_recent_submenu = Submenu::with_id(app, "open_recent_submenu", "Open Recent", true)?;
    rebuild_recent_submenu(app, &open_recent_submenu);
    let undo_item = MenuItem::with_id(app, "menu_undo", "Undo", true, Some("CommandOrCtrl+Z"))?;
    let redo_item = MenuItem::with_id(
        app,
        "menu_redo",
        "Redo",
        true,
        Some("CommandOrCtrl+Shift+Z"),
    )?;
    let collab_start_item = MenuItem::with_id(
        app,
        "menu_collab_start",
        "Start Session",
        true,
        Some("CommandOrCtrl+Shift+L"),
    )?;
    let collab_join_item =
        MenuItem::with_id(app, "menu_collab_join", "Join Session…", true, None::<&str>)?;
    let collab_leave_item = MenuItem::with_id(
        app,
        "menu_collab_leave",
        "Leave Session",
        true,
        None::<&str>,
    )?;
    let collab_submenu = Submenu::with_items(
        app,
        "Collaboration",
        true,
        &[&collab_start_item, &collab_join_item, &collab_leave_item],
    )?;
    let chat_panel_item = MenuItem::with_id(app, "menu_chat_panel", "Chat", true, None::<&str>)?;
    let check_updates_item = MenuItem::with_id(
        app,
        "menu_check_updates",
        "Check for Updates…",
        true,
        None::<&str>,
    )?;
    let preferences_item = MenuItem::with_id(
        app,
        "menu_preferences",
        "Preferences…",
        true,
        Some("CommandOrCtrl+,"),
    )?;
    let debug_viewport_cursor = CheckMenuItem::with_id(
        app,
        "debug_viewport_cursor_overlay",
        "Viewport cursor debug overlay",
        true,
        false,
        None::<&str>,
    )?;
    let debug_logo_light = CheckMenuItem::with_id(
        app,
        "debug_logo_light_controls",
        "Logo controls",
        true,
        false,
        None::<&str>,
    )?;
    let debug_copy_perf = MenuItem::with_id(
        app,
        "debug_copy_performance",
        "Copy performance info",
        true,
        None::<&str>,
    )?;
    let debug_raytrace_bench = MenuItem::with_id(
        app,
        "debug_raytrace_benchmark",
        "Ray trace benchmark (50 frames)",
        true,
        None::<&str>,
    )?;
    let debug_clear_autosaves_item = MenuItem::with_id(
        app,
        "debug_clear_autosaves",
        "Clear autosaves and session…",
        true,
        None::<&str>,
    )?;
    let debug_test_crash = MenuItem::with_id(
        app,
        "debug_test_crash",
        "Test crash report…",
        true,
        None::<&str>,
    )?;
    let debug_menu = Submenu::with_items(
        app,
        "Debug",
        true,
        &[
            &debug_viewport_cursor,
            &debug_logo_light,
            &debug_copy_perf,
            &debug_raytrace_bench,
            &debug_clear_autosaves_item,
            &debug_test_crash,
        ],
    )?;
    let sep = PredefinedMenuItem::separator(app)?;
    let current_mode = *app.state::<Arc<ViewerState>>().rendering_mode.lock();
    let view_render_greedy = CheckMenuItem::with_id(
        app,
        "view_render_greedy",
        "Blocky",
        true,
        matches!(current_mode, RenderingMode::Greedy),
        None::<&str>,
    )?;
    let view_render_marching = CheckMenuItem::with_id(
        app,
        "view_render_marching",
        "Smooth",
        true,
        matches!(current_mode, RenderingMode::MarchingCubes),
        None::<&str>,
    )?;
    let view_render_dual = CheckMenuItem::with_id(
        app,
        "view_render_dual",
        "Puffy",
        true,
        matches!(current_mode, RenderingMode::DualContour),
        None::<&str>,
    )?;
    let sep_before_ray = PredefinedMenuItem::separator(app)?;
    let view_render_ray = CheckMenuItem::with_id(
        app,
        "menu_view_render_ray",
        "Ray Tracing",
        true,
        matches!(current_mode, RenderingMode::Ray),
        None::<&str>,
    )?;
    let rendering_submenu = Submenu::with_items(
        app,
        "Rendering",
        true,
        &[
            &view_render_greedy,
            &view_render_marching,
            &view_render_dual,
            &sep_before_ray,
            &view_render_ray,
        ],
    )?;
    let is_ortho = !app.state::<Arc<ViewerState>>().camera.lock().perspective;
    let ortho_view_item = CheckMenuItem::with_id(
        app,
        "menu_view_ortho",
        "Orthographic",
        true,
        is_ortho,
        None::<&str>,
    )?;
    let sep_view_extras = PredefinedMenuItem::separator(app)?;
    let view_show_borders = CheckMenuItem::with_id(
        app,
        "menu_view_show_borders",
        "Show borders",
        true,
        false,
        None::<&str>,
    )?;
    let view_hide_ui = CheckMenuItem::with_id(
        app,
        "menu_view_hide_ui",
        "Hide UI",
        true,
        false,
        None::<&str>,
    )?;
    let sep_view_stamp = PredefinedMenuItem::separator(app)?;
    let view_stamp_book = MenuItem::with_id(
        app,
        "menu_view_stamp_book",
        "Stamp book…",
        true,
        None::<&str>,
    )?;
    let sep_before_chat = PredefinedMenuItem::separator(app)?;

    let mut file_inserted = false;
    let mut edit_inserted = false;
    let mut view_inserted = false;
    for item in menu.items()? {
        if let MenuItemKind::Submenu(sub) = item {
            let text = sub.text()?;
            #[cfg(target_os = "macos")]
            if text == app.package_info().name.clone() {
                sub.remove_at(0)?;
                sub.insert(&about_item, 0)?;
                sub.insert(&preferences_item, 1)?;
                sub.insert(&check_updates_item, 2)?;
            }
            #[cfg(not(target_os = "macos"))]
            if text == "Help" {
                sub.remove_at(0)?;
                sub.insert(&about_item, 0)?;
            }
            if text == "File" {
                sub.prepend_items(&[
                    &new_item,
                    &open_item,
                    &open_recent_submenu,
                    &save_item,
                    &save_as_item,
                    &close_project_item,
                    &export_glb_item,
                    &sep,
                ])?;
                // Also under File so it works when OS menus are localized (no reliance on "View").
                sub.append(&collab_submenu)?;
                #[cfg(not(target_os = "macos"))]
                sub.append(&check_updates_item)?;
                file_inserted = true;
            } else if text == "Edit" {
                #[cfg(not(target_os = "macos"))]
                {
                    let sep_edit = PredefinedMenuItem::separator(app)?;
                    sub.append(&sep_edit)?;
                    sub.append(&preferences_item)?;
                }
                // macOS (and many platforms) already ship Undo/Redo in Edit. Do not append ours — that
                // duplicates entries. Voxel undo/redo uses the same shortcuts via the webview (`App.tsx`).
                edit_inserted = true;
            } else if text == "View" {
                sub.append(&rendering_submenu)?;
                sub.append(&ortho_view_item)?;
                sub.append(&sep_view_extras)?;
                sub.append(&view_show_borders)?;
                sub.append(&view_hide_ui)?;
                sub.append(&sep_view_stamp)?;
                sub.append(&view_stamp_book)?;
                sub.append(&sep_before_chat)?;
                sub.append(&chat_panel_item)?;
                view_inserted = true;
            }
        }
    }

    if !view_inserted {
        let view_menu = Submenu::with_items(
            app,
            "View",
            true,
            &[
                &rendering_submenu,
                &ortho_view_item,
                &view_render_ray,
                &sep_view_extras,
                &view_show_borders,
                &view_hide_ui,
                &sep_view_stamp,
                &view_stamp_book,
                &sep_before_chat,
                &chat_panel_item,
            ],
        )?;
        menu.append(&view_menu)?;
    }

    if !file_inserted {
        let close = PredefinedMenuItem::close_window(app, None)?;
        #[cfg(target_os = "macos")]
        {
            let file_menu = Submenu::with_items(
                app,
                "File",
                true,
                &[
                    &new_item,
                    &open_item,
                    &save_item,
                    &save_as_item,
                    &export_glb_item,
                    &sep,
                    &collab_submenu,
                    &close,
                ],
            )?;
            menu.prepend(&file_menu)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let file_menu = Submenu::with_items(
                app,
                "File",
                true,
                &[
                    &new_item,
                    &open_item,
                    &save_item,
                    &save_as_item,
                    &export_glb_item,
                    &sep,
                    &collab_submenu,
                    &check_updates_item,
                    &close,
                ],
            )?;
            menu.prepend(&file_menu)?;
        }
    }

    if !edit_inserted {
        #[cfg(target_os = "macos")]
        {
            let edit_menu = Submenu::with_items(app, "Edit", true, &[&undo_item, &redo_item])?;
            menu.append(&edit_menu)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let sep_edit = PredefinedMenuItem::separator(app)?;
            let edit_menu = Submenu::with_items(
                app,
                "Edit",
                true,
                &[&undo_item, &redo_item, &sep_edit, &preferences_item],
            )?;
            menu.append(&edit_menu)?;
        }
    }

    let voxel_hide_selected = MenuItem::with_id(
        app,
        "menu_voxel_hide_selected",
        "Hide selected",
        true,
        None::<&str>,
    )?;
    let voxel_unhide_all = MenuItem::with_id(
        app,
        "menu_voxel_unhide_all",
        "Unhide all",
        true,
        None::<&str>,
    )?;
    let sep_voxel_1 = PredefinedMenuItem::separator(app)?;
    let voxel_hollow =
        MenuItem::with_id(app, "menu_voxel_hollow", "Hollow out", true, None::<&str>)?;
    let voxel_scale = MenuItem::with_id(
        app,
        "menu_voxel_scale",
        "Scale by factor…",
        true,
        None::<&str>,
    )?;
    let voxel_rotate = MenuItem::with_id(
        app,
        "menu_voxel_rotate",
        "Rotate by degrees…",
        true,
        None::<&str>,
    )?;
    let sep_voxel_2 = PredefinedMenuItem::separator(app)?;
    let voxel_mirror_hdr =
        MenuItem::with_id(app, "menu_voxel_mirror_hdr", "Mirror", false, None::<&str>)?;
    let voxel_mirror_x = MenuItem::with_id(
        app,
        "menu_voxel_mirror_x",
        "Across X (YZ plane)",
        true,
        None::<&str>,
    )?;
    let voxel_mirror_y = MenuItem::with_id(
        app,
        "menu_voxel_mirror_y",
        "Across Y (XZ plane)",
        true,
        None::<&str>,
    )?;
    let voxel_mirror_z = MenuItem::with_id(
        app,
        "menu_voxel_mirror_z",
        "Across Z (XY plane)",
        true,
        None::<&str>,
    )?;
    let voxels_submenu = Submenu::with_items(
        app,
        "Voxels",
        true,
        &[
            &voxel_hide_selected,
            &voxel_unhide_all,
            &sep_voxel_1,
            &voxel_hollow,
            &voxel_scale,
            &voxel_rotate,
            &sep_voxel_2,
            &voxel_mirror_hdr,
            &voxel_mirror_x,
            &voxel_mirror_y,
            &voxel_mirror_z,
        ],
    )?;

    let menu_sel_all = MenuItem::with_id(app, "menu_sel_all", "Select All", true, None::<&str>)?;
    let menu_sel_by_color = MenuItem::with_id(
        app,
        "menu_sel_by_color",
        "Select by Color",
        true,
        None::<&str>,
    )?;
    let menu_sel_connected = MenuItem::with_id(
        app,
        "menu_sel_connected",
        "Select Connected",
        true,
        None::<&str>,
    )?;
    let menu_sel_coplanar = MenuItem::with_id(
        app,
        "menu_sel_coplanar",
        "Select Coplanar Faces",
        true,
        None::<&str>,
    )?;
    let menu_sel_coplanar_empty = MenuItem::with_id(
        app,
        "menu_sel_coplanar_empty",
        "Select Coplanar Void",
        true,
        None::<&str>,
    )?;
    let menu_sel_sep1 = PredefinedMenuItem::separator(app)?;
    let menu_sel_grow = MenuItem::with_id(app, "menu_sel_grow", "Grow", true, None::<&str>)?;
    let menu_sel_shrink = MenuItem::with_id(app, "menu_sel_shrink", "Shrink", true, None::<&str>)?;
    let menu_sel_invert = MenuItem::with_id(app, "menu_sel_invert", "Invert", true, None::<&str>)?;
    let menu_sel_sep2 = PredefinedMenuItem::separator(app)?;
    let menu_sel_deselect_all = MenuItem::with_id(
        app,
        "menu_sel_deselect_all",
        "Deselect All",
        true,
        None::<&str>,
    )?;
    let menu_sel_deselect_inner = MenuItem::with_id(
        app,
        "menu_sel_deselect_inner",
        "Deselect Inner Voxels",
        true,
        None::<&str>,
    )?;
    let menu_sel_deselect_voxels = MenuItem::with_id(
        app,
        "menu_sel_deselect_voxels",
        "Deselect Voxels",
        true,
        None::<&str>,
    )?;
    let menu_sel_deselect_empty = MenuItem::with_id(
        app,
        "menu_sel_deselect_empty",
        "Deselect Empty Spaces",
        true,
        None::<&str>,
    )?;
    let menu_sel_sep3 = PredefinedMenuItem::separator(app)?;
    let menu_sel_mode_replace =
        MenuItem::with_id(app, "menu_sel_mode_replace", "Replace", true, None::<&str>)?;
    let menu_sel_mode_add = MenuItem::with_id(
        app,
        "menu_sel_mode_add",
        "Add to Selection",
        true,
        None::<&str>,
    )?;
    let menu_sel_mode_subtract = MenuItem::with_id(
        app,
        "menu_sel_mode_subtract",
        "Subtract from Selection",
        true,
        None::<&str>,
    )?;
    let menu_sel_mode_intersect = MenuItem::with_id(
        app,
        "menu_sel_mode_intersect",
        "Intersect with Selection",
        true,
        None::<&str>,
    )?;
    let menu_sel_sep4 = PredefinedMenuItem::separator(app)?;
    let menu_sel_match_material = CheckMenuItem::with_id(
        app,
        "menu_sel_match_material",
        "Match Material",
        true,
        false,
        None::<&str>,
    )?;
    let selection_submenu = Submenu::with_items(
        app,
        "Selection",
        true,
        &[
            &menu_sel_all,
            &menu_sel_by_color,
            &menu_sel_connected,
            &menu_sel_coplanar,
            &menu_sel_coplanar_empty,
            &menu_sel_sep1,
            &menu_sel_grow,
            &menu_sel_shrink,
            &menu_sel_invert,
            &menu_sel_sep2,
            &menu_sel_deselect_all,
            &menu_sel_deselect_inner,
            &menu_sel_deselect_voxels,
            &menu_sel_deselect_empty,
            &menu_sel_sep3,
            &menu_sel_mode_replace,
            &menu_sel_mode_add,
            &menu_sel_mode_subtract,
            &menu_sel_mode_intersect,
            &menu_sel_sep4,
            &menu_sel_match_material,
        ],
    )?;

    place_voxelle_custom_top_level_menus(&menu, &selection_submenu, &voxels_submenu, &debug_menu)?;
    menu.set_as_app_menu()?;
    Ok((
        SelectionMenuState {
            match_material: menu_sel_match_material.clone(),
            viewport_cursor_debug: debug_viewport_cursor.clone(),
            logo_light_controls: debug_logo_light.clone(),
            view_show_borders: view_show_borders.clone(),
            view_hide_ui: view_hide_ui.clone(),
            render_greedy: view_render_greedy.clone(),
            render_marching: view_render_marching.clone(),
            render_dual: view_render_dual.clone(),
            render_ray: view_render_ray.clone(),
            ortho_toggle: ortho_view_item.clone(),
            sel_all: menu_sel_all.clone(),
            sel_by_color: menu_sel_by_color.clone(),
            sel_connected: menu_sel_connected.clone(),
            sel_coplanar: menu_sel_coplanar.clone(),
            sel_coplanar_empty: menu_sel_coplanar_empty.clone(),
            sel_grow: menu_sel_grow.clone(),
            sel_shrink: menu_sel_shrink.clone(),
            sel_invert: menu_sel_invert.clone(),
            sel_deselect_all: menu_sel_deselect_all.clone(),
            sel_deselect_inner: menu_sel_deselect_inner.clone(),
            sel_deselect_voxels: menu_sel_deselect_voxels.clone(),
            sel_deselect_empty: menu_sel_deselect_empty.clone(),
        },
        RecentMenuState {
            submenu: open_recent_submenu,
        },
    ))
}
