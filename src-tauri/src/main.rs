#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sidecar;

use std::sync::Mutex;
use tauri::{Manager, WindowEvent};

struct AppState {
    api_port: u16,
    sidecar: Mutex<Option<sidecar::SidecarHandle>>,
}

/// Expose the dynamically assigned API port to the frontend via invoke.
#[tauri::command]
fn get_api_port(state: tauri::State<AppState>) -> u16 {
    state.api_port
}

fn main() {
    let port = sidecar::reserve_port().expect("Failed to reserve a free port for the API sidecar");

    tauri::Builder::default()
        .setup(move |app| {
            let handle = app.handle().clone();

            let sidecar_handle =
                sidecar::spawn(&handle, port).expect("Failed to spawn keiba-ai-backend sidecar");

            // Wait up to 10 s for the backend health check to pass before
            // the WebView becomes interactive (non-blocking in practice because
            // setup runs before the main window is shown).
            if !sidecar::wait_for_ready(port, 10_000, 200) {
                eprintln!(
                    "Warning: keiba-ai-backend did not become healthy within 10 s on port {}",
                    port
                );
            }

            app.manage(AppState {
                api_port: port,
                sidecar: Mutex::new(Some(sidecar_handle)),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_api_port])
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                let app = window.app_handle();
                if let Some(state) = app.try_state::<AppState>() {
                    if let Ok(mut guard) = state.sidecar.lock() {
                        if let Some(handle) = guard.take() {
                            sidecar::shutdown(handle, state.api_port);
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
