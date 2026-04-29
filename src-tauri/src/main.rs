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
    tauri::Builder::default()
        .setup(|app| {
            let handle = app.handle().clone();

            // Reserve a port, spawn sidecar, and wait for /api/health.
            // Retries up to 3 times with a fresh port to absorb port-race or
            // slow-startup failures.
            let (sidecar_handle, port) = sidecar::spawn_with_retry(&handle, 3, 10_000)
                .expect("Failed to start keiba-ai-backend sidecar");

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
