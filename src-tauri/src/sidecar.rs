use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use tauri::{AppHandle, Manager};

/// Newtype around the Win32 HANDLE so the whole SidecarHandle can be moved
/// across threads. The underlying handle is just an opaque kernel id; the
/// Win32 calls we use against it (CloseHandle / SetInformationJobObject /
/// AssignProcessToJobObject) are themselves thread-safe, so Send+Sync is sound.
#[cfg(target_os = "windows")]
struct JobHandle(windows::Win32::Foundation::HANDLE);

#[cfg(target_os = "windows")]
unsafe impl Send for JobHandle {}
#[cfg(target_os = "windows")]
unsafe impl Sync for JobHandle {}

pub struct SidecarHandle {
    pub child: Child,
    #[cfg(target_os = "windows")]
    job_handle: JobHandle,
}

/// Bind to 127.0.0.1:0, record the assigned port, then drop the listener so
/// the port is free when the sidecar process later binds to it.
///
/// There is a small race window between the listener being dropped and the
/// sidecar binding to the port. In practice this is mitigated by the caller
/// retrying via `spawn_with_retry` — if the sidecar fails to become healthy
/// (e.g. another process grabbed the port), the caller obtains a fresh port.
pub fn reserve_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    // listener is dropped here, releasing the port
    Ok(port)
}

/// Reserve a port, spawn the sidecar, and wait for it to become healthy.
/// Retries up to `max_attempts` times with a fresh port if the sidecar fails
/// to start (covers transient port races and slow startup).
pub fn spawn_with_retry(
    app: &AppHandle,
    max_attempts: u32,
    health_timeout_ms: u64,
) -> Result<(SidecarHandle, u16), Box<dyn std::error::Error>> {
    let mut last_err: Option<Box<dyn std::error::Error>> = None;

    for attempt in 1..=max_attempts {
        let port = match reserve_port() {
            Ok(p) => p,
            Err(e) => {
                last_err = Some(Box::new(e));
                continue;
            }
        };

        let handle = match spawn(app, port) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("attempt {}/{}: spawn failed: {}", attempt, max_attempts, e);
                last_err = Some(e);
                continue;
            }
        };

        if wait_for_ready(port, health_timeout_ms, 200) {
            return Ok((handle, port));
        }

        eprintln!(
            "attempt {}/{}: sidecar did not become healthy on port {} within {} ms — retrying",
            attempt, max_attempts, port, health_timeout_ms
        );
        // Kill the unhealthy child before retrying with a new port.
        shutdown(handle, port);
    }

    Err(last_err.unwrap_or_else(|| {
        format!("sidecar failed to become healthy after {} attempts", max_attempts).into()
    }))
}

/// Return the OS + arch target triple string used by Tauri's externalBin naming.
fn target_triple() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return "x86_64-pc-windows-msvc";
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    return "aarch64-pc-windows-msvc";
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    return "x86_64-apple-darwin";
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return "aarch64-apple-darwin";
    #[cfg(all(
        not(target_os = "windows"),
        not(target_os = "macos"),
        target_arch = "x86_64"
    ))]
    return "x86_64-unknown-linux-gnu";
    #[cfg(all(
        not(target_os = "windows"),
        not(target_os = "macos"),
        target_arch = "aarch64"
    ))]
    return "aarch64-unknown-linux-gnu";
    #[allow(unreachable_code)]
    "unknown-unknown-unknown"
}

/// Resolve the sidecar binary path from the app's resource directory.
///
/// Tauri's `externalBin` places the binary alongside the app with a target
/// triple suffix, e.g. `keiba-ai-backend-x86_64-pc-windows-msvc.exe`.
fn resolve_sidecar_path(app: &AppHandle) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let resource_dir = app.path().resource_dir()?;

    let exe_suffix = if cfg!(target_os = "windows") {
        ".exe"
    } else {
        ""
    };
    let binary_name = format!("keiba-ai-backend-{}{}", target_triple(), exe_suffix);

    let path = resource_dir.join("binaries").join(&binary_name);
    if !path.exists() {
        return Err(format!(
            "sidecar binary not found at {}. Run scripts/build_backend.sh first.",
            path.display()
        )
        .into());
    }
    Ok(path)
}

/// Spawn the FastAPI sidecar process with KEIBA_API_PORT set.
///
/// On Windows the child process is added to a Job Object so it is
/// automatically terminated if Tauri exits unexpectedly (e.g. crash).
pub fn spawn(app: &AppHandle, port: u16) -> Result<SidecarHandle, Box<dyn std::error::Error>> {
    let binary_path = resolve_sidecar_path(app)?;

    // Tauri の release ビルドは windows_subsystem="windows" のため stdout/stderr の
    // console handle が無効。Stdio::inherit() で渡すと子プロセス側で uvicorn の
    // ログ初期化が失敗してすぐ落ちる。そのため明示的に null にする。
    // ログが必要になったらファイルにリダイレクトする方針。

    // PyInstaller 環境では keiba_ai/core/paths.py の _repo_root() が .git を
    // 見つけられず Path.cwd() にフォールバックして二重パスを生成するため、
    // Tauri 側から data 置き場を明示する（exe の隣 = games/keiba-ai/data）。
    let data_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("data")));
    if let Some(d) = &data_dir {
        let _ = std::fs::create_dir_all(d);
    }

    let mut command = Command::new(&binary_path);
    command
        .env("KEIBA_API_PORT", port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(d) = &data_dir {
        command.env("KEIBA_DATA_DIR", d);
    }
    let child = command
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar '{}': {}", binary_path.display(), e))?;

    #[cfg(target_os = "windows")]
    let job_handle = JobHandle(attach_job_object(&child)?);

    Ok(SidecarHandle {
        child,
        #[cfg(target_os = "windows")]
        job_handle,
    })
}

/// Gracefully shut down the sidecar: first try the HTTP shutdown endpoint,
/// then fall back to killing the process.
pub fn shutdown(mut handle: SidecarHandle, port: u16) {
    let shutdown_url = format!("http://127.0.0.1:{}/api/internal/shutdown", port);

    // Attempt a graceful HTTP shutdown with a 3-second timeout.
    let graceful = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .ok()
        .and_then(|c| c.post(&shutdown_url).send().ok())
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if !graceful {
        let _ = handle.child.kill();
    }

    // Wait so the OS reclaims the port before we exit.
    let _ = handle.child.wait();

    // Windows: closing the Job Object handle terminates any child processes
    // that were assigned to it, covering crash/force-kill scenarios.
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::Foundation::CloseHandle;
        unsafe {
            let _ = CloseHandle(handle.job_handle.0);
        }
    }
}

/// Wait up to `max_wait_ms` ms for the sidecar's health endpoint to respond,
/// polling every `interval_ms` ms. Returns true if the backend became healthy.
pub fn wait_for_ready(port: u16, max_wait_ms: u64, interval_ms: u64) -> bool {
    let health_url = format!("http://127.0.0.1:{}/api/health", port);
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(interval_ms))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };

    let steps = max_wait_ms / interval_ms;
    for _ in 0..steps {
        if client
            .get(&health_url)
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(interval_ms));
    }
    false
}

// ---------------------------------------------------------------------------
// Windows-only: Job Object helpers
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
fn attach_job_object(
    child: &Child,
) -> Result<windows::Win32::Foundation::HANDLE, Box<dyn std::error::Error>> {
    use windows::Win32::{
        Foundation::CloseHandle,
        System::{
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
                SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            },
            Threading::{OpenProcess, PROCESS_ALL_ACCESS},
        },
    };

    let job = unsafe { CreateJobObjectW(None, None)? };

    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    unsafe {
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )?;
    }

    let pid = child.id();
    let process = unsafe { OpenProcess(PROCESS_ALL_ACCESS, false, pid)? };
    unsafe {
        AssignProcessToJobObject(job, process)?;
    }
    unsafe {
        let _ = CloseHandle(process);
    }

    Ok(job)
}
