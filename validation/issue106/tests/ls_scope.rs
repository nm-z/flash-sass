use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("gpuq-issue106-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).expect("create isolated fixture directory");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Worker(Child);

impl Worker {
    fn start(
        cli: &Path,
        db: Option<&Path>,
        state_home: &Path,
        queue: &Path,
        lock: &Path,
        label: &str,
        seconds: &str,
    ) -> Self {
        let child = cli_command(cli, db, state_home, queue, lock)
            .args(["run", "--name", label, "--exclude", "all", "--"])
            .arg("/bin/sleep")
            .arg(seconds)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start gpuq fixture job");
        Self(child)
    }

    fn wait(&mut self) -> ExitStatus {
        self.0.wait().expect("wait for gpuq fixture job")
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

fn cli_command(
    cli: &Path,
    db: Option<&Path>,
    state_home: &Path,
    queue: &Path,
    lock: &Path,
) -> Command {
    let mut command = Command::new("ruby");
    command.arg(cli);
    if let Some(db) = db {
        command.arg("--db").arg(db);
    }
    command
        .env_remove("GPUQ_DB")
        .env("XDG_STATE_HOME", state_home)
        .env("GPUQ_QUEUE", queue)
        .env("GPUQ_LOCK", lock)
}

fn ls(cli: &Path, db: Option<&Path>, state_home: &Path, queue: &Path, lock: &Path) -> String {
    let output = cli_command(cli, db, state_home, queue, lock)
        .args(["ls", "10"])
        .output()
        .expect("run gpuq ls");
    assert!(
        output.status.success(),
        "gpuq ls failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("gpuq ls returned UTF-8 JSON")
}

fn wait_for_status(
    cli: &Path,
    db: Option<&Path>,
    state_home: &Path,
    queue: &Path,
    lock: &Path,
    label: &str,
    status: &str,
) -> String {
    let label_field = format!("\"label\": \"{label}\"");
    let status_field = format!("\"status\": \"{status}\"");
    let mut last = String::new();
    for _ in 0..240 {
        last = ls(cli, db, state_home, queue, lock);
        if last.contains(&label_field) && last.contains(&status_field) {
            return last;
        }
        thread::sleep(Duration::from_millis(25));
    }
    panic!("job {label} did not reach {status}; last listing: {last}");
}

fn wait_for_ticket(queue: &Path) -> usize {
    for _ in 0..240 {
        let count = fs::read_dir(queue)
            .expect("read isolated shared FIFO")
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "job"))
            .count();
        if count > 0 {
            return count;
        }
        thread::sleep(Duration::from_millis(25));
    }
    panic!("queued job did not create a shared FIFO ticket");
}

#[test]
fn ls_scope_is_per_database_while_queue_tickets_are_shared() {
    // The existing Ruby CLI runs against only temporary databases, queue, and lock paths.
    let scratch = Scratch::new();
    let cli = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/gpuq/gpuq.rb");
    let queue = scratch.path("queue");
    let lock = scratch.path("lock");
    let state_home = scratch.path("state");
    let alternate_db = scratch.path("issue9-history.sqlite3");

    let mut alternate = Worker::start(
        &cli,
        Some(&alternate_db),
        &state_home,
        &queue,
        &lock,
        "alternate-running",
        "6",
    );
    let alternate_listing = wait_for_status(
        &cli,
        Some(&alternate_db),
        &state_home,
        &queue,
        &lock,
        "alternate-running",
        "running",
    );
    assert!(alternate_listing.contains("alternate-running"));

    let mut default = Worker::start(
        &cli,
        None,
        &state_home,
        &queue,
        &lock,
        "default-waiting",
        "0.1",
    );
    let default_listing = wait_for_status(
        &cli,
        None,
        &state_home,
        &queue,
        &lock,
        "default-waiting",
        "queued",
    );
    assert!(!default_listing.contains("alternate-running"));

    assert!(wait_for_ticket(&queue) > 0);

    let alternate_listing = ls(&cli, Some(&alternate_db), &state_home, &queue, &lock);
    assert!(alternate_listing.contains("alternate-running"));
    assert!(!alternate_listing.contains("default-waiting"));

    assert!(default.wait().success(), "default fixture job failed");
    assert!(alternate.wait().success(), "alternate fixture job failed");
}
