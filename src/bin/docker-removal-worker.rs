//! The trusted host worker of `DockerRemovalService`.
//!
//! Run only by the service, as root, on the Docker daemon host. It enters the mount namespace
//! of a paused container; see `openai_agents_sdk::sandbox::sandboxes::docker_removal`.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(
        openai_agents_sdk::sandbox::sandboxes::docker_removal_worker::worker_main(&args),
    );
}
