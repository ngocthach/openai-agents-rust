//! Port of `tests/sandbox/test_unix_local.py::test_client_delete_keeps_workspace_removal_off_the_event_loop`.
//!
//! Python patches `shutil.rmtree` to hold the removal open until an observer task on the event
//! loop answers. Here the runtime is single-threaded, so the observer only runs while `delete`
//! is suspended: it watches the workspace and records whether it ever saw it partly removed. A
//! removal that ran inline on the runtime thread would finish before the observer could look.
#![cfg(all(feature = "sandbox", unix))]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use openai_agents_sdk::sandbox::sandboxes::unix_local::UnixLocalSandboxClient;
use openai_agents_sdk::sandbox::{Manifest, SandboxClient, SandboxSession};

const FILES: usize = 4000;

fn entries(root: &PathBuf) -> Option<usize> {
    std::fs::read_dir(root).ok().map(|dir| dir.count())
}

#[tokio::test(flavor = "current_thread")]
async fn client_delete_keeps_workspace_removal_off_the_runtime_thread() {
    let client = UnixLocalSandboxClient::new();
    let session: Arc<dyn SandboxSession> = client
        .create(Some(Manifest::new()))
        .await
        .expect("create session");
    session.start().await.expect("start session");
    let root = PathBuf::from(session.workspace_root());
    for index in 0..FILES {
        std::fs::write(root.join(format!("payload-{index}.txt")), b"payload").unwrap();
    }
    let total = entries(&root).expect("the workspace exists");
    assert!(total >= FILES);

    let done = Arc::new(AtomicBool::new(false));
    let partial_sightings = Arc::new(AtomicUsize::new(0));
    let observer = tokio::spawn({
        let (root, done, partial) = (root.clone(), done.clone(), partial_sightings.clone());
        async move {
            while !done.load(Ordering::SeqCst) {
                if let Some(count) = entries(&root)
                    && count < total
                {
                    partial.fetch_add(1, Ordering::SeqCst);
                }
                tokio::task::yield_now().await;
            }
        }
    });

    client.delete(session).await.expect("delete");
    done.store(true, Ordering::SeqCst);
    observer.await.expect("observer");

    assert!(
        partial_sightings.load(Ordering::SeqCst) > 0,
        "the runtime never ran while the workspace was being removed"
    );
    // The removal still targets the session's own workspace root.
    assert!(!root.exists());
}
