use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::watch;

/// A cloneable cooperative cancellation handle.
///
/// Clones share the same cancellation state. Calling [`Self::cancel`] wakes
/// every task currently waiting on [`Self::cancelled`].
#[derive(Clone)]
pub struct CancellationToken {
    state: Arc<CancelState>,
    receiver: watch::Receiver<bool>,
}

struct CancelState {
    cancelled: AtomicBool,
    sender: watch::Sender<bool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        let (sender, receiver) = watch::channel(false);
        Self {
            state: Arc::new(CancelState {
                cancelled: AtomicBool::new(false),
                sender,
            }),
            receiver,
        }
    }

    /// Return whether cancellation has already been requested.
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    /// Request cancellation. This operation is idempotent.
    pub fn cancel(&self) {
        if !self.state.cancelled.swap(true, Ordering::AcqRel) {
            let _ = self.state.sender.send(true);
        }
    }

    /// Wait until cancellation is requested.
    ///
    /// This returns immediately when the token is already cancelled.
    pub async fn cancelled(&self) {
        let mut receiver = self.receiver.clone();
        if *receiver.borrow() {
            return;
        }
        let _ = receiver.changed().await;
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CancellationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CancellationToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn clones_share_cancellation_and_wake_waiters() {
        let token = CancellationToken::new();
        let waiter = token.clone();
        let task = tokio::spawn(async move {
            waiter.cancelled().await;
            waiter.is_cancelled()
        });

        token.cancel();
        assert!(task.await.unwrap());
        assert!(token.is_cancelled());
    }
}
