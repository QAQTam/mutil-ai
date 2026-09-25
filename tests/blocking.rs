#![cfg(feature = "blocking")]

use mutil_ai::{BlockingAdapter, Error, OpenAIChat};

#[test]
fn blocking_adapter_rejects_async_context() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let error = BlockingAdapter::new(OpenAIChat::new("blocking-test"))
            .err()
            .unwrap();
        assert!(matches!(error, Error::InvalidRequest(message) if message.contains("async")));
    });
}
