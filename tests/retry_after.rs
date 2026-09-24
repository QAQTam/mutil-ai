use std::time::Duration;

use mutil_ai::{Error, RetrySource};

#[test]
fn api_error_exposes_retry_after() {
    let error = Error::Api {
        provider: "test",
        status: 429,
        body: "rate limited".to_string(),
        retry_after: Some(Duration::from_secs(12)),
        retry_source: Some(RetrySource::RetryAfterSeconds),
        request_id: Some("req-429".to_string()),
    };

    assert_eq!(error.retry_after(), Some(Duration::from_secs(12)));
    assert_eq!(error.retry_source(), Some(RetrySource::RetryAfterSeconds));
    assert!(error.is_retryable());
}

#[test]
fn client_errors_are_not_retryable() {
    let error = Error::Api {
        provider: "test",
        status: 401,
        body: "unauthorized".to_string(),
        retry_after: None,
        retry_source: None,
        request_id: None,
    };

    assert_eq!(error.retry_after(), None);
    assert!(!error.is_retryable());
}
