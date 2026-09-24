use std::sync::Arc;
use std::thread;

use mutil_ai::SseParser;

fn run_sessions(sessions: usize, workers: usize, chunks: &Arc<Vec<Vec<u8>>>) -> usize {
    let base = sessions / workers;
    let remainder = sessions % workers;
    let mut next_start = 0;
    let ranges: Vec<(usize, usize)> = (0..workers)
        .map(|worker| {
            let count = base + usize::from(worker < remainder);
            let range = (next_start, next_start + count);
            next_start += count;
            range
        })
        .collect();

    let handles: Vec<_> = ranges
        .into_iter()
        .map(|(start, end)| {
            let chunks = Arc::clone(chunks);
            thread::spawn(move || {
                let mut parsers: Vec<SseParser> = (start..end).map(|_| SseParser::new()).collect();
                let mut buffer = Vec::new();
                let mut events = 0;

                for chunk in chunks.iter() {
                    for parser in &mut parsers {
                        buffer.clear();
                        parser
                            .push_into(chunk, &mut buffer)
                            .expect("SSE parse should not fail");
                        events += buffer.len();
                    }
                }
                events
            })
        })
        .collect();

    handles
        .into_iter()
        .map(|handle| handle.join().expect("session worker"))
        .sum()
}

fn session_chunks(events_per_session: usize) -> Arc<Vec<Vec<u8>>> {
    Arc::new(
        (0..events_per_session)
            .map(|_| b"data: {\"content\":\"tok\"}\n\n".to_vec())
            .collect(),
    )
}

#[test]
fn ten_thousand_sessions_smoke_test() {
    const SESSIONS: usize = 10_000;
    const EVENTS_PER_SESSION: usize = 4;
    let chunks = session_chunks(EVENTS_PER_SESSION);
    let events = run_sessions(SESSIONS, 12, &chunks);

    assert_eq!(events, SESSIONS * EVENTS_PER_SESSION);
}

#[test]
#[ignore = "10k session stress test; run with --ignored --nocapture"]
fn ten_thousand_sessions_parse_without_loss() {
    const SESSIONS: usize = 10_000;
    const EVENTS_PER_SESSION: usize = 100;
    let chunks = session_chunks(EVENTS_PER_SESSION);
    let events = run_sessions(SESSIONS, 12, &chunks);

    assert_eq!(events, SESSIONS * EVENTS_PER_SESSION);
}
