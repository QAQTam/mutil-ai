#[cfg(target_os = "linux")]
use std::hint::black_box;
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::{StreamExt, stream};
use mutil_ai::{SseEvent, SseParser};
use sse_stream::SseByteStream;

fn message_stream() -> &'static [u8] {
    b": keepalive\n\
event: update\n\
id: 1\n\
data: hello\n\
data: world\n\
\n\
event: second\n\
id: 2\n\
data: line1\n\
data: line2\n\
\n"
}

fn our_messages(input: &[u8], chunk_size: usize) -> Vec<(Option<String>, String, Option<String>)> {
    let mut parser = SseParser::new();
    let mut events = Vec::new();
    for chunk in input.chunks(chunk_size) {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap());

    events
        .into_iter()
        .filter_map(|event| match event {
            SseEvent::Message(message) => Some((message.event, message.data, message.id)),
            SseEvent::Retry(_) => None,
        })
        .collect()
}

fn sse_stream_messages(
    input: &[u8],
    chunk_size: usize,
) -> Vec<(Option<String>, String, Option<String>)> {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async move {
        let chunks: Vec<Result<Bytes, std::io::Error>> = input
            .chunks(chunk_size)
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect();
        let mut events = SseByteStream::new(stream::iter(chunks));
        let mut output = Vec::new();

        while let Some(event) = events.next().await {
            let event = event.unwrap();
            if let Some(data) = event.data {
                output.push((event.event, data, event.id));
            }
        }
        output
    })
}

#[test]
fn hand_written_matches_sse_stream_for_message_events() {
    for chunk_size in 1..=64 {
        let ours = our_messages(message_stream(), chunk_size);
        let theirs = sse_stream_messages(message_stream(), chunk_size);
        assert_eq!(ours, theirs, "chunk_size={chunk_size}");
    }
}

#[cfg(target_os = "linux")]
fn cpu_time() -> Duration {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("read /proc/self/stat");
    let end_comm = stat.rfind(')').expect("stat has comm");
    let fields: Vec<&str> = stat[end_comm + 2..].split_whitespace().collect();

    // Fields 14 and 15 (utime, stime), zero-based after field 3.
    let utime: u64 = fields[11].parse().expect("utime");
    let stime: u64 = fields[12].parse().expect("stime");
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let hz = if ticks > 0 { ticks as f64 } else { 100.0 };

    Duration::from_secs_f64((utime + stime) as f64 / hz)
}

#[cfg(target_os = "linux")]
fn build_token_chunks(token_count: usize, events_per_chunk: usize) -> Vec<Bytes> {
    const EVENT: &str = "data: {\"content\":\"tok\"}\n\n";
    let mut chunks = Vec::with_capacity(token_count.div_ceil(events_per_chunk));
    let mut current = String::new();

    for index in 0..token_count {
        current.push_str(EVENT);
        if (index + 1) % events_per_chunk == 0 {
            chunks.push(Bytes::from(std::mem::take(&mut current)));
        }
    }

    if !current.is_empty() {
        chunks.push(Bytes::from(current));
    }
    chunks
}

#[cfg(target_os = "linux")]
fn run_ours(chunks: &[Bytes]) -> (Duration, usize) {
    let start_cpu = cpu_time();
    let start_wall = Instant::now();
    let mut parser = SseParser::new();
    let mut event_buffer = Vec::new();
    let mut events = 0;

    for chunk in chunks {
        event_buffer.clear();
        parser.push_into(chunk, &mut event_buffer).unwrap();
        events += event_buffer.len();
    }
    event_buffer.clear();
    parser.finish_into(&mut event_buffer).unwrap();
    events += event_buffer.len();

    let elapsed_cpu = cpu_time().saturating_sub(start_cpu);
    let elapsed_wall = start_wall.elapsed();
    black_box(events);
    black_box(elapsed_wall);
    (elapsed_cpu, events)
}

#[cfg(target_os = "linux")]
fn run_sse_stream(runtime: &tokio::runtime::Runtime, chunks: &[Bytes]) -> (Duration, usize) {
    let start_cpu = cpu_time();
    let start_wall = Instant::now();

    let events = runtime.block_on(async {
        let stream = stream::iter(chunks.iter().cloned().map(Ok::<_, std::io::Error>));
        let mut events = SseByteStream::new(stream);
        let mut count = 0;
        while let Some(event) = events.next().await {
            event.unwrap();
            count += 1;
        }
        count
    });

    let elapsed_cpu = cpu_time().saturating_sub(start_cpu);
    let elapsed_wall = start_wall.elapsed();
    black_box(events);
    black_box(elapsed_wall);
    (elapsed_cpu, events)
}

#[cfg(target_os = "linux")]
fn coefficient_of_variation(samples: &[Duration]) -> f64 {
    let mean = samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample.as_secs_f64() - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean
}

#[cfg(target_os = "linux")]
fn report(label: &str, samples: &[Duration], events: usize) {
    let mean = samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64;
    let min = samples.iter().min().copied().unwrap_or_default();
    let max = samples.iter().max().copied().unwrap_or_default();
    let cv = coefficient_of_variation(samples);
    println!("{label}: events={events}, mean={mean:.6}s, min={min:?}, max={max:?}, cv={cv:.4}");
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "performance comparison; run with --ignored --nocapture"]
fn compare_10k_tokens_per_second_cpu() {
    const TOKENS: usize = 200_000; // 20 seconds at 10,000 tok/s
    const ROUNDS: usize = 10;

    for events_per_chunk in [1, 10, 100] {
        let chunks = build_token_chunks(TOKENS, events_per_chunk);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let mut ours = Vec::with_capacity(ROUNDS);
        let mut theirs = Vec::with_capacity(ROUNDS);
        let mut our_events = 0;
        let mut their_events = 0;

        for _ in 0..ROUNDS {
            let (cpu, events) = run_ours(&chunks);
            ours.push(cpu);
            our_events = events;

            let (cpu, events) = run_sse_stream(&runtime, &chunks);
            theirs.push(cpu);
            their_events = events;
        }

        println!("\n--- {TOKENS} tokens, {events_per_chunk} events/chunk ---");
        report("hand-written", &ours, our_events);
        report("sse-stream", &theirs, their_events);
    }
}
