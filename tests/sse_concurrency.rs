#![cfg(target_os = "linux")]

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_executor::block_on;
use futures_util::{StreamExt, stream};
use mutil_ai::SseParser;
use sse_stream::SseByteStream;

const TOKENS_PER_CLIENT: usize = 300_000; // 30 seconds at 10,000 tok/s
const ROUNDS: usize = 3;
const EVENTS_PER_CHUNK: [usize; 2] = [1, 10];

fn client_counts() -> Vec<usize> {
    let logical = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let mut counts = vec![1];

    for candidate in [4, 8, 12] {
        if candidate <= logical && !counts.contains(&candidate) {
            counts.push(candidate);
        }
    }

    counts
}

#[derive(Debug, Clone)]
struct RunResult {
    worker_cpu: Duration,
    process_cpu: Duration,
    wall: Duration,
    per_client_cpu: Vec<Duration>,
    events: usize,
}

fn current_tid() -> i32 {
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
}

fn cpu_time_from_stat(path: &str) -> Duration {
    let stat = std::fs::read_to_string(path).expect("read proc stat");
    let end_comm = stat.rfind(')').expect("stat has comm");
    let fields: Vec<&str> = stat[end_comm + 2..].split_whitespace().collect();

    let utime: u64 = fields[11].parse().expect("utime");
    let stime: u64 = fields[12].parse().expect("stime");
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let hz = if ticks > 0 { ticks as f64 } else { 100.0 };

    Duration::from_secs_f64((utime + stime) as f64 / hz)
}

fn process_cpu_time() -> Duration {
    cpu_time_from_stat("/proc/self/stat")
}

fn thread_cpu_time(tid: i32) -> Duration {
    cpu_time_from_stat(&format!("/proc/self/task/{tid}/stat"))
}

fn build_token_chunks(token_count: usize, events_per_chunk: usize) -> Arc<Vec<Bytes>> {
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

    Arc::new(chunks)
}

fn run_ours_concurrent(clients: usize, chunks: &Arc<Vec<Bytes>>) -> RunResult {
    let barrier = Arc::new(Barrier::new(clients));
    let start_process_cpu = process_cpu_time();
    let start_wall = Instant::now();

    let handles: Vec<_> = (0..clients)
        .map(|_| {
            let chunks = Arc::clone(chunks);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let tid = current_tid();
                let start_cpu = thread_cpu_time(tid);
                let mut parser = SseParser::new();
                let mut event_buffer = Vec::new();
                let mut events = 0;

                for chunk in chunks.iter() {
                    event_buffer.clear();
                    parser.push_into(chunk, &mut event_buffer).unwrap();
                    events += event_buffer.len();
                }
                event_buffer.clear();
                parser.finish_into(&mut event_buffer).unwrap();
                events += event_buffer.len();

                let cpu = thread_cpu_time(tid).saturating_sub(start_cpu);
                black_box(events);
                (events, cpu)
            })
        })
        .collect();

    let mut per_client_cpu = Vec::with_capacity(clients);
    let mut events = 0;
    for handle in handles {
        let (client_events, cpu) = handle.join().expect("client thread");
        events += client_events;
        per_client_cpu.push(cpu);
    }

    let worker_cpu = per_client_cpu.iter().copied().sum();
    RunResult {
        worker_cpu,
        process_cpu: process_cpu_time().saturating_sub(start_process_cpu),
        wall: start_wall.elapsed(),
        per_client_cpu,
        events,
    }
}

fn run_sse_stream_concurrent(clients: usize, chunks: &Arc<Vec<Bytes>>) -> RunResult {
    let barrier = Arc::new(Barrier::new(clients));
    let start_process_cpu = process_cpu_time();
    let start_wall = Instant::now();

    let handles: Vec<_> = (0..clients)
        .map(|_| {
            let chunks = Arc::clone(chunks);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let tid = current_tid();
                let start_cpu = thread_cpu_time(tid);

                let events = block_on(async move {
                    let stream =
                        stream::iter(chunks.iter().cloned().map(Ok::<Bytes, std::io::Error>));
                    let mut events = SseByteStream::new(stream);
                    let mut count = 0;
                    while let Some(event) = events.next().await {
                        event.unwrap();
                        count += 1;
                    }
                    count
                });

                let cpu = thread_cpu_time(tid).saturating_sub(start_cpu);
                black_box(events);
                (events, cpu)
            })
        })
        .collect();

    let mut per_client_cpu = Vec::with_capacity(clients);
    let mut events = 0;
    for handle in handles {
        let (client_events, cpu) = handle.join().expect("client thread");
        events += client_events;
        per_client_cpu.push(cpu);
    }

    let worker_cpu = per_client_cpu.iter().copied().sum();
    RunResult {
        worker_cpu,
        process_cpu: process_cpu_time().saturating_sub(start_process_cpu),
        wall: start_wall.elapsed(),
        per_client_cpu,
        events,
    }
}

fn mean_duration(samples: &[Duration]) -> Duration {
    let mean = samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64;
    Duration::from_secs_f64(mean)
}

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

fn summarize(runs: &[RunResult]) -> (Duration, Duration, Duration, f64, f64) {
    let worker_cpu: Vec<Duration> = runs.iter().map(|run| run.worker_cpu).collect();
    let process_cpu: Vec<Duration> = runs.iter().map(|run| run.process_cpu).collect();
    let wall: Vec<Duration> = runs.iter().map(|run| run.wall).collect();
    let client_cv = runs
        .iter()
        .map(|run| coefficient_of_variation(&run.per_client_cpu))
        .sum::<f64>()
        / runs.len() as f64;

    (
        mean_duration(&worker_cpu),
        mean_duration(&process_cpu),
        mean_duration(&wall),
        coefficient_of_variation(&worker_cpu),
        client_cv,
    )
}

fn report(label: &str, clients: usize, tokens: usize, runs: &[RunResult]) {
    let (worker_cpu, process_cpu, wall, round_cv, client_cv) = summarize(runs);
    let cpu_per_token_us = worker_cpu.as_secs_f64() * 1_000_000.0 / tokens as f64;
    let target_wall = Duration::from_secs_f64(tokens as f64 / 10_000.0);
    let headroom = target_wall.as_secs_f64() / worker_cpu.as_secs_f64();

    println!(
        "{label:>12} clients={clients:>2}: tokens={tokens}, events={}, \
worker_cpu={worker_cpu:?}, process_cpu={process_cpu:?}, wall={wall:?}, \
cpu/token={cpu_per_token_us:.3}us, round_cv={round_cv:.4}, client_cv={client_cv:.4}, \
10k-tok/s-headroom={headroom:.1}x",
        runs[0].events
    );
}

#[test]
#[ignore = "concurrency performance comparison; run with --ignored --nocapture"]
fn compare_multi_client_10k_tokens_per_second() {
    for events_per_chunk in EVENTS_PER_CHUNK {
        println!("\n=== events_per_chunk={events_per_chunk} ===");
        for clients in client_counts() {
            let chunks = build_token_chunks(TOKENS_PER_CLIENT, events_per_chunk);
            let tokens = clients * TOKENS_PER_CLIENT;

            let mut ours = Vec::with_capacity(ROUNDS);
            let mut theirs = Vec::with_capacity(ROUNDS);

            for _ in 0..ROUNDS {
                ours.push(run_ours_concurrent(clients, &chunks));
                theirs.push(run_sse_stream_concurrent(clients, &chunks));
            }

            report("hand-written", clients, tokens, &ours);
            report("sse-stream", clients, tokens, &theirs);
        }
    }
}
