//! A minimal in-process Redis server for tests: the command subset the
//! store actually issues, executed with real Redis command semantics
//! (single-threaded per command, integer counters, lazy key expiry) over a
//! real TCP socket on loopback, speaking RESP2.
//!
//! This exists so the store's client code runs against the real wire
//! protocol, not a mock of the client library, in environments that have
//! no Redis server binary. It is exposed behind the `test-server`
//! feature — never enabled in production builds — for this crate's own
//! tests and for downstream integration tests (the app's replica-arm
//! runtime tests).

use std::collections::{BTreeMap, HashMap};
use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// One snapshot of the live string keyspace, taken after a mutating
/// command while tracing is on.
pub type KeySnapshot = BTreeMap<String, String>;

/// Handle to the running test server.
pub struct TestRedis {
    addr: std::net::SocketAddr,
    db: Arc<Db>,
    // Connection threads block in accept() for the process's lifetime;
    // they die with the test binary. (Leaking a bounded number of threads
    // per test beats deadlocking Drop on join.)
    _threads: Vec<std::thread::JoinHandle<()>>,
}

struct Db {
    strings: Mutex<HashMap<String, (String, Option<u64>)>>,
    sets: Mutex<HashMap<String, Vec<String>>>,
    zsets: Mutex<HashMap<String, Vec<(f64, String)>>>,
    streams: Mutex<HashMap<String, Vec<String>>>,
    stream_seq: Mutex<u64>,
    trace: Mutex<Option<Vec<KeySnapshot>>>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

impl std::fmt::Debug for TestRedis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TestRedis")
            .field("addr", &self.addr)
            .field("threads", &self._threads.len())
            .finish()
    }
}

impl TestRedis {
    /// Start the server on an ephemeral loopback port.
    ///
    /// # Errors
    ///
    /// `std::io::Error` when the loopback listener cannot be bound.
    pub fn start() -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let db = Arc::new(Db {
            strings: Mutex::new(HashMap::new()),
            sets: Mutex::new(HashMap::new()),
            zsets: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            stream_seq: Mutex::new(0),
            trace: Mutex::new(None),
        });
        let mut threads = Vec::new();
        // A bounded pool of connection threads is plenty for tests.
        for _ in 0..4 {
            let Ok(cloned) = listener.try_clone() else {
                break;
            };
            let db = Arc::clone(&db);
            threads.push(std::thread::spawn(move || {
                loop {
                    let (stream, _) = match cloned.accept() {
                        Ok(pair) => pair,
                        Err(_) => return,
                    };
                    serve_connection(stream, &db);
                }
            }));
        }
        Ok(Self {
            addr,
            db,
            _threads: threads,
        })
    }

    /// The URL the store connects to.
    #[must_use]
    pub fn url(&self) -> String {
        format!("redis://{}", self.addr)
    }

    /// Number of entries in a stream (XLEN equivalent).
    #[must_use]
    pub fn stream_len(&self, key: &str) -> usize {
        self.db
            .streams
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .map_or(0, Vec::len)
    }

    /// Raw string value, honoring expiry like the server does.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<String> {
        self.db
            .strings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key)
            .and_then(|(value, expiry)| {
                (expiry.is_none_or(|at| now_ms() < at)).then(|| value.clone())
            })
    }

    /// Begin recording a snapshot of the live string keyspace after every
    /// mutating command (`SET SETEX DEL EXPIRE INCR DECR INCRBY DECRBY`,
    /// and `EVAL ZADD ZREM` once those commands exist): every state a
    /// concurrent reader of the store could observe.
    pub fn start_trace(&self) {
        *self
            .db
            .trace
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Vec::new());
    }

    /// Take the recorded snapshots and stop tracing.
    #[must_use]
    pub fn take_trace(&self) -> Vec<KeySnapshot> {
        self.db
            .trace
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
            .unwrap_or_default()
    }
}

fn serve_connection(stream: TcpStream, db: &Arc<Db>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(clone) => clone,
        Err(_) => return,
    });
    let mut writer = stream;
    loop {
        let args = match read_command(&mut reader) {
            Some(args) => args,
            None => return,
        };
        let reply = handle(&args, db);
        if writer.write_all(reply.as_bytes()).is_err() {
            return;
        }
    }
}

/// Reads one RESP2 command (array of bulk strings); `None` on EOF or a
/// malformed frame (which ends the connection, as a real server would
/// close it).
fn read_command<R: Read>(reader: &mut R) -> Option<Vec<String>> {
    let mut line = Vec::new();
    read_line(reader, &mut line)?;
    let header = String::from_utf8_lossy(&line).trim().to_owned();
    let count: usize = header.strip_prefix('*')?.parse().ok()?;
    let mut args = Vec::with_capacity(count);
    for _ in 0..count {
        let mut header = Vec::new();
        read_line(reader, &mut header)?;
        let len_header = String::from_utf8_lossy(&header).trim().to_owned();
        let len: usize = len_header.strip_prefix('$')?.parse().ok()?;
        let mut payload = vec![0_u8; len];
        reader.read_exact(&mut payload).ok()?;
        let mut crlf = [0_u8; 2];
        reader.read_exact(&mut crlf).ok()?;
        args.push(String::from_utf8_lossy(&payload).into_owned());
    }
    Some(args)
}

fn read_line<R: Read>(reader: &mut R, buffer: &mut Vec<u8>) -> Option<()> {
    buffer.clear();
    loop {
        let mut byte = [0_u8; 1];
        if reader.read_exact(&mut byte).is_err() {
            return None;
        }
        if byte[0] == b'\n' {
            if buffer.last() == Some(&b'\r') {
                buffer.pop();
            }
            return Some(());
        }
        buffer.push(byte[0]);
    }
}

fn integer(value: i64) -> String {
    format!(":{value}\r\n")
}

fn simple(value: &str) -> String {
    format!("+{value}\r\n")
}

fn error(message: &str) -> String {
    format!("-ERR {message}\r\n")
}

fn bulk(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("${}\r\n{value}\r\n", value.len()),
        None => "$-1\r\n".to_owned(),
    }
}

fn live(map: &HashMap<String, (String, Option<u64>)>, key: &str) -> Option<String> {
    map.get(key)
        .and_then(|(value, expiry)| (expiry.is_none_or(|at| now_ms() < at)).then(|| value.clone()))
}

fn handle_get(db: &Arc<Db>, rest: &[String]) -> String {
    let key = rest.first().map(String::as_str).unwrap_or_default();
    let map = db
        .strings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    bulk(live(&map, key).as_deref())
}

fn handle_set(db: &Arc<Db>, rest: &[String]) -> String {
    let (Some(key), Some(value)) = (rest.first(), rest.get(1)) else {
        return error("SET needs a key and value");
    };
    let mut keep_ttl = false;
    let mut expiry: Option<u64> = None;
    let mut index = 2;
    while let Some(option) = rest.get(index) {
        match option.to_ascii_uppercase().as_str() {
            "PX" => {
                let ms: u64 = rest
                    .get(index + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                expiry = Some(now_ms() + ms);
                index += 2;
            }
            "EX" => {
                let secs: u64 = rest
                    .get(index + 1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                expiry = Some(now_ms() + secs * 1000);
                index += 2;
            }
            "KEEPTTL" => {
                keep_ttl = true;
                index += 1;
            }
            other => return error(&format!("unsupported SET option {other}")),
        }
    }
    let mut map = db
        .strings
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let existing_ttl = map.get(key).and_then(|(_, at)| *at);
    map.insert(
        key.clone(),
        (value.clone(), if keep_ttl { existing_ttl } else { expiry }),
    );
    simple("OK")
}

fn score_bound(raw: &str) -> Option<f64> {
    match raw.to_ascii_lowercase().as_str() {
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        other => other.parse().ok(),
    }
}

fn zadd(db: &Arc<Db>, rest: &[String]) -> String {
    // ZADD key score member
    let (Some(key), Some(score), Some(member)) = (rest.first(), rest.get(1), rest.get(2)) else {
        return error("ZADD needs a key, a score and a member");
    };
    let Ok(score) = score.parse::<f64>() else {
        return error("ZADD score is not a number");
    };
    let mut zsets = db
        .zsets
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let bucket = zsets.entry(key.clone()).or_default();
    let mut added = 0_i64;
    match bucket.iter_mut().find(|(_, existing)| existing == member) {
        Some(entry) => entry.0 = score,
        None => {
            bucket.push((score, member.clone()));
            added = 1;
        }
    }
    bucket.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    integer(added)
}

fn zrem(db: &Arc<Db>, rest: &[String]) -> String {
    // ZREM key member [member ...]
    let (Some(key), members) = (rest.first(), rest.get(1..).unwrap_or(&[])) else {
        return error("ZREM needs a key and a member");
    };
    if members.is_empty() {
        return error("ZREM needs a member");
    }
    let mut zsets = db
        .zsets
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let removed = match zsets.get_mut(key) {
        Some(bucket) => {
            let before = bucket.len();
            bucket.retain(|(_, member)| !members.contains(member));
            before - bucket.len()
        }
        None => 0,
    };
    integer(i64::try_from(removed).unwrap_or(0))
}

fn zrangebyscore(db: &Arc<Db>, rest: &[String]) -> String {
    // ZRANGEBYSCORE key min max [LIMIT offset count], inclusive bounds,
    // -inf/+inf accepted like plain numbers.
    let (Some(key), Some(min), Some(max)) = (rest.first(), rest.get(1), rest.get(2)) else {
        return error("ZRANGEBYSCORE needs a key and a score range");
    };
    let (Some(min), Some(max)) = (score_bound(min), score_bound(max)) else {
        return error("ZRANGEBYSCORE bound is not a number");
    };
    let mut offset = 0_usize;
    let mut count = usize::MAX;
    let mut index = 3;
    while let Some(option) = rest.get(index) {
        if option.eq_ignore_ascii_case("LIMIT") {
            offset = rest
                .get(index + 1)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            count = rest
                .get(index + 2)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            index += 3;
        } else {
            return error(&format!("unsupported ZRANGEBYSCORE option {option}"));
        }
    }
    let zsets = db
        .zsets
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let selected: Vec<&String> = zsets
        .get(key)
        .map(|bucket| {
            bucket
                .iter()
                .filter(|(score, _)| *score >= min && *score <= max)
                .map(|(_, member)| member)
                .skip(offset)
                .take(count)
                .collect()
        })
        .unwrap_or_default();
    let mut payload = format!("*{}\r\n", selected.len());
    for member in &selected {
        payload.push_str(&bulk(Some(member)));
    }
    payload
}

#[expect(
    clippy::too_many_lines,
    reason = "one match over the supported command set; each arm is a line or three"
)]
fn handle(args: &[String], db: &Arc<Db>) -> String {
    let Some(command) = args.first().map(|c| c.to_ascii_uppercase()) else {
        return error("empty command");
    };
    let rest = args.get(1..).unwrap_or(&[]);
    let reply = match command.as_str() {
        "PING" => simple("PONG"),
        "AUTH" | "SELECT" | "CLIENT" | "COMMAND" => simple("OK"),
        "HELLO" => error("unknown command 'HELLO'"),
        "GET" => handle_get(db, rest),
        "SET" => handle_set(db, rest),
        "DEL" => {
            let mut removed = 0_i64;
            let mut map = db
                .strings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for key in rest {
                if map.remove(key).is_some() {
                    removed += 1;
                }
            }
            integer(removed)
        }
        "EXISTS" => {
            let key = rest.first().map(String::as_str).unwrap_or_default();
            let map = db
                .strings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            integer(i64::from(live(&map, key).is_some()))
        }
        "EXPIRE" => {
            let (Some(key), Some(secs)) = (rest.first(), rest.get(1)) else {
                return error("EXPIRE needs a key and seconds");
            };
            let secs: u64 = secs.parse().unwrap_or(0);
            let mut map = db
                .strings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match map.get_mut(key) {
                Some((_, at)) => {
                    *at = Some(now_ms() + secs * 1000);
                    integer(1)
                }
                None => integer(0),
            }
        }
        "SETEX" => {
            // SETEX key seconds value
            let (Some(key), Some(secs), Some(value)) = (rest.first(), rest.get(1), rest.get(2))
            else {
                return error("SETEX needs a key, seconds and value");
            };
            let secs: u64 = secs.parse().unwrap_or(0);
            let mut map = db
                .strings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            map.insert(key.clone(), (value.clone(), Some(now_ms() + secs * 1000)));
            simple("OK")
        }
        "INCR" | "DECR" | "INCRBY" | "DECRBY" => {
            let key = rest.first().map(String::as_str).unwrap_or_default();
            let delta: i64 = match command.as_str() {
                "INCR" => 1,
                "DECR" => -1,
                "INCRBY" | "DECRBY" => {
                    rest.get(1).and_then(|v| v.parse().ok()).unwrap_or(0)
                        * if command == "DECRBY" { -1 } else { 1 }
                }
                _ => 0,
            };
            let mut map = db
                .strings
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if live(&map, key).is_none() {
                map.insert(key.to_owned(), ("0".to_owned(), None));
            }
            let Some(entry) = map.get_mut(key) else {
                return error("key vanished");
            };
            let current: i64 = entry.0.parse().unwrap_or(0);
            let next = current.saturating_add(delta);
            entry.0 = next.to_string();
            integer(next)
        }
        "SADD" => {
            let Some(key) = rest.first() else {
                return error("SADD needs a key");
            };
            let members = rest.get(1..).unwrap_or(&[]);
            let mut sets = db
                .sets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let bucket = sets.entry(key.clone()).or_default();
            let mut added = 0_i64;
            for member in members {
                if !bucket.contains(member) {
                    bucket.push(member.clone());
                    added += 1;
                }
            }
            integer(added)
        }
        "SMEMBERS" => {
            let key = rest.first().map(String::as_str).unwrap_or_default();
            let sets = db
                .sets
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let members = sets.get(key).cloned().unwrap_or_default();
            let mut payload = format!("*{}\r\n", members.len());
            for member in members {
                payload.push_str(&bulk(Some(&member)));
            }
            payload
        }
        "ZADD" => zadd(db, rest),
        "ZREM" => zrem(db, rest),
        "ZRANGEBYSCORE" => zrangebyscore(db, rest),
        "XADD" => {
            // XADD key id field value [field value ...] — the store sends
            // `*` for the id; any id is accepted.
            let Some(key) = rest.first() else {
                return error("XADD needs a key");
            };
            let fields = rest.get(2..).unwrap_or(&[]);
            let mut streams = db
                .streams
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let payload = fields.join("\u{1f}");
            streams.entry(key.clone()).or_default().push(payload);
            let mut seq = db
                .stream_seq
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *seq += 1;
            simple(&format!("0-{seq}"))
        }
        other => error(&format!("unknown command '{other}'")),
    };
    if is_traced_mutation(&command) && !reply.starts_with('-') {
        trace_snapshot(db);
    }
    reply
}

/// The mutating commands whose application is recorded by `start_trace`.
fn is_traced_mutation(command: &str) -> bool {
    matches!(
        command,
        "SET"
            | "SETEX"
            | "DEL"
            | "EXPIRE"
            | "INCR"
            | "DECR"
            | "INCRBY"
            | "DECRBY"
            | "EVAL"
            | "ZADD"
            | "ZREM"
    )
}

/// Records one snapshot of the live string keyspace, when tracing is on.
fn trace_snapshot(db: &Arc<Db>) {
    let tracing_on = db
        .trace
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .is_some();
    if !tracing_on {
        return;
    }
    let snapshot: KeySnapshot = {
        let strings = db
            .strings
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        strings
            .iter()
            .filter(|(_, (_, expiry))| expiry.is_none_or(|at| now_ms() < at))
            .map(|(key, (value, _))| (key.clone(), value.clone()))
            .collect()
    };
    if let Some(trace) = db
        .trace
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_mut()
    {
        trace.push(snapshot);
    }
}
