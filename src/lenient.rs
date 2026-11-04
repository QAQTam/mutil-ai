//! Lenient JSON value parsing for provider tool-call arguments.
//!
//! Wire-hardening only: models behind compat gateways sometimes emit *relaxed
//! JSON* in function arguments — unquoted scalar values (most commonly
//! glob-like strings such as `*.rs`), single-quoted strings, and trailing
//! commas. Standard `serde_json` rejects all of it, so the argument historically
//! reached the tool layer as a raw string and the typed tool rejected it
//! ("invalid type: string, expected struct …").
//!
//! The repair is deliberately conservative:
//! - strict parse first — a valid payload is never touched;
//! - the rewriter only quotes unquoted scalars, rewrites single-quoted strings,
//!   and strips trailing commas — keys are expected to stay quoted (observed
//!   provider behavior quotes keys consistently and slips only on values);
//! - if the repaired text still fails to parse, the caller gets `None` (or the
//!   original string passthrough) exactly like before — fail-open to the
//!   previous behavior, never to invented data.

use serde_json::Value;

/// Parse a JSON payload; on strict failure attempt a conservative repair.
///
/// Returns `None` only when neither the original nor the repaired text parses.
pub fn parse(value: &str) -> Option<Value> {
    if let Ok(parsed) = serde_json::from_str(value) {
        return Some(parsed);
    }
    let repaired = repair(value)?;
    serde_json::from_str(&repaired).ok()
}

/// SDK-facing shim matching the historical `parse_json_or_string` contract:
/// unparseable input is returned as `Value::String` (never an error).
pub fn value_or_string(value: &str) -> Value {
    parse(value).unwrap_or_else(|| Value::String(value.to_owned()))
}

/// Attempt to repair *relaxed JSON* into parseable JSON.
///
/// Returns `None` when no repair rule applies or the repaired text is still
/// not valid JSON — callers must then keep the original string.
pub fn repair(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let repaired = rewrite(trimmed.as_bytes());
    if repaired == trimmed {
        return None;
    }
    serde_json::from_str::<Value>(&repaired).ok().map(|_| repaired)
}

/// Single-pass rewriter over the raw bytes of a JSON document.
///
/// - double-quoted strings: copied verbatim (escapes included);
/// - single-quoted strings: rewritten to double-quoted, inner `"` escaped;
/// - `,` directly followed by `}` / `]` (whitespace allowed): dropped;
/// - after `:`: an unquoted scalar value is quoted and escaped. Bareword
///   scanning is brace-depth aware so glob patterns like `*.{ts,js,tsx}`
///   survive as a single value.
fn rewrite(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + 16);
    let mut i = 0usize;
    // Byte offset in `out` of a comma that may turn out to be trailing.
    let mut pending_comma: Option<usize> = None;

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'"' => {
                pending_comma = None;
                out.push('"');
                i = copy_string(bytes, i + 1, &mut out, b'"');
            }
            b'\'' => {
                pending_comma = None;
                out.push('"');
                i = rewrite_single_quoted(bytes, i + 1, &mut out);
            }
            b',' => {
                pending_comma = Some(out.len());
                out.push(',');
                i += 1;
            }
            b'{' | b'[' => {
                pending_comma = None;
                out.push(b as char);
                i += 1;
            }
            b'}' | b']' => {
                if let Some(cut) = pending_comma.take() {
                    out.truncate(cut);
                }
                out.push(b as char);
                i += 1;
            }
            b':' => {
                pending_comma = None;
                out.push(':');
                i += 1;
                i = quote_bareword_value(bytes, i, &mut out);
            }
            c if (c as char).is_ascii_whitespace() => {
                // Whitespace does not resolve a pending comma.
                out.push(c as char);
                i += 1;
            }
            c => {
                pending_comma = None;
                out.push(c as char);
                i += 1;
            }
        }
    }
    out
}

/// Copy a double-quoted string body (starting after the opening quote) until
/// the closing quote; `out` receives the raw bytes including the terminator.
fn copy_string(bytes: &[u8], mut i: usize, out: &mut String, quote: u8) -> usize {
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' && i + 1 < bytes.len() {
            out.push(c as char);
            out.push(bytes[i + 1] as char);
            i += 2;
            continue;
        }
        out.push(c as char);
        i += 1;
        if c == quote {
            break;
        }
    }
    i
}

/// Rewrite a single-quoted string body to a double-quoted one.
fn rewrite_single_quoted(bytes: &[u8], mut i: usize, out: &mut String) -> usize {
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' && i + 1 < bytes.len() {
            out.push(c as char);
            out.push(bytes[i + 1] as char);
            i += 2;
            continue;
        }
        if c == b'\'' {
            out.push('"');
            return i + 1;
        }
        if c == b'"' {
            out.push('\\');
        }
        out.push(c as char);
        i += 1;
    }
    i
}

/// After a `:`: if the next non-whitespace byte starts a bareword scalar
/// (not a quote, structure, number, or JSON literal), quote and escape it.
/// Returns the next scan position.
fn quote_bareword_value(bytes: &[u8], start: usize, out: &mut String) -> usize {
    let mut i = start;
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
        out.push(bytes[i] as char);
        i += 1;
    }
    let Some(&first) = bytes.get(i) else {
        return i;
    };
    let well_formed = matches!(
        first,
        b'"' | b'\'' | b'{' | b'[' | b'-' | b'0'..=b'9'
    ) || bytes[i..].starts_with(b"true")
        || bytes[i..].starts_with(b"false")
        || bytes[i..].starts_with(b"null");
    if well_formed {
        return i;
    }
    let (raw, next) = bareword_end(bytes, i);
    let literal = String::from_utf8_lossy(raw);
    match serde_json::from_str::<Value>(&literal) {
        // A bareword that is itself a valid non-string JSON literal
        // (number/bool/null) stays raw; everything else gets quoted.
        Ok(v) if !v.is_string() => out.push_str(&literal),
        _ => {
            out.push('"');
            for c in literal.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
    }
    next
}

/// A bareword value ends at the first structural char at brace-depth 0
/// (`,` `}` `]`); braces/brackets inside the bareword raise depth so glob
/// patterns like `*.{ts,js,tsx}` survive as one value.
fn bareword_end(bytes: &[u8], start: usize) -> (&[u8], usize) {
    let mut i = start;
    let mut depth: usize = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            b',' if depth == 0 => break,
            _ => {}
        }
        i += 1;
    }
    let end = i;
    let mut trimmed = end;
    while trimmed > start && (bytes[trimmed - 1] as char).is_ascii_whitespace() {
        trimmed -= 1;
    }
    (&bytes[start..trimmed], end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Regression vectors lifted verbatim from qaqh-backend daemon log
    /// (`Responses 收口流条目参数损坏`) — glm-5.3-flash relaxed-JSON args.
    #[test]
    fn daemon_log_glob_vectors_repair() {
        let cases = [
            (
                r#"{"glob":*.rs,"max_results":40,"pattern":"v1beta|gemini"}"#,
                json!({"glob": "*.rs", "max_results": 40, "pattern": "v1beta|gemini"}),
            ),
            (
                r#"{"glob":*.{ts,js,tsx},"max_results":20,"pattern":"setWindowOpenHandler|will-navi"}"#,
                json!({"glob": "*.{ts,js,tsx}", "max_results": 20,
                       "pattern": "setWindowOpenHandler|will-navi"}),
            ),
            (
                r#"{"glob":!node_modules,"max_results":40,"paths":["/home/u/p"]}"#,
                json!({"glob": "!node_modules", "max_results": 40, "paths": ["/home/u/p"]}),
            ),
            (
                r#"{"glob":codex-rs/core/src/**,"max_results":40,"pattern":"enum .*Input|TurnInputR"}"#,
                json!({"glob": "codex-rs/core/src/**", "max_results": 40,
                       "pattern": "enum .*Input|TurnInputR"}),
            ),
        ];
        for (raw, expected) in cases {
            assert_eq!(parse(raw), Some(expected), "case: {raw}");
        }
    }

    #[test]
    fn valid_json_is_untouched() {
        let raw = r#"{"glob":"*.rs","n":-1.5,"ok":true,"nil":null,"arr":[1,2]}"#;
        assert_eq!(parse(raw), Some(serde_json::from_str(raw).unwrap()));
        assert_eq!(repair(raw), None, "no repair rule should apply");
    }

    #[test]
    fn single_quotes_and_trailing_commas() {
        assert_eq!(parse("{'a':1,'b':'x',}"), Some(json!({"a": 1, "b": "x"})));
        assert_eq!(
            parse(r#"{"list":[1,2,],"obj":{"k":"v",}}"#),
            Some(json!({"list": [1, 2], "obj": {"k": "v"}}))
        );
        assert_eq!(
            parse(r#"{"x":1 , "y":2}"#),
            Some(json!({"x": 1, "y": 2})),
            "comma followed by whitespace then key must survive"
        );
    }

    #[test]
    fn garbage_still_fails_closed() {
        assert_eq!(parse("not json at all"), None);
        assert_eq!(parse(""), None);
        assert_eq!(
            value_or_string("plain text"),
            Value::String("plain text".into())
        );
    }

    #[test]
    fn bareword_with_quotes_inside_is_escaped() {
        assert_eq!(
            parse(r#"{"pat":he said "hi"}"#),
            Some(json!({"pat": "he said \"hi\""}))
        );
    }

    #[test]
    fn literal_barewords_stay_raw() {
        // A bareword that parses as a non-string JSON literal is preserved.
        assert_eq!(
            parse(r#"{"a":NaN,"b":Infinity}"#),
            Some(json!({"a": "NaN", "b": "Infinity"}))
        );
    }
}
