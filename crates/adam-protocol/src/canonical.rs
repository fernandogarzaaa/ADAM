//! CP/1 canonical form: the single byte-exact encoding a document hashes over.
//!
//! `serde_json::to_string` is almost canonical, but it preserves whatever key
//! order a value carries and CP/1 requires members sorted by UTF-8 code unit.
//! This module imposes that ordering, reusing serde_json only for scalar
//! rendering — where its escaping rules (escape `"`, `\`, and `U+0000`–`U+001F`,
//! using `\b \f \n \r \t` where they exist; leave non-ASCII literal) already
//! match `protocol/cp1/SPEC.md` section 2 exactly.
//!
//! Floats are rejected rather than rendered. CP/1 puts no floating point on the
//! wire, and ADAM's internal representations are full of `f32` confidences and
//! fitness scores — so this rejection is the boundary that forces every one of
//! them through an explicit basis-point conversion instead of silently emitting
//! bytes that hash differently in EVE's TypeScript binding.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Why a value could not be rendered in CP/1 canonical form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalError {
    /// A non-integer number was present. See the module docs.
    FloatNotPermitted { path: String },
    /// A `null` was present. CP/1 omits absent keys instead.
    NullNotPermitted { path: String },
}

impl std::fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalError::FloatNotPermitted { path } => write!(
                f,
                "CP/1 canonical form permits integers only; found a floating-point number at {path} (ratios cross the wire as basis points)"
            ),
            CanonicalError::NullNotPermitted { path } => write!(
                f,
                "CP/1 canonical form omits absent values rather than writing null; found null at {path}"
            ),
        }
    }
}

impl std::error::Error for CanonicalError {}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Render `value` in CP/1 canonical form.
///
/// ```
/// use adam_protocol::canonical::to_canonical;
/// let value = serde_json::json!({ "b": 1, "a": [2, { "d": 3, "c": 4 }] });
/// assert_eq!(to_canonical(&value).unwrap(), r#"{"a":[2,{"c":4,"d":3}],"b":1}"#);
/// ```
pub fn to_canonical(value: &Value) -> Result<String, CanonicalError> {
    let mut out = String::new();
    write_value(value, "$", &mut out)?;
    Ok(out)
}

fn write_value(value: &Value, path: &str, out: &mut String) -> Result<(), CanonicalError> {
    match value {
        Value::Null => Err(CanonicalError::NullNotPermitted {
            path: path.to_string(),
        }),
        Value::Bool(b) => {
            out.push_str(if *b { "true" } else { "false" });
            Ok(())
        }
        Value::Number(n) => {
            if n.is_f64() {
                return Err(CanonicalError::FloatNotPermitted {
                    path: path.to_string(),
                });
            }
            out.push_str(&n.to_string());
            Ok(())
        }
        Value::String(s) => {
            out.push_str(&Value::String(s.clone()).to_string());
            Ok(())
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, &format!("{path}[{i}]"), out)?;
            }
            out.push(']');
            Ok(())
        }
        Value::Object(map) => {
            // `serde_json::Map` is a BTreeMap unless the `preserve_order`
            // feature is enabled, in which case it is insertion-ordered.
            // Sorting explicitly is correct either way, rather than depending
            // on a feature flag a transitive dependency could switch on.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_value(&map[key], &format!("{path}.{key}"), out)?;
            }
            out.push('}');
            Ok(())
        }
    }
}

/// SHA-256 over the canonical form of `document` with
/// `provenance.content_hash` removed — a document cannot commit to its own
/// hash.
///
/// Everything else is inside the hash on purpose, including evidence and
/// `derived_from`: the provenance chain is only unforgeable if substituting
/// the evidence changes the hash (SPEC.md section 4.1).
pub fn content_hash(document: &Value) -> Result<String, CanonicalError> {
    let mut unsealed = document.clone();
    if let Some(provenance) = unsealed
        .get_mut("provenance")
        .and_then(Value::as_object_mut)
    {
        provenance.remove("content_hash");
    }
    Ok(sha256_hex(to_canonical(&unsealed)?.as_bytes()))
}

/// Compute and write `provenance.content_hash` into `document`, returning it.
///
/// Sealing is idempotent: any previously recorded hash is stripped before the
/// new one is computed, so sealing twice yields the same value.
pub fn seal(document: &mut Value) -> Result<String, CanonicalError> {
    let hash = content_hash(document)?;
    if let Some(provenance) = document
        .get_mut("provenance")
        .and_then(Value::as_object_mut)
    {
        provenance.insert("content_hash".to_string(), Value::String(hash.clone()));
    }
    Ok(hash)
}

/// Whether `document` carries a `provenance.content_hash` equal to its true hash.
pub fn verify_seal(document: &Value) -> Result<bool, CanonicalError> {
    match document
        .get("provenance")
        .and_then(|p| p.get("content_hash"))
        .and_then(Value::as_str)
    {
        Some(recorded) => Ok(recorded == content_hash(document)?),
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_keys_are_sorted_at_every_depth() {
        let value = json!({ "z": 1, "a": { "y": 2, "b": 3 } });
        assert_eq!(
            to_canonical(&value).unwrap(),
            r#"{"a":{"b":3,"y":2},"z":1}"#
        );
    }

    #[test]
    fn array_order_is_preserved() {
        assert_eq!(
            to_canonical(&json!({ "xs": [3, 1, 2] })).unwrap(),
            r#"{"xs":[3,1,2]}"#
        );
    }

    #[test]
    fn strings_use_short_escapes_and_keep_non_ascii_literal() {
        let value = json!({ "s": "a\"b\\c\nd\te—f" });
        assert_eq!(
            to_canonical(&value).unwrap(),
            "{\"s\":\"a\\\"b\\\\c\\nd\\te—f\"}"
        );
    }

    #[test]
    fn floats_are_rejected_with_their_location() {
        // The boundary that forces ADAM's f32 confidences through an explicit
        // basis-point conversion instead of onto the wire.
        assert_eq!(
            to_canonical(&json!({ "belief": { "confidence": 0.82f64 } })),
            Err(CanonicalError::FloatNotPermitted {
                path: "$.belief.confidence".to_string()
            })
        );
    }

    #[test]
    fn nulls_are_rejected_with_their_location() {
        assert_eq!(
            to_canonical(&json!({ "xs": [1, null] })),
            Err(CanonicalError::NullNotPermitted {
                path: "$.xs[1]".to_string()
            })
        );
    }

    #[test]
    fn integers_render_without_fraction_or_exponent() {
        assert_eq!(
            to_canonical(&json!({ "big": 4294967295u32, "neg": -10000, "zero": 0 })).unwrap(),
            r#"{"big":4294967295,"neg":-10000,"zero":0}"#
        );
    }

    #[test]
    fn seal_then_verify_round_trips_and_detects_tampering() {
        let mut document = json!({
            "statement": "tests catch regressions",
            "provenance": { "authored_by": "adam", "evidence": [] }
        });
        assert_eq!(seal(&mut document).unwrap().len(), 64);
        assert!(verify_seal(&document).unwrap());

        document["statement"] = json!("tests do not catch regressions");
        assert!(!verify_seal(&document).unwrap());
    }

    #[test]
    fn seal_is_idempotent() {
        let mut document = json!({
            "statement": "x",
            "provenance": { "content_hash": "0".repeat(64) }
        });
        let first = seal(&mut document).unwrap();
        assert_eq!(seal(&mut document).unwrap(), first);
    }

    #[test]
    fn evidence_is_inside_the_hash_so_it_cannot_be_substituted() {
        let base = json!({ "s": "x", "provenance": { "evidence": [] } });
        let with_evidence = json!({ "s": "x", "provenance": { "evidence": ["a log line"] } });
        assert_ne!(
            content_hash(&base).unwrap(),
            content_hash(&with_evidence).unwrap()
        );
    }

    #[test]
    fn a_document_without_provenance_never_verifies() {
        assert!(!verify_seal(&json!({ "statement": "x" })).unwrap());
    }
}
