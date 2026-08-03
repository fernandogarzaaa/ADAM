//! CP/1 transport: the signed envelope and the line-delimited JSON protocol.
//!
//! The envelope carries its payload as a **string**, not a nested object, so
//! the bytes that were hashed are exactly the bytes transmitted. Nesting the
//! document would let the receiver's JSON writer re-render it and invalidate a
//! hash that was correct when computed.
//!
//! HMAC is implemented here over `sha2` rather than pulled in as a dependency:
//! the construction is twelve lines, and a vendored protocol crate that a
//! future consumer may copy should stay dependency-light.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::canonical::{self, sha256_hex, CanonicalError};

pub const ENVELOPE_SCHEMA: &str = "cp1_signed_envelope";

const BLOCK_SIZE: usize = 64;

/// HMAC-SHA256, hex-encoded (RFC 2104).
pub fn hmac_sha256_hex(key: &[u8], message: &[u8]) -> String {
    // Keys longer than the block size are replaced by their own hash; shorter
    // ones are zero-padded.
    let mut block = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let digest = Sha256::digest(key);
        block[..digest.len()].copy_from_slice(&digest);
    } else {
        block[..key.len()].copy_from_slice(key);
    }

    let mut inner_key = [0x36u8; BLOCK_SIZE];
    let mut outer_key = [0x5cu8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        inner_key[i] ^= block[i];
        outer_key[i] ^= block[i];
    }

    let mut inner = Sha256::new();
    inner.update(inner_key);
    inner.update(message);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(outer_key);
    outer.update(inner_digest);
    format!("{:x}", outer.finalize())
}

/// A CP/1 document wrapped for transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedEnvelope {
    pub cp: String,
    pub schema: String,
    /// Canonical-form JSON of the document, as a string.
    pub payload: String,
    pub sha256: String,
    /// HMAC-SHA256 over `sha256`, keyed by the fleet secret. Optional: over a
    /// stdio subprocess boundary the parent already controls the child, so
    /// requiring a shared secret there would be ceremony without a threat.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hmac: Option<String>,
}

/// Why an envelope was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    BadSchema,
    /// `payload` does not hash to `sha256`: altered in transit.
    HashMismatch,
    SignatureMissing,
    SignatureInvalid,
    NoKeyToVerify,
    MalformedPayload,
    /// The payload parsed but is not canonical, so its `content_hash` could not
    /// have been computed over what was sent.
    NotCanonical,
    /// The document's `provenance.content_hash` does not match its content.
    SealBroken,
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EnvelopeError::BadSchema => "not a CP/1 signed envelope",
            EnvelopeError::HashMismatch => "payload hash mismatch (tampered in transit)",
            EnvelopeError::SignatureMissing => "fleet key configured but envelope is unsigned",
            EnvelopeError::SignatureInvalid => "HMAC signature does not verify",
            EnvelopeError::NoKeyToVerify => "envelope is signed but no fleet key is configured",
            EnvelopeError::MalformedPayload => "payload is not valid JSON",
            EnvelopeError::NotCanonical => "payload is not in CP/1 canonical form",
            EnvelopeError::SealBroken => "document content_hash does not match its content",
        })
    }
}

impl std::error::Error for EnvelopeError {}

impl SignedEnvelope {
    /// Wrap a document, sealing it first.
    pub fn seal(document: &Value, fleet_key: Option<&[u8]>) -> Result<Self, CanonicalError> {
        let mut document = document.clone();
        canonical::seal(&mut document)?;
        let payload = canonical::to_canonical(&document)?;
        let sha256 = sha256_hex(payload.as_bytes());
        let hmac = fleet_key.map(|key| hmac_sha256_hex(key, sha256.as_bytes()));
        Ok(Self {
            cp: crate::CP.to_string(),
            schema: ENVELOPE_SCHEMA.to_string(),
            payload,
            sha256,
            hmac,
        })
    }

    /// Verify the envelope and return the document it carries.
    ///
    /// Checks run outermost-first — schema, transport hash, signature, then the
    /// document's own seal — so the cheapest rejection happens first.
    pub fn open(&self, fleet_key: Option<&[u8]>) -> Result<Value, EnvelopeError> {
        if self.cp != crate::CP || self.schema != ENVELOPE_SCHEMA {
            return Err(EnvelopeError::BadSchema);
        }
        if sha256_hex(self.payload.as_bytes()) != self.sha256 {
            return Err(EnvelopeError::HashMismatch);
        }
        match (fleet_key, &self.hmac) {
            (Some(key), Some(mac)) => {
                let expected = hmac_sha256_hex(key, self.sha256.as_bytes());
                if !constant_time_eq(expected.as_bytes(), mac.as_bytes()) {
                    return Err(EnvelopeError::SignatureInvalid);
                }
            }
            (Some(_), None) => return Err(EnvelopeError::SignatureMissing),
            (None, Some(_)) => return Err(EnvelopeError::NoKeyToVerify),
            (None, None) => {}
        }

        let document: Value =
            serde_json::from_str(&self.payload).map_err(|_| EnvelopeError::MalformedPayload)?;
        if canonical::to_canonical(&document).map_err(|_| EnvelopeError::NotCanonical)?
            != self.payload
        {
            return Err(EnvelopeError::NotCanonical);
        }
        if !canonical::verify_seal(&document).map_err(|_| EnvelopeError::NotCanonical)? {
            return Err(EnvelopeError::SealBroken);
        }
        Ok(document)
    }

    /// Render as one line of the line-delimited JSON transport.
    pub fn to_line(&self) -> String {
        let value = serde_json::to_value(self).expect("SignedEnvelope always serializes");
        canonical::to_canonical(&value).expect("SignedEnvelope contains only strings")
    }

    /// Parse one line of the line-delimited JSON transport.
    pub fn from_line(line: &str) -> Result<Self, EnvelopeError> {
        serde_json::from_str(line).map_err(|_| EnvelopeError::BadSchema)
    }
}

/// Compare two byte strings without an early exit on the first difference.
///
/// Signature comparison with `==` leaks, through timing, how many leading bytes
/// of a forged MAC were correct — enough to reconstruct one byte at a time.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document() -> Value {
        json!({
            "cp": "cp1",
            "type": "Mutation",
            "id": "88888888-8888-4888-8888-888888888888",
            "target": "preferences.tone",
            "provenance": {
                "authored_by": "adam",
                "produced_at": "2026-01-01T00:00:00.000Z",
                "origin": "evolution:analyze",
                "evidence": [],
                "derived_from": []
            }
        })
    }

    #[test]
    fn hmac_matches_rfc_4231_test_case_1() {
        // Anchoring to a published vector proves this hand-rolled HMAC is the
        // real construction and not merely self-consistent — the failure mode
        // a round-trip test cannot detect.
        let key = [0x0bu8; 20];
        assert_eq!(
            hmac_sha256_hex(&key, b"Hi There"),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
    }

    #[test]
    fn hmac_matches_rfc_4231_test_case_2() {
        assert_eq!(
            hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn hmac_handles_a_key_longer_than_the_block_size() {
        // RFC 4231 test case 6: 131-byte key, which must be hashed down first.
        let key = [0xaau8; 131];
        assert_eq!(
            hmac_sha256_hex(
                &key,
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            ),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn unsigned_round_trip_returns_the_same_document() {
        let envelope = SignedEnvelope::seal(&document(), None).unwrap();
        let opened = envelope.open(None).unwrap();
        assert_eq!(opened["target"], json!("preferences.tone"));
        assert!(canonical::verify_seal(&opened).unwrap());
    }

    #[test]
    fn signed_round_trip_verifies_under_the_same_key() {
        let key = b"fleet-secret";
        let envelope = SignedEnvelope::seal(&document(), Some(key)).unwrap();
        assert!(envelope.hmac.is_some());
        assert!(envelope.open(Some(key)).is_ok());
    }

    #[test]
    fn a_different_key_does_not_verify() {
        let envelope = SignedEnvelope::seal(&document(), Some(b"fleet-secret")).unwrap();
        assert_eq!(
            envelope.open(Some(b"other-secret")),
            Err(EnvelopeError::SignatureInvalid)
        );
    }

    #[test]
    fn key_presence_must_match_on_both_sides() {
        let unsigned = SignedEnvelope::seal(&document(), None).unwrap();
        assert_eq!(
            unsigned.open(Some(b"k")),
            Err(EnvelopeError::SignatureMissing)
        );
        let signed = SignedEnvelope::seal(&document(), Some(b"k")).unwrap();
        assert_eq!(signed.open(None), Err(EnvelopeError::NoKeyToVerify));
    }

    #[test]
    fn tampering_with_the_payload_is_caught_by_the_transport_hash() {
        let mut envelope = SignedEnvelope::seal(&document(), None).unwrap();
        envelope.payload = envelope.payload.replace("concise", "verbose");
        envelope.payload = envelope.payload.replace("tone", "tona");
        assert_eq!(envelope.open(None), Err(EnvelopeError::HashMismatch));
    }

    #[test]
    fn a_consistently_rehashed_payload_is_still_caught_by_the_document_seal() {
        // The interesting attack: edit the payload AND recompute the transport
        // hash. The document's own content_hash is what stops it, which is why
        // CP/1 has both.
        let mut envelope = SignedEnvelope::seal(&document(), None).unwrap();
        envelope.payload = envelope.payload.replace("tone", "tona");
        envelope.sha256 = sha256_hex(envelope.payload.as_bytes());
        assert_eq!(envelope.open(None), Err(EnvelopeError::SealBroken));
    }

    #[test]
    fn a_non_canonical_payload_is_refused() {
        let envelope = SignedEnvelope::seal(&document(), None).unwrap();
        let padded = format!(" {}", envelope.payload);
        let tampered = SignedEnvelope {
            sha256: sha256_hex(padded.as_bytes()),
            payload: padded,
            ..envelope
        };
        assert_eq!(tampered.open(None), Err(EnvelopeError::NotCanonical));
    }

    #[test]
    fn a_foreign_schema_is_refused_before_anything_else() {
        let mut envelope = SignedEnvelope::seal(&document(), None).unwrap();
        envelope.schema = "something_else".to_string();
        assert_eq!(envelope.open(None), Err(EnvelopeError::BadSchema));
    }

    #[test]
    fn line_protocol_round_trips_without_embedded_newlines() {
        let envelope = SignedEnvelope::seal(&document(), Some(b"k")).unwrap();
        let line = envelope.to_line();
        assert!(!line.contains('\n'));
        assert_eq!(SignedEnvelope::from_line(&line).unwrap(), envelope);
    }

    #[test]
    fn constant_time_eq_matches_ordinary_equality() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
