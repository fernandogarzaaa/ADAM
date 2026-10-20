//! Conservative detection of credential-shaped content.
//!
//! This is the second line of defense behind the rule that raw
//! secrets never enter leaves at all (spec section 5): a leaf whose
//! content looks like it carries a credential is excluded from
//! summarization inputs, and a summary containing such a shape is
//! rejected outright. The detector deliberately errs toward false
//! positives; a wrongly excluded leaf stays verbatim and readable,
//! which is the safe direction.

/// Well-known token prefixes that are credentials on sight.
const TOKEN_PREFIXES: &[&str] = &[
    "sk-ant-",
    "sk-live-",
    "sk_live_",
    "ghp_",
    "gho_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "AIza",
    "-----BEGIN",
];

/// Labels that introduce a credential value when followed by `:` or
/// `=` and a value of at least six characters.
const SECRET_LABELS: &[&str] = &[
    "password",
    "passwd",
    "api_key",
    "apikey",
    "api-key",
    "secret_key",
    "client_secret",
    "access_token",
    "auth_token",
    "private_key",
    "credential",
];

/// Returns true when `content` contains anything shaped like a
/// credential: a known token prefix, an AWS-style access key id, a
/// labelled secret assignment, or a long high-entropy token.
pub fn contains_credential_shaped(content: &str) -> bool {
    if TOKEN_PREFIXES.iter().any(|prefix| content.contains(prefix)) {
        return true;
    }
    if has_aws_access_key_id(content) {
        return true;
    }
    if has_labelled_secret(content) {
        return true;
    }
    content.split_whitespace().any(is_long_mixed_token)
}

/// AWS access key ids are `AKIA` followed by 16 uppercase/digit
/// characters.
fn has_aws_access_key_id(content: &str) -> bool {
    let bytes = content.as_bytes();
    if bytes.len() < 20 {
        return false;
    }
    for start in 0..=bytes.len() - 20 {
        if bytes[start..].starts_with(b"AKIA")
            && bytes[start + 4..start + 20]
                .iter()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

fn has_labelled_secret(content: &str) -> bool {
    let lower = content.to_lowercase();
    for label in SECRET_LABELS {
        for (idx, _) in lower.match_indices(label) {
            let rest = &lower[idx + label.len()..];
            let rest = rest.trim_start();
            let Some(rest) = rest.strip_prefix(':').or_else(|| rest.strip_prefix('=')) else {
                continue;
            };
            let value_len = rest
                .trim_start()
                .chars()
                .take_while(|c| !c.is_whitespace())
                .count();
            if value_len >= 6 {
                return true;
            }
        }
    }
    false
}

/// A whitespace-delimited token of 40+ characters drawn only from the
/// base64/base64url alphabet, mixing digits, uppercase, and lowercase
/// letters, is shaped like a key or token rather than prose. URLs are
/// exempt by construction: their `:` and other punctuation fail the
/// alphabet check.
fn is_long_mixed_token(token: &str) -> bool {
    let token = token.trim_matches(|c: char| ",;:\"'()[]{}<>".contains(c));
    if token.chars().count() < 40 {
        return false;
    }
    let mut has_digit = false;
    let mut has_upper = false;
    let mut has_lower = false;
    for c in token.chars() {
        if c.is_ascii_digit() {
            has_digit = true;
        } else if c.is_ascii_uppercase() {
            has_upper = true;
        } else if c.is_ascii_lowercase() {
            has_lower = true;
        } else if !"+/=_-.".contains(c) {
            return false;
        }
    }
    has_digit && has_upper && has_lower
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_aws_access_key_ids() {
        assert!(contains_credential_shaped(
            "the key is AKIAIOSFODNN7EXAMPLE thanks"
        ));
    }

    #[test]
    fn detects_labelled_secrets_and_known_prefixes() {
        assert!(contains_credential_shaped("password: hunter2hunter2"));
        assert!(contains_credential_shaped("api_key = abcdef123456"));
        assert!(contains_credential_shaped(
            "use sk-ant-api03-abcdefghijklmnop for this"
        ));
    }

    #[test]
    fn plain_prose_and_vault_references_pass() {
        assert!(!contains_credential_shaped(
            "Inan shared the X credential via the vault, never the value."
        ));
        assert!(!contains_credential_shaped(
            "contact: can we move the deploy to friday? cookie: yes."
        ));
        assert!(!contains_credential_shaped(
            "see https://example.com/Some/Long/Path/To/A/Document for details"
        ));
    }

    #[test]
    fn aws_key_id_at_exact_minimum_length_is_detected() {
        // "AKIA" + 16 chars = exactly 20 bytes: the shortest string
        // that can hold a key id at all.
        assert!(contains_credential_shaped("AKIAIOSFODNN7EXAMPLE"));
        // A bare prefix with nothing after it is not a key id, and
        // must not fall through to the scan loop either.
        assert!(!contains_credential_shaped("AKIA"));
    }

    #[test]
    fn aws_prefix_with_wrong_tail_characters_is_not_a_key_id() {
        // AKIA followed by lowercase: the prefix alone must not be
        // enough; the 16-char uppercase/digit tail is required.
        assert!(!contains_credential_shaped(
            "the marker AKIAiosfodnn7example is not a key"
        ));
        // Sixteen uppercase/digit chars with no AKIA prefix at all.
        assert!(!contains_credential_shaped("prefix ZZZZABCDEFGHIJKLMNOP end"));
    }

    /// A 40+ char token mixing digits, upper, and lower case, drawn
    /// from the token alphabet.
    fn mixed_token(chars: usize) -> String {
        let unit = "Ab3";
        let mut s = String::new();
        while s.chars().count() + unit.len() <= chars {
            s.push_str(unit);
        }
        while s.chars().count() < chars {
            s.push('c');
        }
        s
    }

    #[test]
    fn long_mixed_token_length_boundary() {
        // 39 chars: below the threshold, not credential-shaped.
        assert!(!contains_credential_shaped(&format!(
            "value {} here",
            mixed_token(39)
        )));
        // Exactly 40: at the threshold, detected.
        assert!(contains_credential_shaped(&format!(
            "value {} here",
            mixed_token(40)
        )));
    }

    #[test]
    fn long_mixed_token_punctuation_rules() {
        // Surrounding punctuation is trimmed before the check.
        let wrapped = format!("\"({}),\"", mixed_token(42));
        assert!(contains_credential_shaped(&wrapped));
        // A '.' inside the token is in the allowed alphabet.
        let mut dotted = mixed_token(42);
        dotted.replace_range(10..11, ".");
        assert!(contains_credential_shaped(&format!("value {dotted} here")));
        // A character outside the alphabet ('@') disqualifies the
        // token even at length with all three classes present.
        let mut with_at = mixed_token(42);
        with_at.replace_range(10..11, "@");
        assert!(!contains_credential_shaped(&format!(
            "value {with_at} here"
        )));
    }
}
