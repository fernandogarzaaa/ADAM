# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Dependabot configuration for cargo and GitHub Actions (weekly).
- `.env.example` documenting the environment variables ADAM reads (names only); local `.env` files are now git-ignored.
- SECURITY.md describing private vulnerability reporting.
- This CHANGELOG.

### Security

- Bump `rustls` 0.23.43 → 0.23.45 in `Cargo.lock` (RUSTSEC-2026-0285, TLS 1.3 handshake messages accepted across encryption levels; transitive via fastembed → hf-hub → ureq) so the `cargo audit` CI step passes.
