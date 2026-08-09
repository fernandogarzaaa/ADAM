#!/usr/bin/env bash
# Re-vendor the CP/1 normative source from AXIOM-AETHER.
#
# ADAM implements a hand-written CP/1 binding and verifies it against the shared
# fixture corpus. That corpus is a *copy*: there is no build-time dependency on
# AXIOM-AETHER, which is what lets three repositories in three languages ship on
# independent cadences. The cost is that the copy must be refreshed when the
# protocol changes, and this script is how.
#
#   scripts/vendor-protocol.sh [path-to-AXIOM-AETHER]
#
# Defaults to a sibling checkout at ../AXIOM-AETHER. Afterwards run
# `cargo test -p adam-protocol`: the manifest check passes only if the copy is
# byte-identical to the source, and the conformance checks fail if ADAM's
# binding does not implement whatever changed.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${1:-$root/../AXIOM-AETHER}/protocol/cp1"

if [[ ! -d "$source_root" ]]; then
  echo "No CP/1 source at $source_root" >&2
  echo "Pass the path to an AXIOM-AETHER checkout:" >&2
  echo "  scripts/vendor-protocol.sh /path/to/AXIOM-AETHER" >&2
  exit 1
fi

# Files a binding must carry verbatim. Mirrors MANIFEST.sha256's entries.
#
# SPEC.md is here because the manifest hashes it: the normative text is part of
# what a binding claims to conform to, and a copy that drifted from the source
# would let this repository cite a specification nobody else is reading.
for relative in VERSION MANIFEST.sha256 SPEC.md schema/cp1.schema.json fixtures/canonical.jsonl; do
  mkdir -p "$root/protocol/cp1/$(dirname "$relative")"
  cp "$source_root/$relative" "$root/protocol/cp1/$relative"
  echo "vendored $relative"
done

echo
echo "CP/1 $(cat "$root/protocol/cp1/VERSION") vendored from $source_root."
echo "Run 'cargo test -p adam-protocol' to confirm this binding still conforms."
