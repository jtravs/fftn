#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Downloads a pinned file and checks its SHA-256; the file is removed when the hash differs.
#
#   .github/scripts/fetch.sh <url> <sha256> <output file>
set -euo pipefail
url=$1 want=$2 out=$3
curl --fail --silent --show-error --location --retry 5 --retry-all-errors -o "$out" "$url"
if command -v sha256sum > /dev/null; then
  got=$(sha256sum "$out" | cut -d' ' -f1)
else
  got=$(shasum -a 256 "$out" | cut -d' ' -f1)
fi
if [[ $got != "$want" ]]; then
  rm -f "$out"
  echo "fetch: SHA-256 of $url is $got, expected $want" >&2
  exit 1
fi
echo "fetch: $(basename "$out") verified ($want)"
