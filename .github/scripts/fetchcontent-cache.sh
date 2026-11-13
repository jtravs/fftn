#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Keeps the archives that FetchContent downloads (GoogleTest, Boost, FFTW) between CI runs.
# FetchContent's download step skips an archive that is already in place with the pinned
# hash, so the archives are copied into every gate preset's build tree before the gate
# configures it, and collected from them afterwards. Only the archives are kept, never a
# build: each run still unpacks and builds them for its own compiler.
#
#   .github/scripts/fetchcontent-cache.sh restore <cache directory>
#   .github/scripts/fetchcontent-cache.sh save <cache directory>
set -euo pipefail
action=$1 cache=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
# Relative to build/<preset>/_deps: <name>-subbuild/<name>-populate-prefix/src/<archive>.
archives() {
  find "$1" -path '*-subbuild/*-populate-prefix/src/*' -type f \
    \( -name '*.tar.gz' -o -name '*.tar.xz' -o -name '*.tgz' -o -name '*.zip' \) |
    sed "s|^$1/||"
}
case $action in
  restore)
    [[ -d $cache ]] || { echo "fetchcontent-cache: nothing cached yet"; exit 0; }
    presets=$(cmake --list-presets | sed -n 's/^ *"\([^"]*\)".*/\1/p')
    n=0
    for preset in $presets; do
      while IFS= read -r rel; do
        mkdir -p "build/$preset/_deps/$(dirname "$rel")"
        cp "$cache/$rel" "build/$preset/_deps/$rel"
        n=$((n + 1))
      done < <(archives "$cache")
    done
    echo "fetchcontent-cache: $n archives placed for presets: ${presets//$'\n'/ }"
    ;;
  save)
    mkdir -p "$cache"
    for deps in build/*/_deps; do
      [[ -d $deps ]] || continue
      while IFS= read -r rel; do
        mkdir -p "$cache/$(dirname "$rel")"
        cp "$deps/$rel" "$cache/$rel"
      done < <(archives "$deps")
    done
    echo "fetchcontent-cache: $(archives "$cache" | wc -l | tr -d ' ') archives kept"
    ;;
  *)
    echo "usage: fetchcontent-cache.sh restore|save <cache directory>" >&2
    exit 2
    ;;
esac
