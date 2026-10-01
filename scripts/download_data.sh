#!/usr/bin/env bash
# Fetch the extra melody sources into data/raw/ (gitignored). Run from the repo root.
#   HookTheory (Sheet Sage): CC BY-NC-SA 3.0 -- non-commercial use only
#   POP909:                  MIT
set -euo pipefail
mkdir -p data/raw

if [ ! -f data/raw/Hooktheory.json.gz ]; then
  curl -fL -o data/raw/Hooktheory.json.gz \
    https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory.json.gz
fi
echo "917b7cd5" | grep -q "^$(shasum -a 256 data/raw/Hooktheory.json.gz | cut -c1-8)$" \
  || { echo "Hooktheory.json.gz checksum mismatch" >&2; exit 1; }

if [ ! -d data/raw/POP909 ]; then
  curl -fL -o data/raw/POP909.zip https://github.com/music-x-lab/POP909-Dataset/raw/master/POP909.zip
  unzip -q data/raw/POP909.zip -d data/raw/
  rm data/raw/POP909.zip
fi
echo "done: $(ls data/raw)"
