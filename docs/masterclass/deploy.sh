#!/bin/sh
set -eu

cd "$(dirname "$0")"
python3 verify.py

assets_dir=$(mktemp -d)
trap 'rm -f "$assets_dir/index.html"; rmdir "$assets_dir"' EXIT
cp index.html "$assets_dir/"

npx --yes cf@1.0.0-rc.4 deploy "$assets_dir" "$@"
