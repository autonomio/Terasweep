#!/usr/bin/env python3
"""Verify the Git handoff without extracting archives or rerunning benchmarks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'benchmarks/SHA256SUMS'
LINE = re.compile(r'^([a-fA-F0-9]{64}) [ *](.+)$')
# High-confidence credential formats, not a comprehensive security audit.
SECRETS = re.compile(
    rb'(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{60,255}'
    rb'|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'
)
INNER_MANIFESTS = {
    'avx512_kernel_experiment.tar.gz': 'avx512_lab/input-sha256.txt',
    'handwritten_avx512_experiment.tar.gz': 'handwritten_avx512_experiment/checksums.txt',
    'terasweep_reverse_funnel_experiment.tar.gz': 'terasweep-integrated/SHA256SUMS',
    'terasweep_scaling_experiment.tar.gz': 'terasweep-scaling-experiment/SHA256SUMS',
}


def safe_path(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError(f'Unsafe path: {name!r}')
    return p


def hash_stream(stream: BinaryIO, label: str, scan: bool = False) -> str:
    h = hashlib.sha256()
    tail = b''
    while True:
        block = stream.read(1 << 20)
        if not block:
            break
        h.update(block)
        if scan and SECRETS.search(tail + block):
            raise ValueError(f'Possible credential in {label}; review before publishing')
        tail = block[-512:]
    return h.hexdigest()


def hash_file(path: Path, scan: bool = False) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing regular file: {path}')
    with path.open('rb') as stream:
        return hash_stream(stream, str(path), scan)


def parse_sums(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if not line:
            continue
        m = LINE.fullmatch(line)
        if m is None:
            raise ValueError('Malformed SHA256SUMS line')
        expected, name = m.groups()
        name = str(safe_path(name))
        if name in result:
            raise ValueError(f'Duplicate checksum entry: {name}')
        result[name] = expected.lower()
    return result


def require_equal(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise ValueError(f'Checksum mismatch: {label}')


def verify_archive(path: Path) -> tuple[dict[str, str], int]:
    hashes = {}
    manifests = {}
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = str(safe_path(member.name))
            if not (member.isdir() or member.isfile()):
                raise ValueError(f'Non-regular archive member: {path.name}:{name}')
            if member.isdir():
                continue
            if name in hashes:
                raise ValueError(f'Duplicate archive member: {path.name}:{name}')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'Unreadable archive member: {name}')
            with stream:
                hashes[name] = hash_stream(stream, f'{path.name}:{name}', scan=True)
            if name == INNER_MANIFESTS.get(path.name):
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f'Unreadable embedded checksum list: {name}')
                with stream:
                    manifests[name] = stream.read().decode('utf-8')
    checked = 0
    required = INNER_MANIFESTS.get(path.name)
    if required is not None and required not in manifests:
        raise ValueError(f'Missing embedded checksum list in {path.name}')
    for name, text in manifests.items():
        prefix = PurePosixPath(name).parent
        for rel, expected in parse_sums(text).items():
            member = str(prefix / safe_path(rel))
            require_equal(hashes.get(member, ''), expected, f'{path.name}:{member}')
            checked += 1
    print(f'Archive OK: {path.name}: {len(hashes)} files; {checked} embedded checksums')
    return hashes, checked


def main() -> None:
    expected = parse_sums(MANIFEST.read_text(encoding='utf-8'))
    actual_files = {
        str(p.relative_to(ROOT)) for p in (ROOT / 'benchmarks').rglob('*')
        if p.is_file() and p != MANIFEST
    } | {'scripts/verify_evidence.py'}
    if actual_files != set(expected):
        raise ValueError(f'Manifest coverage mismatch; missing={actual_files-set(expected)}, stale={set(expected)-actual_files}')
    for rel, digest in expected.items():
        require_equal(hash_file(ROOT / rel, scan=not rel.endswith('.tar.gz')), digest, rel)
    print(f'Handoff files OK: {len(expected)} checksums')
    source = json.loads((ROOT / 'benchmarks/handoff/pipeline-source-sha256.json').read_text())
    for rel, digest in source.items():
        require_equal(hash_file(ROOT / safe_path(rel)), digest, rel)
    print(f'Measured pipeline source OK: {len(source)} files')
    archive_hashes = {}
    member_count = embedded_count = 0
    for path in sorted((ROOT / 'benchmarks').rglob('*.tar.gz')):
        hashes, checked = verify_archive(path)
        archive_hashes[path.name] = hashes
        member_count += len(hashes)
        embedded_count += checked
    history = json.loads((ROOT / 'benchmarks/mac-history/inventory.json').read_text())
    for entry in history['files']:
        archive = Path(entry['archive']).name
        require_equal(archive_hashes[archive].get(entry['member'], ''), entry['sha256'], f'{archive}:{entry["member"]}')
    print(f'Mac-history inventory OK: {len(history["files"])} member checksums')
    print(json.dumps({'status': 'PASS', 'handoff_files': len(expected),
        'source_files': len(source), 'archives': len(archive_hashes),
        'archive_members': member_count, 'embedded_checksums': embedded_count,
        'history_checksums': len(history['files'])}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f'Evidence verification FAILED: {error}', file=sys.stderr)
        raise SystemExit(1) from error
