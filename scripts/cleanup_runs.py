#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys, time, tarfile
from typing import List


EXCLUDE_DIRS = {os.path.normpath('runs/_aggregated'), os.path.normpath('runs/_catalog')}


def should_include(path: str, min_age_minutes: int) -> bool:
    # Only operate under runs/
    if not path.startswith('runs' + os.sep):
        return False
    # Exclude aggregated/catalog trees
    for ex in EXCLUDE_DIRS:
        if os.path.normpath(path).startswith(ex + os.sep) or os.path.normpath(path) == ex:
            return False
    # Keep only JSON logs/reports
    if not (path.endswith('.jsonl') or path.endswith('.json')):
        return False
    # Age filter (avoid currently-being-written files)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    age_min = (time.time() - mtime) / 60.0
    return age_min >= float(min_age_minutes)


def collect_files(root: str, min_age_minutes: int) -> List[str]:
    out: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if should_include(p, min_age_minutes):
                out.append(p)
    return sorted(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Archive and remove older run logs under runs/')
    p.add_argument('--min-age-minutes', type=int, default=10, help='only archive files older than this (default: 10)')
    p.add_argument('--out-dir', default='runs/_archive', help='archive destination directory')
    p.add_argument('--dry-run', action='store_true', help='list files only; do not archive/remove')
    args = p.parse_args(argv)

    files = collect_files('runs', args.min_age_minutes)
    if not files:
        print('[info] no files eligible for archive (age filter)')
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    archive_path = os.path.join(args.out_dir, f'logs_archive_{ts}.tar.gz')
    manifest_path = os.path.join(args.out_dir, f'manifest_{ts}.txt')

    # Always write manifest for audit
    with open(manifest_path, 'w', encoding='utf-8') as mf:
        for pth in files:
            mf.write(pth + '\n')

    if args.dry_run:
        print(f'[dry-run] would archive {len(files)} files -> {archive_path}')
        print(manifest_path)
        return 0

    # Create tar.gz archive with relative paths
    with tarfile.open(archive_path, 'w:gz') as tf:
        for pth in files:
            try:
                tf.add(pth, arcname=pth)
            except Exception:
                pass

    # Remove originals
    removed = 0
    for pth in files:
        try:
            os.remove(pth)
            removed += 1
        except Exception:
            pass

    print({'archive': archive_path, 'manifest': manifest_path, 'files': len(files), 'removed': removed})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

