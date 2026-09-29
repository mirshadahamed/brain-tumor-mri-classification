"""Verify shared membership, labels, content hashes, and group separation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / 'data_preparation' / 'manifests'


def main():
    summary = json.loads((MANIFESTS / 'split_summary.json').read_text())
    seen_paths, seen_hashes, seen_groups = set(), set(), set()
    for split in ('train', 'validation', 'test'):
        with (MANIFESTS / f'{split}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == summary['split_totals'][split]
        paths, hashes, groups = set(), set(), set()
        counts = {}
        for row in rows:
            path = ROOT / 'dataset' / row['source_path']
            assert row['split'] == split
            assert int(row['class_id']) == summary['class_to_idx'][row['class']]
            assert path.is_file(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
            assert row['source_path'] not in paths
            assert row['pixel_sha256'] not in hashes
            paths.add(row['source_path'])
            hashes.add(row['pixel_sha256'])
            groups.add(row['group_id'])
            counts[row['class']] = counts.get(row['class'], 0) + 1
        assert counts == summary['split_counts'][split]
        assert paths.isdisjoint(seen_paths)
        assert hashes.isdisjoint(seen_hashes)
        assert groups.isdisjoint(seen_groups)
        seen_paths.update(paths)
        seen_hashes.update(hashes)
        seen_groups.update(groups)
        print(f'{split}: {len(rows)} images verified')
    print('All source hashes, labels, counts, and split boundaries passed.')


if __name__ == '__main__':
    main()
