"""Offline review tool. Does not add candidates to the runtime allowlist."""
import argparse
import contextlib
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from monkey_textdet.common import save_json
from monkey_textdet.manifest_identity import compare_manifest_streams


def open_manifest(stack, path):
    if path.suffix.lower() == '.zip':
        archive = stack.enter_context(zipfile.ZipFile(path))
        if archive.namelist().count('shared_manifest_v1.jsonl') != 1:
            raise ValueError('ZIP must contain exactly one root shared_manifest_v1.jsonl')
        return stack.enter_context(archive.open('shared_manifest_v1.jsonl'))
    return stack.enter_context(path.open('rb'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with contextlib.ExitStack() as stack:
        result = compare_manifest_streams(open_manifest(stack, args.reference),
                                          open_manifest(stack, args.candidate))
    save_json(args.report, result)
    print('Compared rows:', result['rows'], 'byte-different rows:', result['byte_different_rows'])
    print('Candidate:', result['candidate_manifest_sha256'])
    print('Report:', args.report)


if __name__ == '__main__':
    main()
