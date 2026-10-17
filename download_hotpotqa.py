"""Download original HotpotQA JSON and convert it to COMI JSONL.

Source: https://hotpotqa.github.io/ (CC BY-SA 4.0).
The labeled dev distractor split becomes test.jsonl, not the official test set.
Uses only the Python standard library. Conversion loads one original split
into memory at a time (the training JSON is approximately 535 MB on disk).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


BASE_URL = 'http://curtis.ml.cmu.edu/datasets/hotpot/'
SPLITS = {
    'train': ('hotpot_train_v1.1.json', 'train.jsonl'),
    'dev': ('hotpot_dev_distractor_v1.json', 'test.jsonl'),
}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / 'data' / 'hotpotqa'


def download(url: str, destination: Path) -> None:
    """Cache complete downloads; failed transfers never become original files."""
    if destination.is_file() and destination.stat().st_size > 0:
        print(f'Using original: {destination}', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        temporary = None
        try:
            print(f'Downloading {url} (attempt {attempt}/3)', flush=True)
            with urlopen(url, timeout=60) as response:
                expected = response.headers.get('Content-Length')
                received = 0
                next_report = 64 * 1024 * 1024
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix='.part', delete=False
                ) as output:
                    temporary = Path(output.name)
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if received >= next_report:
                            print(f'  {received / 1024**2:.0f} MiB downloaded', flush=True)
                            next_report += 64 * 1024 * 1024
                if received == 0 or (expected is not None and received != int(expected)):
                    raise OSError(f'Incomplete download: received {received}, expected {expected} bytes')
            temporary.replace(destination)
            return
        except (OSError, URLError) as error:
            if attempt == 3:
                raise RuntimeError(f'Failed to download {url}: {error}') from error
            print(f'Download failed: {error}; retrying...', flush=True)
            time.sleep(attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def to_comi(example: dict) -> dict:
    """Keep all paragraphs in source order; never select by supporting facts."""
    for key in ('_id', 'question', 'answer', 'context'):
        if key not in example:
            raise ValueError(f'HotpotQA example is missing {key!r}')
    if not isinstance(example['question'], str) or not isinstance(example['answer'], str):
        raise ValueError('Expected a labeled HotpotQA question with a string answer')
    paragraphs = []
    for title, sentences in example['context']:
        if not isinstance(title, str) or not isinstance(sentences, list):
            raise ValueError('Expected context entries of [title, [sentences]]')
        if not all(isinstance(sentence, str) for sentence in sentences):
            raise ValueError('Context sentences must be strings')
        paragraphs.append(f'{title}\n{" ".join(s.strip() for s in sentences)}')
    return {
        'id': example['_id'],
        'input': '\n\n'.join(paragraphs),
        'prompt': example['question'],
        'answer': [example['answer']],
    }


def convert(original: Path, destination: Path, limit: int = 0) -> int:
    print(f'Converting {original.name} -> {destination}', flush=True)
    with original.open(encoding='utf-8') as source:
        examples = json.load(source)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f'{original} must contain a non-empty JSON array')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='\n',
            dir=destination.parent, suffix='.part', delete=False,
        ) as output:
            temporary = Path(output.name)
            for index, example in enumerate(examples):
                if limit and index >= limit:
                    break
                try:
                    row = to_comi(example)
                except (ValueError, TypeError) as error:
                    raise ValueError(f'{original}: example {index}: {error}') from error
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Wrote {count} examples', flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=['train', 'dev'])
    parser.add_argument('--skip-download', action='store_true', help='Convert existing originals offline')
    parser.add_argument('--limit', type=int, default=0, help='First N examples per JSONL; 0 = all. Originals stay complete.')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    for split in dict.fromkeys(args.splits):
        filename, output_name = SPLITS[split]
        original = args.output_dir / 'original' / filename
        if args.skip_download:
            if not original.is_file():
                parser.error(f'Original not found: {original}')
        else:
            download(BASE_URL + filename, original)
        convert(original, args.output_dir / output_name, args.limit)
    print('Original files are preserved. Re-running regenerates the selected JSONL files.')


if __name__ == '__main__':
    main()
