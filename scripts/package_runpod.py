"""Package source and prepared data, including uncommitted branch changes."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/hotpotqa/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_files = [manifest_path]
    for name, info in manifest["files"].items():
        path = manifest_path.parent / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        data_files.append(path)
    source_files = [path for path in root.iterdir() if path.is_file() and
                    (path.suffix in {".py", ".sh", ".json", ".md", ".txt", ".ini", ".ipynb"} or
                     path.name in {"LICENSE", ".gitignore", ".gitattributes"})]
    for directory in ("scripts", "tests", "imgs"):
        source_files.extend(path for path in (root / directory).rglob("*")
                            if path.is_file() and "__pycache__" not in path.parts)
    output = root / "artifacts/comi-runpod.zip"
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(source_files + data_files):
            relative = path.relative_to(root).as_posix()
            if path.suffix == ".sh":
                archive.writestr(relative, path.read_bytes().replace(b"\r\n", b"\n"))
            else:
                archive.write(path, relative)
    with zipfile.ZipFile(temporary) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"Corrupt archive member: {corrupt}")
    temporary.replace(output)
    print(output)
    print("Size MiB:", round(output.stat().st_size / 2**20, 1))


if __name__ == "__main__":
    main()
