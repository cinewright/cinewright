"""Source files are UTF-8: git stores bytes as they are, and a Latin-1 byte in a .tsx file reaches the UI as U+FFFD."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tests", "scripts", "installer", "web/src", "web/scripts", "design")
TEXT_SUFFIXES = {".py", ".ts", ".tsx", ".mjs", ".mts", ".js", ".css", ".json", ".md", ".nsi", ".nsh", ".txt", ".yaml", ".yml", ".sql"}
SKIPPED_PARTS = {"web_dist", "node_modules", "__pycache__"}


def source_files():
    for directory in SOURCE_DIRS:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and not SKIPPED_PARTS.intersection(path.parts):
                yield path


def test_every_source_file_is_utf8():
    files = list(source_files())
    assert len(files) > 50
    not_utf8 = []
    for path in files:
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            not_utf8.append(f"{path.relative_to(ROOT).as_posix()} (byte {exc.start})")
    assert not_utf8 == []
