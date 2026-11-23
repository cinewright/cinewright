"""Build the Cinewright Windows installer.

Steps:
  1. web        npm ci / npm run build in web/ (output: src/cinewright/web_dist)
  2. wheel      python -m build --wheel --outdir dist
  3. uv         pinned uv.exe for Windows (installer/pins.json), sha256-verified, cached in build/cache
  4. lock       build/installer/requirements.lock: the runtime dependencies from pyproject.toml, resolved for
                Windows with uv and pinned by sha256 (the installer installs it with --require-hashes)
  5. icon       build/installer/cinewright.ico from the logo SVGs (installer/make_icon.py)
  6. installer  makensis installer/cinewright.nsi -> dist/CinewrightSetup-<version>.exe

Runs on Linux (CI) and Windows. On Linux without makensis on PATH it installs the nsis package with
apt-get when that needs no password (CI runners, root containers), else it unpacks NSIS from the
distribution's packages into build/nsis without root.
Usage: python scripts/build.py [--skip-web] [--skip-installer]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
WEB_DIST = ROOT / "src" / "cinewright" / "web_dist"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
CACHE = BUILD / "cache"
STAGE = BUILD / "installer"
NSIS_ROOT = BUILD / "nsis"
INSTALLER = ROOT / "installer"
PINS = INSTALLER / "pins.json"
NSI = INSTALLER / "cinewright.nsi"
NOTICES = INSTALLER / "THIRD-PARTY-NOTICES.txt"
LICENSE = ROOT / "LICENSE"
PYPROJECT = ROOT / "pyproject.toml"
LOCK_PYTHON = "3.12"  # the runtime the installer creates (installer/cinewright.nsi)
LOCK_PLATFORM = "x86_64-pc-windows-msvc"

NODE_RANGE = "^20.19.0 || >=22.12.0"  # engines field of web/package.json
UV_ASSET = "uv-x86_64-pc-windows-msvc.zip"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/download/"
WHEEL_NAME = re.compile(r"^cinewright-(?P<version>[^-]+)-py3-none-any\.whl$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
STEPS = ("web", "wheel", "uv", "lock", "icon", "installer")

NSIS_HELP_WINDOWS = """makensis (NSIS 3) was not found.
Install NSIS 3 from https://nsis.sourceforge.io/Download (or: winget install NSIS.NSIS),
then add its folder to PATH, or run this script with --skip-installer."""


class BuildError(Exception):
    """A build step failed; the message says why."""


class ChecksumError(BuildError):
    """A downloaded file does not match its pinned sha256."""


def log(step: str, message: str) -> None:
    number = f"{STEPS.index(step) + 1}/{len(STEPS)} " if step in STEPS else ""
    print(f"[build {number}{step}] {message}", flush=True)


def run(step: str, cmd: list, cwd: Path = ROOT, env: dict | None = None, check: bool = True) -> int:
    args = [str(part) for part in cmd]
    log(step, "$ " + " ".join(args))
    result = subprocess.run(args, cwd=cwd, env=env, check=False)  # noqa: S603 - fixed build tool invocations
    if check and result.returncode != 0:
        raise BuildError(f"{Path(args[0]).name} {args[1] if len(args) > 1 else ''} failed with exit code {result.returncode}")
    return result.returncode


# Helpers (unit-tested)


def wheel_version(name: str) -> str:
    """Return the version from a Cinewright wheel file name; only pure-Python py3-none-any wheels are valid."""
    match = WHEEL_NAME.match(name)
    if not match:
        raise BuildError(f"unexpected wheel file name {name!r}; expected cinewright-<version>-py3-none-any.whl")
    return match["version"]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ChecksumError(f"sha256 mismatch for {path.name}: expected {expected}, got {actual}")


def parse_node_version(text: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)\S*", text.strip())
    if not match:
        raise BuildError(f"cannot parse the Node.js version from {text.strip()!r}")
    return int(match[1]), int(match[2]), int(match[3])


def node_supported(version: tuple[int, int, int]) -> bool:
    """Whether a Node.js version satisfies NODE_RANGE (^20.19.0 || >=22.12.0)."""
    major, minor = version[:2]
    return (major == 20 and minor >= 19) or (major, minor) >= (22, 12)


def load_pins(path: Path = PINS) -> dict:
    """Load and validate installer/pins.json."""
    try:
        pins = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BuildError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(pins, dict) or set(pins) != {"uv"}:
        raise BuildError(f"{path.name} must contain exactly the key 'uv'")
    uv = pins["uv"]
    keys = {"version", "published", "asset", "url", "sha256"}
    if not isinstance(uv, dict) or set(uv) != keys or not all(isinstance(uv[k], str) for k in keys):
        raise BuildError(f"{path.name}: 'uv' must have exactly the string fields {sorted(keys)}")
    if not SEMVER.match(uv["version"]):
        raise BuildError(f"{path.name}: uv version {uv['version']!r} is not X.Y.Z")
    try:
        date.fromisoformat(uv["published"])
    except ValueError as exc:
        raise BuildError(f"{path.name}: uv published date {uv['published']!r} is not YYYY-MM-DD") from exc
    if uv["asset"] != UV_ASSET:
        raise BuildError(f"{path.name}: uv asset must be {UV_ASSET}")
    if uv["url"] != f"{UV_RELEASES}{uv['version']}/{UV_ASSET}":
        raise BuildError(f"{path.name}: uv url must be {UV_RELEASES}<version>/{UV_ASSET}")
    if not SHA256.match(uv["sha256"]):
        raise BuildError(f"{path.name}: uv sha256 must be 64 lowercase hex characters")
    return pins


def compose_license(license_text: str, notices: str) -> bytes:
    """Licence page text: Cinewright's licence, then the third-party notices; ASCII with CRLF line ends."""
    text = license_text.strip() + "\n\n\n" + notices.strip() + "\n"
    try:
        data = text.replace("\r\n", "\n").encode("ascii")
    except UnicodeEncodeError as exc:
        raise BuildError("the licence page text must be ASCII") from exc
    return data.replace(b"\n", b"\r\n")


# Steps


def build_web() -> None:
    if not (WEB / "package.json").is_file():
        raise BuildError("web/package.json not found; the release build needs the web UI (use --skip-web for a dev build)")
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        raise BuildError(f"Node.js {NODE_RANGE} with npm is required to build the web UI (https://nodejs.org)")
    probe = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)  # noqa: S603
    version = parse_node_version(probe.stdout)
    if not node_supported(version):
        raise BuildError(f"Node.js {'.'.join(map(str, version))} is not supported; the web build needs {NODE_RANGE}")
    log("web", f"node {'.'.join(map(str, version))}")
    install = ["ci"] if (WEB / "package-lock.json").is_file() else ["install"]
    run("web", [npm, *install, "--no-audit", "--no-fund"], cwd=WEB)
    if WEB_DIST.exists():
        shutil.rmtree(WEB_DIST)
    run("web", [npm, "run", "build"], cwd=WEB)
    if not (WEB_DIST / "index.html").is_file():
        raise BuildError("npm run build did not produce src/cinewright/web_dist/index.html")


def build_wheel(expect_web: bool) -> tuple[Path, str]:
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob("cinewright-*.whl"):
        old.unlink()
    run("wheel", [sys.executable, "-m", "build", "--wheel", "--outdir", DIST])
    wheels = sorted(DIST.glob("cinewright-*.whl"))
    if len(wheels) != 1:
        raise BuildError(f"expected one wheel in dist/, found {len(wheels)}")
    wheel = wheels[0]
    version = wheel_version(wheel.name)
    with zipfile.ZipFile(wheel) as archive:
        has_web = "cinewright/web_dist/index.html" in archive.namelist()
    if expect_web and not has_web:
        raise BuildError(f"{wheel.name} does not contain cinewright/web_dist/index.html")
    log("wheel", f"{wheel.name} ({wheel.stat().st_size / 1e6:.1f} MB, web UI {'included' if has_web else 'NOT included'})")
    return wheel, version


def download(url: str, dest: Path, expected_sha256: str) -> None:
    """Download url to dest; dest only appears once its sha256 matches."""
    if not url.startswith("https://"):
        raise BuildError(f"refusing non-HTTPS download {url}")
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as handle:  # noqa: S310 - https only
            shutil.copyfileobj(response, handle)
        verify_sha256(part, expected_sha256)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise BuildError(f"download of {url} failed: {exc}") from exc
    except ChecksumError:
        part.unlink()
        raise
    part.replace(dest)


def stage_uv(pins: dict) -> Path:
    uv = pins["uv"]
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / f"uv-{uv['version']}-{uv['asset']}"
    if archive.is_file():
        try:
            verify_sha256(archive, uv["sha256"])
            log("uv", f"using cached {archive.name}")
        except ChecksumError:
            log("uv", f"cached {archive.name} does not match the pin; downloading again")
            archive.unlink()
    if not archive.is_file():
        log("uv", f"downloading uv {uv['version']} from {uv['url']}")
        download(uv["url"], archive, uv["sha256"])
    log("uv", f"sha256 {uv['sha256']} verified")
    with zipfile.ZipFile(archive) as zipped:
        members = [name for name in zipped.namelist() if PurePosixPath(name).name == "uv.exe"]
        if len(members) != 1:
            raise BuildError(f"{archive.name} does not contain exactly one uv.exe")
        data = zipped.read(members[0])
    STAGE.mkdir(parents=True, exist_ok=True)
    target = STAGE / "uv.exe"
    target.write_bytes(data)
    return target


def compile_lock_command(output: Path) -> list:
    """The uv command that resolves the runtime dependencies (no extras, so no dev tools) for Windows with hashes."""
    return [
        sys.executable, "-m", "uv", "pip", "compile", PYPROJECT,
        "--python-version", LOCK_PYTHON, "--python-platform", LOCK_PLATFORM,
        "--only-binary", ":all:", "--generate-hashes", "--no-header", "--no-config", "-o", output,
    ]  # fmt: skip


def stage_lock() -> Path:
    """Generate build/installer/requirements.lock with the uv Python package (a dev extra) and check it."""
    STAGE.mkdir(parents=True, exist_ok=True)
    target = STAGE / "requirements.lock"
    target.unlink(missing_ok=True)
    try:
        run("lock", compile_lock_command(target))
    except BuildError as exc:
        raise BuildError(f"{exc}; the lock step needs the uv package from the dev extras (pip install -e '.[dev]')") from exc
    if not target.is_file():
        raise BuildError("uv pip compile did not write requirements.lock")
    check_lock(target.read_text(encoding="utf-8"))
    log("lock", f"wrote {target.relative_to(ROOT)}")
    return target


def check_lock(text: str) -> None:
    """Every requirement in the lock must be pinned (==) and carry a sha256 hash, or --require-hashes fails at install."""
    entries, current = [], None
    for line in text.splitlines():
        if line.startswith(("#", " ", "	")) or not line.strip():
            if current is not None and "--hash=sha256:" in line:
                current["hashed"] = True
            continue
        current = {"line": line.rstrip(" \\"), "hashed": False}
        entries.append(current)
    if not entries:
        raise BuildError("requirements.lock is empty")
    for entry in entries:
        if "==" not in entry["line"] or not entry["hashed"]:
            raise BuildError(f"requirements.lock entry without a pinned version or hash: {entry['line']}")


def stage_icon() -> Path:
    spec = importlib.util.spec_from_file_location("cinewright_make_icon", INSTALLER / "make_icon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    icon = module.build_icon(STAGE / "cinewright.ico")
    log("icon", f"wrote {icon.relative_to(ROOT)}")
    return icon


def stage_license() -> Path:
    target = STAGE / "LICENSE.txt"
    STAGE.mkdir(parents=True, exist_ok=True)
    target.write_bytes(compose_license(LICENSE.read_text(encoding="utf-8"), NOTICES.read_text(encoding="utf-8")))
    return target


def install_nsis_with_apt() -> Path | None:
    """Install the nsis package system-wide with apt-get if that needs no password; returns makensis or None."""
    if not shutil.which("apt-get"):
        return None
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        prefix = []
    elif shutil.which("sudo") and subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode == 0:  # noqa: S607
        prefix = ["sudo", "-n"]
    else:
        return None
    log("installer", "makensis not on PATH; installing the nsis package with apt-get")
    if run("installer", [*prefix, "apt-get", "update", "-qq"], check=False) != 0:
        log("installer", "apt-get update reported errors; trying the install anyway")
    if run("installer", [*prefix, "apt-get", "install", "-y", "-qq", "nsis"], check=False) != 0:
        log("installer", "apt-get install failed; unpacking the packages without root instead")
        return None
    found = shutil.which("makensis")
    return Path(found) if found else None


def find_makensis_linux() -> tuple[Path, dict | None]:
    """makensis on PATH, else an earlier unpack in build/nsis, else apt-get install (no password), else a rootless unpack."""
    found = shutil.which("makensis")
    if found:
        return Path(found), None
    if not (NSIS_ROOT / "usr" / "bin" / "makensis").is_file():
        installed = install_nsis_with_apt()
        if installed:
            return installed, None
    return fetch_nsis_linux()


def fetch_nsis_linux() -> tuple[Path, dict]:
    """Unpack the distribution's nsis packages into build/nsis (no root needed)."""
    makensis = NSIS_ROOT / "usr" / "bin" / "makensis"
    env = {**os.environ, "NSISDIR": str(NSIS_ROOT / "usr" / "share" / "nsis")}
    if makensis.is_file():
        return makensis, env
    if not shutil.which("apt-get") or not shutil.which("dpkg"):
        raise BuildError("makensis was not found and apt-get/dpkg are unavailable; install NSIS 3 (package 'nsis')")
    debs = CACHE / "nsis-debs"
    if debs.exists():
        shutil.rmtree(debs)
    debs.mkdir(parents=True)
    log("installer", "makensis not on PATH; fetching the nsis packages with apt-get download")
    try:
        run("installer", ["apt-get", "download", "nsis", "nsis-common"], cwd=debs)
    except BuildError:
        # Package lists can be missing or stale (CI images often drop them): refresh a private copy.
        log("installer", "apt-get download failed; refreshing package lists in build/cache/apt")
        lists = CACHE / "apt" / "lists"
        (lists / "partial").mkdir(parents=True, exist_ok=True)
        (CACHE / "apt" / "cache" / "archives" / "partial").mkdir(parents=True, exist_ok=True)
        options = ["-o", f"Dir::State::Lists={lists}", "-o", f"Dir::Cache={CACHE / 'apt' / 'cache'}", "-o", "Debug::NoLocking=1"]
        # Non-fatal: third-party sources whose keyrings only root can read fail, the distribution's lists still arrive.
        if run("installer", ["apt-get", *options, "update"], check=False) != 0:
            log("installer", "apt-get update reported errors; trying the download anyway")
        run("installer", ["apt-get", *options, "download", "nsis", "nsis-common"], cwd=debs)
    for deb in sorted(debs.glob("*.deb")):
        run("installer", ["dpkg", "-x", deb, NSIS_ROOT])
    if not makensis.is_file():
        raise BuildError("the nsis packages did not contain usr/bin/makensis")
    return makensis, env


def find_makensis() -> tuple[Path, dict | None]:
    if sys.platform.startswith("linux"):
        return find_makensis_linux()
    found = shutil.which("makensis")
    if found:
        return Path(found), None
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
            candidate = Path(base) / "NSIS" / "makensis.exe" if base else None
            if candidate and candidate.is_file():
                return candidate, None
        raise BuildError(NSIS_HELP_WINDOWS)
    raise BuildError("makensis (NSIS 3) was not found; install NSIS 3 and add it to PATH")


def build_installer(wheel: Path, version: str, uv_exe: Path, icon: Path, license_file: Path, lock_file: Path) -> Path:
    makensis, env = find_makensis()
    for old in DIST.glob("CinewrightSetup-*.exe"):
        old.unlink()
    out = DIST / f"CinewrightSetup-{version}.exe"
    defines = {
        "VERSION": version,
        "WHEEL": wheel,
        "UV_EXE": uv_exe,
        "ICON": icon,
        "LICENSE_FILE": license_file,
        "LOCK_FILE": lock_file,
        "OUTFILE": out,
    }
    run("installer", [makensis, "-V3", "-WX", "-INPUTCHARSET", "UTF8", *(f"-D{k}={v}" for k, v in defines.items()), NSI], env=env)
    if not out.is_file():
        raise BuildError(f"makensis did not produce {out.relative_to(ROOT)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Cinewright wheel and Windows installer.")
    parser.add_argument("--skip-web", action="store_true", help="do not build the web UI (dev only; the wheel takes web_dist as it is)")
    parser.add_argument("--skip-installer", action="store_true", help="stage uv.exe, lock, icon and licence but do not run makensis")
    args = parser.parse_args(argv)
    try:
        pins = load_pins()
        if args.skip_web:
            log("web", "skipped (--skip-web)")
        else:
            build_web()
        wheel, version = build_wheel(expect_web=not args.skip_web)
        uv_exe = stage_uv(pins)
        lock_file = stage_lock()
        icon = stage_icon()
        license_file = stage_license()
        if args.skip_installer:
            log("installer", "skipped (--skip-installer)")
            return 0
        out = build_installer(wheel, version, uv_exe, icon, license_file, lock_file)
        log("installer", f"{out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")
    except BuildError as exc:
        print(f"[build] error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
