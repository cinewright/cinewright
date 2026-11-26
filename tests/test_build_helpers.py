"""Tests for the helpers in scripts/build.py (no network, no makensis)."""

import hashlib
import importlib.util
import io
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WANGP_NOTICE = (
    "Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is source-available under the "
    "WanGP Community License 2.0 and is downloaded from its official repository during setup; it is not part of Cinewright."
)


def _load_build():
    spec = importlib.util.spec_from_file_location("cinewright_build_script", ROOT / "scripts" / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build()


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("cinewright-0.0.0-py3-none-any.whl", "0.0.0"),
        ("cinewright-1.2.3rc1-py3-none-any.whl", "1.2.3rc1"),
        ("cinewright-0.1.0.dev4+g1a2b3c-py3-none-any.whl", "0.1.0.dev4+g1a2b3c"),
    ],
)
def test_wheel_version(name, version):
    assert build.wheel_version(name) == version


@pytest.mark.parametrize(
    "name",
    [
        "cinewright-0.0.0-cp312-cp312-win_amd64.whl",
        "cinewright_worker-0.0.0-py3-none-any.whl",
        "cinewright-0.0.0.tar.gz",
        "cinewright-0.0.0-1-py3-none-any.whl",
    ],
)
def test_wheel_version_rejects_other_files(name):
    with pytest.raises(build.BuildError):
        build.wheel_version(name)


def test_verify_sha256(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"cinewright")
    build.verify_sha256(payload, hashlib.sha256(b"cinewright").hexdigest())
    with pytest.raises(build.ChecksumError):
        build.verify_sha256(payload, hashlib.sha256(b"something else").hexdigest())


def test_download_with_wrong_checksum_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"tampered"))
    dest = tmp_path / "uv.zip"
    with pytest.raises(build.ChecksumError):
        build.download("https://example.invalid/uv.zip", dest, hashlib.sha256(b"original").hexdigest())
    assert list(tmp_path.iterdir()) == []


def test_download_keeps_matching_file(tmp_path, monkeypatch):
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"original"))
    dest = tmp_path / "uv.zip"
    build.download("https://example.invalid/uv.zip", dest, hashlib.sha256(b"original").hexdigest())
    assert dest.read_bytes() == b"original"


def test_download_refuses_plain_http(tmp_path):
    with pytest.raises(build.BuildError):
        build.download("http://example.invalid/uv.zip", tmp_path / "uv.zip", "0" * 64)


def test_pins_file_is_valid():
    uv = build.load_pins(build.PINS)["uv"]
    assert uv["url"].endswith(f"/{uv['version']}/uv-x86_64-pc-windows-msvc.zip")
    assert re.fullmatch(r"[0-9a-f]{64}", uv["sha256"])


def _valid_pins():
    return {
        "uv": {
            "version": "0.13.7",
            "published": "2026-11-03",
            "asset": "uv-x86_64-pc-windows-msvc.zip",
            "url": "https://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-pc-windows-msvc.zip",
            "sha256": "a" * 64,
        }
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("uv"),
        lambda p: p.update(extra={}),
        lambda p: p["uv"].pop("sha256"),
        lambda p: p["uv"].update(sha256="A" * 64),
        lambda p: p["uv"].update(sha256="a" * 63),
        lambda p: p["uv"].update(version="latest"),
        lambda p: p["uv"].update(published="Nov 3"),
        lambda p: p["uv"].update(asset="uv-x86_64-unknown-linux-gnu.tar.gz"),
        lambda p: p["uv"].update(url="http://github.com/astral-sh/uv/releases/download/0.13.7/uv-x86_64-pc-windows-msvc.zip"),
        lambda p: p["uv"].update(url="https://example.com/0.13.7/uv-x86_64-pc-windows-msvc.zip"),
        lambda p: p["uv"].update(url="https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-pc-windows-msvc.zip"),
        lambda p: p["uv"].update(extra="x"),
    ],
)
def test_pins_schema_rejects(tmp_path, mutate):
    pins = _valid_pins()
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(pins))
    build.load_pins(path)
    mutate(pins)
    path.write_text(json.dumps(pins))
    with pytest.raises(build.BuildError):
        build.load_pins(path)


def test_parse_node_version():
    assert build.parse_node_version("v22.14.0\n") == (22, 14, 0)
    assert build.parse_node_version("v20.19.1") == (20, 19, 1)
    with pytest.raises(build.BuildError):
        build.parse_node_version("node: command not found")


def test_license_page_text():
    data = build.compose_license(build.LICENSE.read_text(encoding="utf-8"), build.NOTICES.read_text(encoding="utf-8"))
    text = data.decode("ascii")
    assert text.startswith("MIT License")
    assert "\n" not in text.replace("\r\n", "")
    flat = " ".join(text.split())
    assert WANGP_NOTICE in flat
    assert "https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt" in flat
    assert "Copyright (c) 2025 Astral Software Inc." in flat


def test_license_page_must_be_ascii():
    with pytest.raises(build.BuildError):
        build.compose_license("MIT License ©", "notices")


def test_installer_step_passes_every_required_define(tmp_path, monkeypatch):
    required = set(re.findall(r"^!insertmacro REQUIRE_DEFINE (\w+)$", build.NSI.read_text(encoding="utf-8"), re.M))
    assert required == {"VERSION", "WHEEL", "UV_EXE", "LOCK_FILE", "ICON", "LICENSE_FILE"}
    calls = []

    def fake_run(step, cmd, cwd=build.ROOT, env=None):
        calls.append([str(c) for c in cmd])
        (tmp_path / "CinewrightSetup-1.2.3.exe").write_bytes(b"MZ")

    monkeypatch.setattr(build, "DIST", tmp_path)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "find_makensis", lambda: (Path("makensis"), None))
    monkeypatch.setattr(build, "run", fake_run)
    (tmp_path / "CinewrightSetup-0.9.0.exe").write_bytes(b"stale")
    out = build.build_installer(Path("w.whl"), "1.2.3", Path("uv.exe"), Path("i.ico"), Path("L.txt"), Path("r.lock"))
    assert out == tmp_path / "CinewrightSetup-1.2.3.exe"
    assert not (tmp_path / "CinewrightSetup-0.9.0.exe").exists()
    defines = {arg[2:].split("=", 1)[0]: arg.split("=", 1)[1] for arg in calls[0] if arg.startswith("-D")}
    assert required <= set(defines)
    assert defines["VERSION"] == "1.2.3"
    assert defines["OUTFILE"] == str(out)
    assert calls[0][-1] == str(build.NSI)


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ((18, 20, 0), False),
        ((20, 18, 3), False),
        ((20, 19, 0), True),
        ((20, 20, 1), True),
        ((21, 7, 3), False),
        ((22, 11, 0), False),
        ((22, 12, 0), True),
        ((22, 14, 0), True),
        ((24, 0, 0), True),
    ],
)
def test_node_version_gate_matches_the_web_engines_field(version, supported):
    engines = json.loads((ROOT / "web" / "package.json").read_text(encoding="utf-8"))["engines"]["node"]
    assert engines == build.NODE_RANGE
    assert build.node_supported(version) is supported


class FakeLinux:
    """Fakes PATH lookups, `sudo -n true` and build commands for the makensis lookup on Linux."""

    def __init__(self, monkeypatch, tmp_path, *, on_path=(), sudo_ok=False, euid=1000, install_ok=True):
        self.on_path = {name: f"/usr/bin/{name}" for name in on_path}
        self.commands = []
        self.install_ok = install_ok
        monkeypatch.setattr(build, "NSIS_ROOT", tmp_path / "nsis")
        monkeypatch.setattr(build.shutil, "which", lambda name: self.on_path.get(name))
        monkeypatch.setattr(build.os, "geteuid", lambda: euid, raising=False)
        monkeypatch.setattr(build.subprocess, "run", lambda cmd, **kw: build.subprocess.CompletedProcess(cmd, 0 if sudo_ok else 1))
        monkeypatch.setattr(build, "run", self.run)
        monkeypatch.setattr(build, "fetch_nsis_linux", lambda: ("rootless", {}))

    def run(self, step, cmd, cwd=build.ROOT, env=None, check=True):
        self.commands.append(" ".join(str(part) for part in cmd))
        if cmd[-1] == "nsis" and "install" in cmd:
            if not self.install_ok:
                return 100
            self.on_path["makensis"] = "/usr/bin/makensis"
        return 0


def test_makensis_on_path_is_used_as_is(monkeypatch, tmp_path):
    fake = FakeLinux(monkeypatch, tmp_path, on_path=("makensis", "apt-get", "sudo"), sudo_ok=True)
    assert build.find_makensis_linux() == (Path("/usr/bin/makensis"), None)
    assert fake.commands == []


def test_makensis_is_installed_with_passwordless_sudo(monkeypatch, tmp_path):
    fake = FakeLinux(monkeypatch, tmp_path, on_path=("apt-get", "sudo"), sudo_ok=True)
    assert build.find_makensis_linux() == (Path("/usr/bin/makensis"), None)
    assert fake.commands == ["sudo -n apt-get update -qq", "sudo -n apt-get install -y -qq nsis"]


def test_makensis_is_installed_directly_as_root(monkeypatch, tmp_path):
    fake = FakeLinux(monkeypatch, tmp_path, on_path=("apt-get",), euid=0)
    assert build.find_makensis_linux() == (Path("/usr/bin/makensis"), None)
    assert fake.commands == ["apt-get update -qq", "apt-get install -y -qq nsis"]


@pytest.mark.parametrize(
    ("on_path", "sudo_ok", "install_ok"),
    [
        (("apt-get", "sudo"), False, True),  # sudo wants a password
        (("apt-get",), False, True),  # no sudo
        (("apt-get", "sudo"), True, False),  # the install fails
    ],
)
def test_makensis_falls_back_to_the_rootless_unpack(monkeypatch, tmp_path, on_path, sudo_ok, install_ok):
    fake = FakeLinux(monkeypatch, tmp_path, on_path=on_path, sudo_ok=sudo_ok, install_ok=install_ok)
    assert build.find_makensis_linux() == ("rootless", {})
    assert all(command.endswith(("update -qq", "install -y -qq nsis")) for command in fake.commands)


def test_an_earlier_rootless_unpack_is_reused_without_sudo(monkeypatch, tmp_path):
    fake = FakeLinux(monkeypatch, tmp_path, on_path=("apt-get", "sudo"), sudo_ok=True)
    unpacked = tmp_path / "nsis" / "usr" / "bin" / "makensis"
    unpacked.parent.mkdir(parents=True)
    unpacked.write_bytes(b"")
    assert build.find_makensis_linux() == ("rootless", {})
    assert fake.commands == []


# Hash-locked dependencies (installer lock step)

GOOD_LOCK = """anyio==4.16.0 \\
    --hash=sha256:aa \\
    --hash=sha256:bb
    # via fastapi
idna==3.20 \\
    --hash=sha256:cc
    # via anyio
"""


def test_lock_step_runs_uv_compile_for_windows_without_dev_extras(tmp_path, monkeypatch):
    calls = []

    def fake_subprocess_run(args, cwd=None, env=None, check=False):
        calls.append(args)
        Path(args[args.index("-o") + 1]).write_text(GOOD_LOCK, encoding="utf-8")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(build, "STAGE", tmp_path)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build.subprocess, "run", fake_subprocess_run)
    lock = build.stage_lock()
    assert lock == tmp_path / "requirements.lock"
    (args,) = calls
    assert args[1:6] == ["-m", "uv", "pip", "compile", str(build.PYPROJECT)]
    assert args[args.index("--python-version") + 1] == "3.12"
    assert args[args.index("--python-platform") + 1] == "x86_64-pc-windows-msvc"
    assert "--generate-hashes" in args
    assert "--extra" not in args and "--all-extras" not in args


def test_lock_step_fails_when_uv_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "STAGE", tmp_path)
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 2))
    with pytest.raises(build.BuildError, match="uv"):
        build.stage_lock()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# only a comment\n",
        "anyio==4.16.0\n    # via fastapi\n",
        "anyio>=4 \\\n    --hash=sha256:aa\n",
    ],
)
def test_lock_check_rejects_unpinned_or_unhashed(text):
    with pytest.raises(build.BuildError):
        build.check_lock(text)


def test_lock_check_accepts_a_hashed_lock():
    build.check_lock(GOOD_LOCK)


def test_installer_script_installs_the_lock_with_hashes_then_the_wheel_without_dependencies():
    nsi = build.NSI.read_text(encoding="utf-8")
    assert nsi.index("--require-hashes") < nsi.index("--no-deps")
    assert '-r "$INSTDIR\\requirements.lock"' in nsi
    assert 'File "/oname=requirements.lock" "${LOCK_FILE}"' in nsi


# Uninstaller guard


def test_uninstaller_checks_the_registered_install_location_before_any_delete():
    nsi = build.NSI.read_text(encoding="utf-8")
    section = nsi[nsi.index('Section "Uninstall"') :]
    guard = section.index("ReadRegStr $R0 HKCU")
    assert '"$R0" != "$INSTDIR"' in section
    assert '${FileExists} "$INSTDIR\\cinewright.ico"' in section
    assert section.index("Abort") > guard
    assert guard < min(section.index("RMDir"), section.index("Delete "), section.index("Rename"), section.index("RMDIR_TREE"))


# A running Cinewright is closed first (#106)

FILE_OPERATION = re.compile(
    r"^\s*(File|Delete|RMDir|Rename|SetOutPath|WriteUninstaller|CopyFiles|CreateDirectory|!insertmacro RMDIR_TREE)\b"
)


def nsi_section_lines(name: str) -> list[str]:
    """The lines of `Section "<name>"` up to its SectionEnd, without comments and blank lines."""
    nsi = build.NSI.read_text(encoding="utf-8")
    start = nsi.index(f'Section "{name}"')
    lines = nsi[start : nsi.index("SectionEnd", start)].splitlines()
    return [line for line in lines if line.strip() and not line.lstrip().startswith((";", "#"))]


def stop_macro_command() -> tuple[str, str]:
    """The STOP_RUNNING_CINEWRIGHT macro and the command line its nsExec call runs, as NSIS writes it."""
    nsi = build.NSI.read_text(encoding="utf-8")
    start = nsi.index("!macro STOP_RUNNING_CINEWRIGHT")
    macro = nsi[start : nsi.index("!macroend", start)]
    return macro, re.search(r"nsExec::ExecToLog `(.*)`", macro).group(1)


@pytest.mark.parametrize(
    ("section", "guard"),
    [("Cinewright", "Call ValidateInstallDir"), ("Uninstall", 'ReadRegStr $R0 HKCU "${UNINST_KEY}"')],
)
def test_installer_closes_a_running_cinewright_after_its_guard_and_before_the_first_file_operation(section, guard):
    lines = nsi_section_lines(section)
    guard_at = next(i for i, line in enumerate(lines) if guard in line)
    stop_at = next(i for i, line in enumerate(lines) if line.strip().startswith("!insertmacro STOP_RUNNING_CINEWRIGHT "))
    first_file_operation = next(i for i, line in enumerate(lines) if FILE_OPERATION.match(line))
    assert guard_at < stop_at < first_file_operation, lines[first_file_operation]


def test_stop_macro_stops_every_process_under_the_installation_folder_and_waits():
    macro, command = stop_macro_command()
    script = command.replace("$$", "$")
    assert command.startswith('"$SYSDIR\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -NonInteractive -Command "')
    # The folder arrives through the environment, set from a register: no path is ever quoted into a command.
    set_folder = macro.index('StrCpy $1 "$INSTDIR"')
    assert set_folder < macro.index("System::Call 'kernel32::SetEnvironmentVariable(t \"CINEWRIGHT_STOP_DIR\", t r1)'")
    assert "$INSTDIR" not in command
    assert "$d=([string]$env:CINEWRIGHT_STOP_DIR).TrimEnd('\\'); if($d.Length -lt 3){exit 2}; $d+='\\';" in script
    assert "Get-CimInstance Win32_Process" in script
    assert "$_.ExecutablePath.StartsWith($d,[StringComparison]::OrdinalIgnoreCase)" in script  # -like would read [ ] as wildcards
    assert "$_.ProcessId -ne $me" in script  # an uninstaller run in place lies in the folder itself
    assert "Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue" in script
    assert "if(-not (Find-Mine)){exit 0}" in script and "exit 3}" in script
    # A process list that cannot be read must not pass for "nothing runs": Get-CimInstance's errors do not stop a
    # script by themselves, so Find-Mine would return nothing and the loop would exit 0.
    assert "-Command \"$ErrorActionPreference='Stop'; try{$d=" in script
    assert script.endswith("}catch{'Could not list processes: '+$_; exit 4}\"")
    assert "${If} $0 == 3" in macro and "Abort" in macro
    assert "${ElseIf} $0 != 0" in macro and "Could not check whether Cinewright is running" in macro
    assert '"' not in script[script.index('-Command "') + len('-Command "') : -1]  # one quoted argument for PowerShell


def test_stop_macro_command_is_valid_nsis_and_fits_its_string_limit():
    _, command = stop_macro_command()
    assert "`" not in command  # the NSIS string is quoted with backticks
    assert re.sub(r"\$\$|\$SYSDIR", "", command).count("$") == 0  # every PowerShell $ is escaped, or makensis -WX fails
    expanded = command.replace("$$", "$").replace("$SYSDIR", "C:\\Windows\\SysWOW64")
    assert len(expanded) < 1024  # NSIS_MAX_STRLEN of a standard NSIS build


def test_installer_notices_list_everything_the_installer_ships():
    flat = " ".join(build.NOTICES.read_text(encoding="utf-8").split())
    for needle in (
        "react 18.3.1",
        "react-dom 18.3.1",
        "scheduler 0.23.2",
        "lucide-react 1.70.0",
        "SIL Open Font License",
        "Geist Mono",
        "uv (https://github.com/astral-sh/uv)",
        "python-build-standalone",
        "WanGP",
    ):
        assert needle in flat, needle
