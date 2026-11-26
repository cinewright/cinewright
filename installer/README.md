# Windows installer

`python scripts/build.py` builds `dist/CinewrightSetup-<version>.exe`. It runs on Linux (CI) and on Windows. The script works through six steps:

1. Builds the web UI (`npm ci`, `npm run build` in `web/`, which needs Node.js `^20.19.0 || >=22.12.0`, the range in `web/package.json`). The output goes to `src/cinewright/web_dist`.
2. Builds the wheel (`python -m build --wheel`). The build fails if the web UI is missing from it.
3. Downloads `uv.exe` for Windows at the version pinned in `pins.json`, checks its sha256 and caches it in `build/cache`.
4. Generates `build/installer/requirements.lock` with `python -m uv pip compile pyproject.toml` for Windows (`--python-version 3.12 --python-platform x86_64-pc-windows-msvc --generate-hashes --only-binary :all:`), runtime dependencies only. The `uv` Python package comes with the `dev` extra and needs network access to PyPI.
5. Renders `cinewright.ico` from `design/logos/cinewright-favicon.svg` (16 px) and `cinewright-mark.svg` (24 to 256 px) with `make_icon.py`.
6. Compiles `cinewright.nsi` with `makensis`. If `makensis` is not on PATH on Linux, the script installs the `nsis` package with `apt-get` when that needs no password (root, or `sudo -n` as on GitHub's runners). Otherwise it unpacks the distribution's `nsis` packages into `build/nsis` with `apt-get download` and `dpkg -x`, which needs no root. On Windows, install NSIS 3 instead.

For development, `--skip-web` leaves out the web UI and `--skip-installer` stops before `makensis`.

## What the installer does

- It installs per user without administrator rights, into `%LOCALAPPDATA%\Programs\Cinewright`. It supports `/S` (silent) and `/D=<folder>`.
- It copies `uv.exe`, the wheel, `requirements.lock`, the licence text and the icon.
- It then runs `uv venv --python 3.12 --managed-python runtime` then `uv pip install --require-hashes -r requirements.lock` (every package pinned by sha256) and `uv pip install --no-deps` for the wheel. Python lives in `python\` and uv's cache in `cache\`, both inside the installation folder. Setup therefore needs an internet connection.
- It creates a Start menu shortcut, and optionally a desktop shortcut, to `runtime\Scripts\pythonw.exe -m cinewright`. It also adds an Add/Remove Programs entry under HKCU.
- Before they change any file, setup and the uninstaller close a running Cinewright: they stop every process whose program lies in the installation folder (the runtime's launcher and the Python interpreter running the core) and wait until those have ended. The worker runs from the data folder and ends together with the core. If a process is still running after 80 checks, half a second apart, they stop with a message and change nothing. Silent mode (`/S`) does the same without asking.
- A newer installer replaces the runtime and the wheel. If the upgrade fails, the previous runtime is restored.
- The uninstaller first checks that the registry's `InstallLocation` equals its own folder and that the folder holds `cinewright.ico`; otherwise it deletes nothing and stops.
- The uninstaller removes everything setup created, plus the shortcuts and the registry entry. It asks before deleting the user data in `%LOCALAPPDATA%\Cinewright`. The default answer, and the only behaviour in silent mode, is to keep the data.

## Pins and licences

- `pins.json` pins the uv release by version and by the sha256 from the release's published `.sha256` file. Only pick releases that are at least two weeks old.
- The licence page shows Cinewright's `LICENSE` followed by `THIRD-PARTY-NOTICES.txt`. The notices cover everything the installer ships or sets up: the WanGP disclosure (downloaded during setup, not bundled), the web UI components (React, react-dom, scheduler, lucide-react, Geist and Geist Mono) with their licence texts, uv (MIT or Apache-2.0, redistributed under MIT), and CPython from python-build-standalone plus the locked Python packages, which are downloaded during setup.
