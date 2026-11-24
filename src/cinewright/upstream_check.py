"""Wan2GP upstream check: is a newer upstream commit eligible for a pin update, and what changed since the pin?

Maintainer tool behind `python -m cinewright.devtools upstream-check [--json]`; the core never imports it. It reads
the `wan2gp` pin from `registry/engines.yaml`, asks the GitHub REST API which commits on the default branch follow
the pinned commit, and picks the newest one whose committer date is at least 7 days old (the specification's
quarantine). For that candidate it reports the files changed since the pin and flags what a reviewer must read.
Committer dates are set upstream, so it also lists the commits up to the candidate that were committed, or merged
into the branch, after the cutoff: the quarantine does not cover them.

It talks only to api.github.com (redirects included), unauthenticated or with `GITHUB_TOKEN`, and changes no files.
Exit codes: 0 no eligible commit (prints nothing), 10 eligible commit (prints the report), 1 the check failed
(reason on stderr), so a failure never reads as "no eligible commit".

Commit messages, file names and patch lines are upstream data: they are printed truncated, with control characters
escaped, and fenced in the text report. A file whose patch GitHub leaves out (large or binary) or sends with fewer
added lines than it counts, and anything beyond a file list GitHub cut short, counts as not inspected, never as
clean. The downloads and shell flags match patterns in added lines: a "no" means no match, not proof.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import sys
import unicodedata
from collections.abc import Container, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, TextIO
from urllib.parse import quote, urljoin, urlsplit

import httpx

from cinewright.core.engines import EnginePinError, load_engine_pins

ENGINE = "wan2gp"
QUARANTINE = timedelta(days=7)

EXIT_NO_CANDIDATE = 0
EXIT_ELIGIBLE = 10
EXIT_FAILED = 1

API_HOST = "api.github.com"
API_URL = f"https://{API_HOST}"
API_VERSION = "2022-11-28"
COMPARE_FILE_LIMIT = 300  # a compare lists at most 300 files and does not say when it cut the list
PAGE_SIZE = 100
MAX_PAGES = 100
MAX_REDIRECTS = 5
TIMEOUT_S = 30.0

LICENCE_FILE = "LICENSE.txt"  # Wan2GP's licence file at the pinned commit
MESSAGE_CHARS = 500
MESSAGE_LINES = 10
LINE_CHARS = 200
PATH_CHARS = 300
LINES_PER_FLAG = 50

FLAGS = {
    "requirements": "requirements or setup files changed",
    "downloads": "new URLs or download code in added lines",
    "shell": "shell, eval, exec or base64 in added lines",
    "licence": f"licence file {LICENCE_FILE} changed",
}
_STATE_TEXT = {"yes": "yes", "no": "no", "not_inspected": "not inspected"}
_CONTENT_PATTERNS = {
    "downloads": re.compile(
        r"(?i:https?://)"
        r"|\b(?:import|from)\s+requests\b|^\s*(?:import|from)\b[^#]*\brequests\b|\brequests\.\w"
        r"|\burllib\d?\b"
        r"|\bhuggingface_hub\b|\bhf_hub_download\b|\bsnapshot_download\b"
    ),
    "shell": re.compile(
        r"\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*)\b|\bshell\s*=\s*True\b|\bchild_process\b"
        r"|^\s*from\s+os\s+import\b[^#]*\b(?:system|popen|exec\w*|spawn\w*)\b"
        r"|(?<![\w.])eval\s*\(|(?<![\w.])exec\s*\("  # not model.eval() or regex.exec()
        r"|\bbuiltins\.(?:eval|exec)\s*\(|\b__import__\s*\("
        r"|\bbase64\b|\bb64(?:en|de)code\b|\batob\s*\(|\bbtoa\s*\("
    ),
}
_REQUIREMENT_NAMES = ("requirements*.txt", "setup.py", "setup_config.json")
_INSTALLATION_DIR = "installation/"
_STATUS_LETTERS = {"added": "A", "removed": "D", "modified": "M", "renamed": "R", "copied": "C", "changed": "T", "unchanged": "U"}

_SHA = re.compile(r"[0-9a-f]{40}")  # used with fullmatch, like the next two
_FULL_NAME = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_BRANCH = re.compile(r"[A-Za-z0-9_.-]{1,100}(?:/[A-Za-z0-9_.-]{1,100}){0,9}")  # printed unfenced: no Markdown or HTML
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class UpstreamCheckError(Exception):
    """The check could not finish; the message says why."""


def read_pin(engines_path: Path | None = None) -> tuple[str, str]:
    """`repo` and `commit` of the `wan2gp` entry in engines.yaml."""
    pins = load_engine_pins(engines_path)
    if ENGINE not in pins:
        raise UpstreamCheckError(f"engines.yaml has no {ENGINE!r} entry")
    return pins[ENGINE].repo, pins[ENGINE].commit


def github_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """A client for the GitHub REST API that follows no redirect by itself (`_GitHub` checks each one)."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": API_VERSION, "User-Agent": "cinewright-devtools"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, timeout=TIMEOUT_S, follow_redirects=False, transport=transport)


class _GitHub:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def get(self, target: str, params: dict[str, Any] | None = None) -> tuple[Any, httpx.Response]:
        """GET a JSON document; `target` is an API path or an absolute api.github.com URL."""
        url = API_URL + target if target.startswith("/") else target
        for _ in range(MAX_REDIRECTS + 1):
            _require_api_url(url)
            try:
                response = self._client.get(url, params=params)
            except (httpx.HTTPError, httpx.InvalidURL) as exc:
                raise UpstreamCheckError(f"GitHub API request {_where(url)} failed: {type(exc).__name__}: {exc}") from exc
            if response.has_redirect_location:  # e.g. 301 for a renamed repository; any other 3xx is a failure below
                try:
                    url = urljoin(str(response.url), response.headers["location"])
                except ValueError as exc:
                    raise UpstreamCheckError(f"GitHub API redirected {_where(str(response.url))} to an invalid address") from exc
                continue
            if response.status_code != 200:
                raise UpstreamCheckError(_failure(response, url))
            try:
                return _json(response), response
            except ValueError as exc:
                raise UpstreamCheckError(f"GitHub API answered {_where(url)} with invalid JSON") from exc
        raise UpstreamCheckError(f"GitHub API redirected {_where(target)} more than {MAX_REDIRECTS} times")


def _require_api_url(url: str) -> None:
    try:
        parts = urlsplit(url)
        allowed = parts.scheme == "https" and parts.hostname == API_HOST and parts.port in (None, 443)
        allowed = allowed and not parts.username and not parts.password
    except ValueError:  # e.g. a port out of range or a broken IPv6 address
        allowed = False
    if not allowed:
        raise UpstreamCheckError(f"refusing to request {_clean(url, 300)}: the check only talks to {API_URL}")


def _where(url: str) -> str:
    try:
        return _clean(urlsplit(url).path, 300)
    except ValueError:
        return _clean(url, 300)


def _json(response: httpx.Response) -> Any:
    """The parsed body; ValueError when it is not JSON, also when it nests too deeply for the parser."""
    try:
        return response.json()
    except RecursionError as exc:
        raise ValueError("JSON nested too deeply") from exc


def _failure(response: httpx.Response, url: str) -> str:
    message = ""
    try:
        body = _json(response)
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        message = body["message"]
    if response.status_code in (403, 429) and (response.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in message.lower()):
        when = ""
        try:
            seconds = int(response.headers.get("x-ratelimit-reset", ""))
            if seconds >= 0:
                when = f", resets at {_iso(datetime.fromtimestamp(seconds, UTC))}"
        except (ValueError, OverflowError, OSError):  # missing, not a number, or out of range
            pass
        return f"GitHub API rate limit exceeded{when}; set GITHUB_TOKEN for a higher limit"
    detail = f": {_clean(message, 200)}" if message else ""
    return f"GitHub API answered {response.status_code} for {_where(url)}{detail}"


def _field(data: Any, key: str, kind: type, what: str) -> Any:
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise UpstreamCheckError(f"unexpected GitHub API response: {what} has no {kind.__name__} {key!r}")
    return value


def check_upstream(client: httpx.Client, repo_url: str, pin: str, now: datetime) -> dict[str, Any] | None:
    """The report for the eligible commit after `pin` in `repo_url`, or None when there is none."""
    segments = [segment for segment in urlsplit(repo_url).path.split("/") if segment]
    if len(segments) != 2 or not _SHA.fullmatch(pin):
        raise UpstreamCheckError(f"the pin {repo_url} @ {pin} is not a GitHub repository and a 40-hex commit")
    github = _GitHub(client)

    repository, _ = github.get(f"/repos/{quote(segments[0], safe='')}/{quote(segments[1], safe='')}")
    repo = _field(repository, "full_name", str, "repository")
    if not _FULL_NAME.fullmatch(repo):
        raise UpstreamCheckError(f"unexpected GitHub API response: repository name {_clean(repo, 100)!r}")
    branch = _field(repository, "default_branch", str, "repository")
    if not _BRANCH.fullmatch(branch):
        raise UpstreamCheckError(f"unexpected GitHub API response: default branch name {_clean(branch, 100)!r}")

    commits = _commits_after(github, repo, pin, branch)
    cutoff = now - QUARANTINE
    dates = [_committed_at(commit) for commit in commits]
    eligible = [(when, index) for index, when in enumerate(dates) if when <= cutoff]
    if not eligible:
        return None
    committed_at, chosen = max(eligible)
    sha = _sha(commits[chosen], "commit")
    message = _field(_field(commits[chosen], "commit", dict, "commit"), "message", str, "commit")

    comparison, _ = github.get(f"/repos/{repo}/compare/{pin}...{sha}")
    status = _field(comparison, "status", str, "compare")
    if status != "ahead":
        raise UpstreamCheckError(f"candidate {sha} does not descend from the pinned commit (compare status {_clean(status, 40)!r})")
    files = _field(comparison, "files", list, "compare")
    late = _after_cutoff(commits, dates, sha, cutoff)

    return {
        "engine": ENGINE,
        "pinned_repo": "/".join(segments),
        "repo": repo,
        "default_branch": branch,
        "pinned_commit": pin,
        "quarantine_days": QUARANTINE.days,
        "cutoff": _iso(cutoff),
        "commits_after_pin": len(commits),
        "candidate": {"sha": sha, "committed_at": _iso(committed_at), "message": _clean_message(message)},
        "commits_since_pin": _field(comparison, "ahead_by", int, "compare"),
        "compare_url": f"https://github.com/{repo}/compare/{pin}...{sha}",
        "after_cutoff": late[:LINES_PER_FLAG],
        "after_cutoff_total": len(late),
        **_inspect(files),
    }


def _commits_after(github: _GitHub, repo: str, pin: str, branch: str) -> list[Any]:
    """Every commit on `branch` after `pin`, oldest first, read page by page."""
    data, response = github.get(f"/repos/{repo}/compare/{pin}...{quote(branch, safe='/')}", params={"per_page": PAGE_SIZE})
    status = _field(data, "status", str, "compare")
    if status == "identical":
        return []
    if status != "ahead":
        raise UpstreamCheckError(
            f"the pinned commit {pin} is not an ancestor of {branch} (compare status {_clean(status, 40)!r}); check the pin by hand"
        )
    total = _field(data, "total_commits", int, "compare")
    commits = list(_field(data, "commits", list, "compare"))
    pages = 1
    while next_url := response.links.get("next", {}).get("url"):
        pages += 1
        if pages > MAX_PAGES:
            raise UpstreamCheckError(f"GitHub listed more than {MAX_PAGES} pages of commits after the pin")
        data, response = github.get(next_url)
        commits.extend(_field(data, "commits", list, "compare page"))
    if len(commits) != total:
        raise UpstreamCheckError(f"GitHub listed {len(commits)} of the {total} commits after the pin on {branch}")
    return commits


def _sha(data: Any, what: str) -> str:
    sha = _field(data, "sha", str, what)
    if not _SHA.fullmatch(sha):
        raise UpstreamCheckError(f"unexpected GitHub API response: {what} id {_clean(sha, 100)!r}")
    return sha


def _after_cutoff(commits: list[Any], dates: list[datetime], candidate: str, cutoff: datetime) -> list[dict[str, Any]]:
    """The commits up to `candidate` (those in pin..candidate) that were committed, or reached the branch, after `cutoff`.

    A commit reaches the branch with the oldest commit of the branch's first-parent line whose history holds it: itself,
    or the merge that brought it in. `commits` is the complete list after the pin, so its parents give the whole graph.
    """
    position: dict[str, int] = {}
    parents: dict[str, list[str]] = {}
    for index, commit in enumerate(commits):
        sha = _sha(commit, "commit")
        position[sha] = index
        parents[sha] = [_sha(parent, "parent commit") for parent in _field(commit, "parents", list, "commit")]
    named = {parent for listed in parents.values() for parent in listed}
    tips = [sha for sha in parents if sha not in named]
    if len(tips) != 1:
        raise UpstreamCheckError(f"unexpected GitHub API response: the commits after the pin end in {len(tips)} branch tips, not 1")
    line = [tips[0]]  # the branch's first-parent line after the pin, newest first
    while parents[line[-1]] and parents[line[-1]][0] in parents and len(line) < len(parents):
        line.append(parents[line[-1]][0])
    reached_with: dict[str, str] = {}
    for head in reversed(line):
        for sha in _history(head, parents, reached_with):
            reached_with[sha] = head
    late = []
    for sha in sorted(_history(candidate, parents, ()), key=position.__getitem__):
        merge = reached_with.get(sha, sha)
        committed, reached = dates[position[sha]], dates[position[merge]]
        if committed > cutoff or reached > cutoff:
            merged = merge != sha
            late.append(
                {
                    "sha": sha,
                    "committed_at": _iso(committed),
                    "reached_branch_with": merge if merged else None,
                    "reached_branch_at": _iso(reached) if merged else None,
                }
            )
    return late


def _history(start: str, parents: dict[str, list[str]], skip: Container[str]) -> list[str]:
    """`start` and its ancestors among the commits after the pin, without walking into `skip`."""
    found: list[str] = []
    stack, seen = [start], set()
    while stack:
        sha = stack.pop()
        if sha in seen or sha in skip or sha not in parents:
            continue
        seen.add(sha)
        found.append(sha)
        stack.extend(parents[sha])
    return found


def _committed_at(commit: Any) -> datetime:
    committer = _field(_field(commit, "commit", dict, "commit"), "committer", dict, "commit")
    text = _field(committer, "date", str, "committer")
    try:
        value = datetime.fromisoformat(text)
    except ValueError as exc:
        raise UpstreamCheckError(f"unexpected GitHub API response: committer date {_clean(text, 40)!r}") from exc
    if value.tzinfo is None:
        raise UpstreamCheckError(f"unexpected GitHub API response: committer date {_clean(text, 40)!r} has no time zone")
    try:
        return value.astimezone(UTC)
    except OverflowError as exc:  # e.g. 9999-12-31T23:59:59-05:00 lies past the last time Python can hold in UTC
        raise UpstreamCheckError(f"unexpected GitHub API response: committer date {_clean(text, 40)!r} is out of range") from exc


class _Matches:
    def __init__(self) -> None:
        self.files: dict[str, None] = {}
        self.lines: list[dict[str, Any]] = []
        self.total = 0

    def add(self, shown_path: str, number: int, text: str) -> None:
        self.files[shown_path] = None
        self.total += 1
        if len(self.lines) < LINES_PER_FLAG:
            self.lines.append({"path": shown_path, "line": number, "text": _clean(text.strip(), LINE_CHARS)})


def _inspect(files: list[Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    not_inspected: list[str] = []
    requirement_files: list[str] = []
    licence_files: list[str] = []
    content = {name: _Matches() for name in _CONTENT_PATTERNS}
    for item in files:
        path = _field(item, "filename", str, "changed file")
        status = _field(item, "status", str, "changed file")
        previous = item.get("previous_filename") if isinstance(item.get("previous_filename"), str) else None
        patch = item.get("patch")
        paths = [p for p in (path, previous) if p is not None]
        shown = _clean(path, PATH_CHARS)
        if any(_is_requirements(p) for p in paths):
            requirement_files.append(shown)
        if LICENCE_FILE in paths:
            licence_files.append(shown)
        inspected = False
        if isinstance(patch, str):
            added = list(_added_lines(patch))
            for number, line in added:
                for name, pattern in _CONTENT_PATTERNS.items():
                    if pattern.search(line):
                        content[name].add(shown, number, line)
            expected = _count(item, "additions")
            inspected = expected is None or len(added) >= expected  # fewer added lines than GitHub counts: a cut patch
        if not inspected:
            not_inspected.append(shown)
        entries.append(
            {
                "path": shown,
                "previous_path": None if previous is None else _clean(previous, PATH_CHARS),
                "status": _clean(status, 20),
                "additions": _count(item, "additions") if inspected else None,
                "deletions": _count(item, "deletions") if inspected else None,
                "inspected": inspected,
            }
        )
    truncated = len(files) >= COMPARE_FILE_LIMIT
    all_listed = not truncated
    all_read = all_listed and not not_inspected

    def state(found: bool, complete: bool) -> str:
        return "yes" if found else "no" if complete else "not_inspected"

    flags: dict[str, Any] = {"requirements": {"state": state(bool(requirement_files), all_listed), "files": requirement_files}}
    for name, matches in content.items():
        flags[name] = {
            "state": state(bool(matches.total), all_read),
            "files": list(matches.files),
            "lines": matches.lines,
            "lines_total": matches.total,
        }
    flags["licence"] = {"state": state(bool(licence_files), all_listed), "files": licence_files}
    return {"files": entries, "file_list_truncated": truncated, "not_inspected": not_inspected, "flags": flags}


def _is_requirements(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.startswith(_INSTALLATION_DIR) or any(fnmatch.fnmatchcase(name, pattern) for pattern in _REQUIREMENT_NAMES)


def _count(item: dict[str, Any], key: str) -> int | None:
    value = item.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _added_lines(patch: str) -> Iterator[tuple[int, str]]:
    """(line number in the new file, text) of each added line of a unified diff hunk list."""
    number = 0
    for raw in patch.split("\n"):
        hunk = _HUNK.match(raw)
        if hunk:
            number = int(hunk[1])
        elif raw.startswith("+"):
            yield number, raw[1:].removesuffix("\r")
            number += 1
        elif not raw.startswith(("-", "\\")):
            number += 1


# Upstream text is data: escape what a terminal or a Markdown renderer would act on, cut it short, fence it.


def _clean(text: str, limit: int | None = None, *, multiline: bool = False) -> str:
    truncated = limit is not None and len(text) > limit
    if truncated:
        text = text[:limit]
    chars = []
    for char in text:
        category = unicodedata.category(char)
        if (char == "\n" and multiline) or char == "\t":
            chars.append(char)
        elif category[0] == "C" or category in ("Zl", "Zp"):
            chars.append(f"\\u{ord(char):04x}" if ord(char) <= 0xFFFF else f"\\U{ord(char):08x}")
        else:
            chars.append(char)
    return "".join(chars) + (" [truncated]" if truncated else "")


def _clean_message(message: str) -> str:
    lines = message.replace("\r\n", "\n").rstrip().split("\n")
    shown = _clean("\n".join(lines[:MESSAGE_LINES]), MESSAGE_CHARS, multiline=True)
    if len(lines) > MESSAGE_LINES and not shown.endswith("[truncated]"):
        shown += "\n[truncated]"
    return shown


def _fence(text: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def render_text(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    branch = report["default_branch"]
    lines = [
        f"Wan2GP upstream: commit {candidate['sha']} is eligible for a pin update.",
        f"Repository:        {report['repo']}, branch {branch}",
    ]
    if report["pinned_repo"] != report["repo"]:
        lines.append(f"                   (engines.yaml names {report['pinned_repo']}, which GitHub redirects)")
    late = report["after_cutoff_total"]
    lines += [
        f"Pinned commit:     {report['pinned_commit']}",
        f"Candidate commit:  {candidate['sha']}, committed {candidate['committed_at']}",
        f"Quarantine:        {report['quarantine_days']} days, cutoff {report['cutoff']}",
        f"                   {_plural(late, 'commit')} up to the candidate committed or merged after the cutoff"
        + (" (listed below; the quarantine does not cover them)" if late else ""),
        f"Commits since pin: {report['commits_since_pin']} ({report['commits_after_pin']} on {branch} after the pin)",
        f"Compare:           {report['compare_url']}",
        "",
        "Flags:",
        *(f"  {label}: {_STATE_TEXT[report['flags'][name]['state']]}" for name, label in FLAGS.items()),
        '  (downloads and shell match patterns in added lines: "no" means no match, not proof)',
        "",
        "Candidate commit message (upstream data, truncated):",
        _fence(candidate["message"]),
        "",
    ]
    if late:
        shown = len(report["after_cutoff"])
        more = f", first {shown} shown" if shown < late else ""
        lines.append(f"Committed or merged after the cutoff ({_plural(late, 'commit')}{more}):")
        for commit in report["after_cutoff"]:
            merge = commit["reached_branch_with"]
            via = f", reached {branch} with {merge} committed {commit['reached_branch_at']}" if merge is not None else ""
            lines.append(f"  {commit['sha']} committed {commit['committed_at']}{via}")
        lines.append("")
    files = report["files"]
    cut = " (the GitHub API cut the list here; files beyond it are not inspected)" if report["file_list_truncated"] else ""
    lines.append(f"Files changed: {len(files)}{cut}")
    if files:
        listing = []
        for entry in files:
            letter = _STATUS_LETTERS.get(entry["status"], "?")
            moved = f"{entry['previous_path']} -> " if entry["previous_path"] is not None else ""
            listing.append(f"{letter}  {moved}{entry['path']}")
        lines.append(_fence("\n".join(listing)))
    if report["not_inspected"]:
        count = _plural(len(report["not_inspected"]), "file")
        lines += ["", f"Not inspected (the GitHub API sent no patch, or one with fewer added lines than it counts): {count}"]
        lines.append(_fence("\n".join(report["not_inspected"])))
    for name, label in FLAGS.items():
        flag = report["flags"][name]
        if not flag["files"]:
            continue
        lines.append("")
        if "lines" in flag:
            shown = len(flag["lines"])
            more = f", first {shown} shown" if shown < flag["lines_total"] else ""
            lines.append(f"{label[0].upper()}{label[1:]} ({_plural(flag['lines_total'], 'line')}{more}, path:line):")
            lines.append(_fence("\n".join(f"{line['path']}:{line['line']}: {line['text']}" for line in flag["lines"])))
        else:
            lines.append(f"{label[0].upper()}{label[1:]}:")
            lines.append(_fence("\n".join(flag["files"])))
    return "\n".join(lines) + "\n"


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2) + "\n"


def _write(stream: TextIO, text: str) -> None:
    """Write without failing on characters the stream cannot encode (a cp1252 pipe on Windows)."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(encoding, "backslashreplace").decode(encoding))
    stream.flush()


def run(
    *,
    as_json: bool = False,
    engines_path: Path | None = None,
    transport: httpx.BaseTransport | None = None,
    now: datetime | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the check and print the report; returns the exit code."""
    try:
        repo_url, pin = read_pin(engines_path)
        with github_client(transport) as client:
            report = check_upstream(client, repo_url, pin, now or datetime.now(UTC))
    except (UpstreamCheckError, EnginePinError) as exc:
        _write(stderr or sys.stderr, f"upstream-check failed: {exc}\n")
        return EXIT_FAILED
    if report is None:
        return EXIT_NO_CANDIDATE
    _write(stdout or sys.stdout, render_json(report) if as_json else render_text(report))
    return EXIT_ELIGIBLE
