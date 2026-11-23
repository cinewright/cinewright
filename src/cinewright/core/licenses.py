"""Licence notices shown in the UI: Cinewright itself, the WanGP disclosure and the core's Python dependencies."""

from __future__ import annotations

import re
from importlib import metadata

from cinewright.api.schemas import LicenseNotice, LicenseNotices

CINEWRIGHT_LICENSE_URL = "https://github.com/cinewright/cinewright/blob/main/LICENSE"

WANGP_NAME = "WanGP (Wan2GP)"
WANGP_LICENSE = "WanGP Community License 2.0"
WANGP_LICENSE_URL = "https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt"
# WanGP's licence requires integrations to disclose its use in UI and docs; use this text verbatim.
WANGP_NOTICE = (
    "Cinewright uses WanGP (Wan2GP) by deepbeepmeep as its first generation engine. WanGP is source-available under the "
    "WanGP Community License 2.0 and is downloaded from its official repository during setup; it is not part of Cinewright."
)

_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _dependency_license(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    for key in ("License-Expression", "License"):
        value = (meta.get(key) or "").strip()
        if value and "\n" not in value and len(value) <= 80:
            return value
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License :: ")]
    return ", ".join(classifiers) or "unknown"


def dependency_notices() -> list[LicenseNotice]:
    try:
        requirements = metadata.requires("cinewright") or []
    except metadata.PackageNotFoundError:
        return []
    notices = []
    for requirement in requirements:
        if "extra ==" in requirement:
            continue
        match = _REQUIREMENT_NAME.match(requirement)
        if not match:
            continue
        try:
            dist = metadata.distribution(match.group(0))
        except metadata.PackageNotFoundError:
            continue
        notices.append(
            LicenseNotice(component=dist.metadata["Name"], license=_dependency_license(dist), url=dist.metadata.get("Home-page"))
        )
    return sorted(notices, key=lambda n: n.component.lower())


def license_notices() -> LicenseNotices:
    return LicenseNotices(
        items=[
            LicenseNotice(component="Cinewright", license="MIT", url=CINEWRIGHT_LICENSE_URL),
            LicenseNotice(component=WANGP_NAME, license=WANGP_LICENSE, url=WANGP_LICENSE_URL, notice=WANGP_NOTICE),
            *dependency_notices(),
        ]
    )
