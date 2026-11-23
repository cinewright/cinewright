"""Render the Windows icon (cinewright.ico) from the logo SVGs.

The 16 px image comes from design/logos/cinewright-favicon.svg (drawn on a 16-unit grid so the
perforations land on whole pixels); 24 to 256 px come from design/logos/cinewright-mark.svg.
The geometry is read from the SVG path data, not redrawn: the path is flattened (arcs included),
filled with the even-odd rule by sampling a supersampled grid at pixel centres, and box-filtered
down so every output pixel's alpha is its exact area coverage.

Usage: python installer/make_icon.py [output.ico]
"""

from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOGOS = ROOT / "design" / "logos"
FAVICON_SVG = LOGOS / "cinewright-favicon.svg"
MARK_SVG = LOGOS / "cinewright-mark.svg"
DEFAULT_OUT = ROOT / "build" / "installer" / "cinewright.ico"

FAVICON_SIZES = (16,)
MARK_SIZES = (24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 16

_SVG_NS = "{http://www.w3.org/2000/svg}"
_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "A": 7, "Z": 0}

Point = tuple[float, float]


class Logo:
    """A single-path, single-colour SVG logo: view box, fill colour and subpath outlines."""

    def __init__(self, path: Path) -> None:
        root = ET.parse(path).getroot()  # noqa: S314 - the repository's own logo files
        min_x, min_y, width, height = (float(v) for v in root.attrib["viewBox"].split())
        if width != height:
            raise ValueError(f"{path.name}: view box must be square")
        paths = root.findall(f"{_SVG_NS}path")
        if len(paths) != 1:
            raise ValueError(f"{path.name}: expected exactly one <path>")
        element = paths[0]
        if element.get("fill-rule") != "evenodd":
            raise ValueError(f"{path.name}: expected fill-rule=evenodd")
        if element.get("transform"):
            raise ValueError(f"{path.name}: transforms are not supported")
        self.origin: Point = (min_x, min_y)
        self.size = width
        self.color = element.attrib["fill"].upper()
        self.d = element.attrib["d"]


def parse_path(d: str) -> list[list[tuple]]:
    """Split absolute path data (M, L, H, V, A, Z) into subpaths of drawing commands.

    Each subpath starts at ("M", x, y); lines are ("L", x, y); arcs keep their seven arguments.
    Relative commands and curves are rejected so a changed logo fails loudly instead of rendering wrong.
    """
    tokens = _TOKEN.findall(d)
    subpaths: list[list[tuple]] = []
    pos = 0
    command = ""
    current: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    while pos < len(tokens):
        if tokens[pos].isalpha():
            command = tokens[pos]
            pos += 1
            if command not in _ARGS:
                raise ValueError(f"unsupported path command {command!r}")
        elif not command or command == "Z":
            raise ValueError(f"number without command at token {pos}")
        count = _ARGS[command]
        args = [float(t) for t in tokens[pos : pos + count]]
        if len(args) != count:
            raise ValueError(f"command {command} needs {count} arguments")
        pos += count
        if command == "M":
            current = start = (args[0], args[1])
            subpaths.append([("M", *current)])
            command = "L"  # further coordinate pairs are implicit line-tos
            continue
        if not subpaths:
            raise ValueError("path data must start with M")
        if command == "Z":
            subpaths[-1].append(("L", *start))
            current = start
            continue
        if command == "L":
            current = (args[0], args[1])
        elif command == "H":
            current = (args[0], current[1])
        elif command == "V":
            current = (current[0], args[0])
        else:
            subpaths[-1].append(("A", *args))
            current = (args[5], args[6])
            continue
        subpaths[-1].append(("L", *current))
    return subpaths


def _arc_points(p0: Point, rx: float, ry: float, phi_deg: float, large: bool, sweep: bool, p1: Point, tolerance: float) -> list[Point]:
    """Flatten an SVG elliptical arc (endpoint parameterisation, SVG 1.1 F.6.5) into points after p0."""
    if p0 == p1:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [p1]
    phi = math.radians(phi_deg)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    scale = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if scale > 1:
        rx, ry = rx * math.sqrt(scale), ry * math.sqrt(scale)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den))
    if large == sweep:
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (p0[0] + p1[0]) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (p0[1] + p1[1]) / 2
    theta1 = math.atan2((y1p - cyp) / ry, (x1p - cxp) / rx)
    theta2 = math.atan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
    delta = theta2 - theta1
    if sweep and delta < 0:
        delta += 2 * math.pi
    elif not sweep and delta > 0:
        delta -= 2 * math.pi
    radius = max(rx, ry)
    step = 2 * math.acos(max(-1.0, 1 - tolerance / radius)) if tolerance < radius else math.pi / 2
    segments = max(2, math.ceil(abs(delta) / step))
    points = []
    for i in range(1, segments):
        t = theta1 + delta * i / segments
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        points.append((cos_phi * ex - sin_phi * ey + cx, sin_phi * ex + cos_phi * ey + cy))
    points.append(p1)
    return points


def flatten(subpaths: list[list[tuple]], tolerance: float) -> list[list[Point]]:
    """Turn parsed subpaths into closed polygons (lists of points)."""
    polygons = []
    for subpath in subpaths:
        _, x, y = subpath[0]
        points: list[Point] = [(x, y)]
        for segment in subpath[1:]:
            if segment[0] == "L":
                points.append((segment[1], segment[2]))
            else:
                rx, ry, phi, large, sweep, x1, y1 = segment[1:]
                points.extend(_arc_points(points[-1], rx, ry, phi, bool(large), bool(sweep), (x1, y1), tolerance))
        polygons.append(points)
    return polygons


def coverage_mask(polygons: list[list[Point]], grid: int, units: float, origin: Point = (0.0, 0.0)) -> bytes:
    """Even-odd fill on a grid x grid raster covering `units` x `units` user units; 255 inside, 0 outside.

    A raster cell is inside when its centre is; edges use the half-open rule, so shapes that share
    an edge neither overlap nor leave a gap.
    """
    k = grid / units
    edges = []
    for poly in polygons:
        for (xa, ya), (xb, yb) in zip(poly, poly[1:] + poly[:1], strict=True):
            if ya != yb:
                edges.append(((xa - origin[0]) * k, (ya - origin[1]) * k, (xb - origin[0]) * k, (yb - origin[1]) * k))
    raster = bytearray(grid * grid)
    for row in range(grid):
        yc = row + 0.5
        crossings = sorted(xa + (yc - ya) * (xb - xa) / (yb - ya) for xa, ya, xb, yb in edges if (ya <= yc < yb) or (yb <= yc < ya))
        for left, right in zip(crossings[0::2], crossings[1::2], strict=True):
            first = max(0, math.ceil(left - 0.5))
            last = min(grid, math.ceil(right - 0.5))
            if last > first:
                raster[row * grid + first : row * grid + last] = b"\xff" * (last - first)
    return bytes(raster)


def render(logo: Logo, size: int, supersample: int = SUPERSAMPLE) -> Image.Image:
    """Render the logo as a size x size RGBA image in its fill colour with anti-aliased alpha."""
    grid = size * supersample
    tolerance = logo.size / grid / 8  # flattening error below 1/8 of a supersampled cell
    polygons = flatten(parse_path(logo.d), tolerance)
    mask = Image.frombytes("L", (grid, grid), coverage_mask(polygons, grid, logo.size, logo.origin))
    alpha = mask.resize((size, size), Image.Resampling.BOX) if supersample > 1 else mask
    color = tuple(int(logo.color[i : i + 2], 16) for i in (1, 3, 5))
    image = Image.new("RGBA", (size, size), (*color, 0))
    image.putalpha(alpha)
    return image


def build_icon(out: Path = DEFAULT_OUT) -> Path:
    """Write the multi-size ICO and return its path."""
    favicon, mark = Logo(FAVICON_SVG), Logo(MARK_SVG)
    images = [render(favicon, s) for s in FAVICON_SIZES] + [render(mark, s) for s in MARK_SIZES]
    images.sort(key=lambda im: im.width, reverse=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, format="ICO", sizes=[im.size for im in images], append_images=images[1:])
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    print(f"wrote {build_icon(target)}")
