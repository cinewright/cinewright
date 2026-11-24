"""Tests for installer/make_icon.py: a multi-size ICO rendered from the logo SVGs."""

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ACCENT = (0x6D, 0x28, 0xD9)


def _load_make_icon():
    spec = importlib.util.spec_from_file_location("cinewright_make_icon", ROOT / "installer" / "make_icon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_icon = _load_make_icon()


@pytest.fixture(scope="module")
def icon(tmp_path_factory):
    path = make_icon.build_icon(tmp_path_factory.mktemp("icon") / "cinewright.ico")
    with Image.open(path) as image:
        yield image


def _frame(icon, size):
    return icon.ico.getimage((size, size)).convert("RGBA")


def test_icon_has_all_sizes(icon):
    assert icon.format == "ICO"
    assert set(icon.info["sizes"]) == {(s, s) for s in (16, 24, 32, 48, 64, 128, 256)}


def test_logo_colour_comes_from_svg():
    assert make_icon.Logo(make_icon.MARK_SVG).color == "#6D28D9"
    assert make_icon.Logo(make_icon.FAVICON_SVG).color == "#6D28D9"


def test_32px_perforations_are_transparent(icon):
    frame = _frame(icon, 32)
    alpha = frame.getchannel("A")
    # Eight sprocket holes: 4x4 rounded squares at x 2 and 26, y 2, 10, 18, 26 (mark.svg, 32-unit grid).
    for x0 in (2, 26):
        for y0 in (2, 10, 18, 26):
            corners = {(x0, y0), (x0 + 3, y0), (x0, y0 + 3), (x0 + 3, y0 + 3)}
            for x in range(x0, x0 + 4):
                for y in range(y0, y0 + 4):
                    a = alpha.getpixel((x, y))
                    if (x, y) in corners:
                        assert 0 < a < 255, (x, y, a)  # 0.5 corner radius: partly covered
                    else:
                        assert a == 0, (x, y, a)
            # The frame around each hole stays opaque.
            assert alpha.getpixel((x0 - 1, y0 + 1)) == 255
            assert alpha.getpixel((x0 + 4, y0 + 1)) == 255
            assert alpha.getpixel((x0 + 1, y0 + 4)) == 255
    # The two film-edge slots: one unit wide at x 8 and 23, from y 2 to 30.
    for x in (8, 23):
        for y in range(2, 30):
            assert alpha.getpixel((x, y)) == 0, (x, y)
        for y in (0, 1, 30, 31):
            assert alpha.getpixel((x, y)) == 255, (x, y)
        assert alpha.getpixel((x - 1, 16)) == 255
        assert alpha.getpixel((x + 1, 16)) == 255
    # Opaque body in the accent colour; outer corners rounded (radius 1.5).
    assert frame.getpixel((16, 16)) == (*ACCENT, 255)
    assert frame.getpixel((1, 1)) == (*ACCENT, 255)
    for corner in ((0, 0), (31, 0), (0, 31), (31, 31)):
        assert 0 < alpha.getpixel(corner) < 255


def test_16px_uses_the_favicon_grid(icon):
    alpha = _frame(icon, 16).getchannel("A")
    holes = {(x, y) for x0 in (1, 13) for y0 in (1, 5, 9, 13) for x in (x0, x0 + 1) for y in (y0, y0 + 1)}
    corners = {(0, 0), (15, 0), (0, 15), (15, 15)}
    for x in range(16):
        for y in range(16):
            a = alpha.getpixel((x, y))
            if (x, y) in holes:
                assert a == 0, (x, y)
            elif (x, y) in corners:
                assert 0 < a < 255
            else:
                assert a == 255, (x, y, a)  # whole pixels only, and no film-edge slots at 16 px


def test_256px_scales_the_mark(icon):
    alpha = _frame(icon, 256).getchannel("A")
    assert alpha.getpixel((32, 32)) == 0  # centre of the first hole (4, 4) * 8
    assert alpha.getpixel((68, 128)) == 0  # inside the slot x 8..9 -> 64..72
    assert alpha.getpixel((60, 128)) == 255
    assert alpha.getpixel((128, 128)) == 255


def test_path_parser_rejects_unsupported_commands():
    with pytest.raises(ValueError):
        make_icon.parse_path("m0 0h1v1z")
    with pytest.raises(ValueError):
        make_icon.parse_path("M0 0C1 1 2 2 3 3Z")


def test_logo_reader_accepts_single_quotes_and_names_missing_attributes(tmp_path):
    good = tmp_path / "single.svg"
    good.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path fill='#6D28D9' fill-rule='evenodd' d='M0 0H16V16H0Z'/></svg>",
        encoding="utf-8",
    )
    assert make_icon.Logo(good).color == "#6D28D9"
    bad = tmp_path / "nofill.svg"
    bad.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M0 0H16V16H0Z"/></svg>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing attribute"):
        make_icon.Logo(bad)
