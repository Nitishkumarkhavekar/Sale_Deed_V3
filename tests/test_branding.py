"""The departmental emblem: that it ships, that it loads, that it fits.

Three separate risks, and the first is the one that actually bites. An image
added by dragging a file into place works perfectly on the machine it was added
on and is simply absent everywhere else - because it was never copied into the
package, or because something references the author's own drive. Neither failure
is visible in a screenshot taken on that machine.

So the tests below assert, in order: the file is inside the shipped tree; nothing
points at an absolute path; the handler that serves it will actually serve it;
the page asks for it; and the geometry cannot distort it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "app" / "ui" / "assets"
LOGO = ASSETS / "income-tax-logo.jpg"


def rule(css: str, selector: str) -> str:
    """The declarations of one rule, with comments removed.

    Read properly rather than by slicing the first N characters after the
    selector: three of these tests used to do that, and all three broke the
    moment a comment above a declaration grew - reporting a missing
    `object-fit` that was sitting four lines further down. A guard that fails
    on prose is a guard people delete.
    """
    start = css.index(selector + " {") + len(selector) + 2
    block = css[start:css.index("}", start)]
    return re.sub(r"/\*.*?\*/", "", block, flags=re.S)


class TestItShipsWithTheApplication:
    def test_the_logo_is_inside_the_package(self):
        """Under `src/`, not beside the project. Anything outside the package
        is absent the moment the application is installed elsewhere."""
        assert LOGO.is_file(), f"{LOGO} is missing"
        assert (ROOT / "src") in LOGO.parents

    def test_it_sits_beside_the_other_served_assets(self):
        """The scheme handler serves exactly one directory, and refuses paths
        that escape it. A logo anywhere else would 404."""
        assert LOGO.parent == ASSETS
        assert (ASSETS / "theme.css").is_file()

    def test_it_is_a_real_jpeg_not_a_placeholder(self):
        data = LOGO.read_bytes()
        assert data[:2] == b"\xff\xd8", "not a JPEG"
        assert len(data) > 5_000, "suspiciously small for artwork"

    def test_it_is_not_excluded_from_version_control(self):
        """`.gitignore` here excludes whole categories - *.pdf, *.bin, model
        weights. An emblem caught by one of those would vanish on clone, which
        is exactly the "works on my machine" failure this file guards."""
        import subprocess

        result = subprocess.run(
            ["git", "check-ignore", str(LOGO.relative_to(ROOT).as_posix())],
            cwd=ROOT, capture_output=True, text=True, check=False)
        assert result.returncode != 0, (
            f"the logo is gitignored by: {result.stdout.strip()}")

    def test_it_is_tracked_by_git(self):
        """Present on disk is not the same as committed."""
        import subprocess

        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch",
             str(LOGO.relative_to(ROOT).as_posix())],
            cwd=ROOT, capture_output=True, text=True, check=False)
        assert result.returncode == 0, "the logo is not tracked by git"


class TestNothingPointsAtThisMachine:
    """The requirement stated outright: the application must not depend on
    `D:\\saledeed v3\\...`."""

    def _sources(self):
        for pattern in ("src/**/*.py", "src/**/*.mustache", "src/**/*.css",
                        "src/**/*.js"):
            yield from ROOT.glob(pattern)

    def test_no_source_file_names_the_original_image_path(self):
        offenders = []
        for path in self._sources():
            text = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"Icometax Logo", text, re.IGNORECASE):
                offenders.append(str(path.relative_to(ROOT)))
        assert not offenders, f"the original filename is referenced by {offenders}"

    def test_the_window_icon_is_resolved_from_the_package(self):
        """From `__file__`, so it is found wherever the application is
        installed - not from a working directory or a drive letter."""
        source = (ROOT / "src" / "app" / "main.py").read_text(encoding="utf-8")
        block = source[source.index("icon_path ="):source.index("icon_path =") + 200]
        assert "__file__" in block
        assert not re.search(r"[A-Za-z]:[\\/]", block)

    def test_a_missing_icon_does_not_stop_the_application(self):
        """A cosmetic file must never be able to prevent a launch."""
        source = (ROOT / "src" / "app" / "main.py").read_text(encoding="utf-8")
        block = source[source.index("icon_path ="):source.index("self.view =")]
        assert "is_file()" in block, "the icon is set without checking it exists"


class TestTheHandlerWillServeIt:
    def test_jpeg_has_a_mime_type(self):
        """Without it the handler falls back to application/octet-stream and
        Chromium declines to render the image - a silent blank."""
        from app.main import MIME

        assert MIME[".jpg"] == b"image/jpeg"
        assert MIME[".jpeg"] == b"image/jpeg"

    def test_the_url_the_page_asks_for_resolves_to_the_file(self):
        """Reproduces the handler's own path arithmetic on the real URL, so a
        change to either the template or the handler breaks this test rather
        than the application."""
        rel = "/assets/income-tax-logo.jpg".lstrip("/")
        target = (ASSETS.resolve() / rel.removeprefix("assets/")).resolve()
        assert target.is_file()
        assert ASSETS.resolve() in target.parents, "the guard would refuse it"

    def test_the_logo_name_is_defined_once(self):
        """The window icon and the served asset must not be able to diverge."""
        from app.main import LOGO_FILE

        assert (ASSETS / LOGO_FILE).is_file()


class TestThePageUsesIt:
    def test_the_shell_requests_the_logo(self, app_service):
        html = app_service.render_page("dashboard", {}, shell_html=True)
        assert "app://ui/assets/income-tax-logo.jpg" in html

    def test_it_carries_alternative_text(self, app_service):
        """A government interface should name its own emblem."""
        html = app_service.render_page("dashboard", {}, shell_html=True)
        assert 'alt="Income Tax Department"' in html

    def test_it_appears_exactly_twice(self, app_service):
        """The top-bar mark and the hero, and nowhere else.

        It used to be once. The top bar carried a tricolour gradient square
        instead - a placeholder standing in for an emblem the application
        already shipped, which is what an operator pointed at. Two marks is now
        the intent, so the count is still pinned: a third copy would mean the
        logo had leaked into page content, which navigation re-inserts.
        """
        html = app_service.render_page("dashboard", {}, shell_html=True)
        assert html.count("income-tax-logo.jpg") == 2

    def test_one_of_them_is_the_top_bar_mark(self, app_service):
        html = app_service.render_page("dashboard", {}, shell_html=True)
        brand = html[html.index('class="brand"'):html.index('class="tabs"')]
        assert "income-tax-logo.jpg" in brand

    def test_the_top_bar_mark_is_no_longer_a_placeholder(self):
        """The gradient square is gone, not merely covered over."""
        css = (ASSETS / "theme.css").read_text(encoding="utf-8")
        assert "linear-gradient" not in rule(css, ".brand .emblem")

    def test_it_is_present_on_every_page(self, app_service):
        """It lives in the shell, so navigation cannot lose it."""
        from app.ui.renderer import PAGES

        for key, _template, _title in PAGES:
            html = app_service.render_page(key, {}, shell_html=True)
            assert "income-tax-logo.jpg" in html, f"absent from {key}"

    def test_the_content_only_render_does_not_repeat_it(self, app_service):
        """Navigation replaces the content area, not the shell. If the logo
        were in the content it would be re-inserted on every navigation and end
        up alongside the shell's copy."""
        inner = app_service.render_page("dashboard", {}, shell_html=False)
        assert "income-tax-logo.jpg" not in inner


class TestItCannotBeDistorted:
    def _css(self):
        return (ASSETS / "theme.css").read_text(encoding="utf-8")

    @pytest.mark.parametrize("selector",
                             (".hero .emblem img", ".brand .emblem img"))
    def test_the_aspect_ratio_is_preserved(self, selector):
        """`object-fit: contain` is the guarantee: whatever box the layout
        gives it, the image letterboxes rather than stretches."""
        assert "object-fit: contain" in rule(self._css(), selector)

    @pytest.mark.parametrize("selector",
                             (".hero .emblem img", ".brand .emblem img"))
    def test_the_ratio_is_known_before_the_image_decodes(self, selector):
        """Without this the box is zero-width until the JPEG arrives and the
        text beside it slides sideways on first paint."""
        assert "aspect-ratio: 535 / 392" in rule(self._css(), selector)

    def test_only_one_dimension_is_fixed(self):
        """Setting both width and height in pixels is how a logo gets squashed
        by a few percent - visible to anyone who knows the artwork, invisible
        to whoever set it."""
        css = self._css()
        block = css[css.index(".hero .emblem img"):]
        block = block[:block.index("}")]
        assert "width: auto" in block

    def test_the_plate_is_not_a_circle(self):
        """The artwork is landscape, so a circular mask crops the wreath and
        clips the ribbon carrying the department's name."""
        css = self._css()
        block = css[css.index(".hero .emblem {"):]
        block = block[:block.index("}")]
        assert "border-radius: 50%" not in block

    def test_the_intrinsic_size_is_declared_on_the_element(self):
        """Width and height attributes let the browser reserve the right box
        before the image loads, so the header does not jump on first paint."""
        base = (ROOT / "src" / "app" / "ui" / "templates"
                / "base.mustache").read_text(encoding="utf-8")
        assert 'width="535" height="392"' in base

    def test_the_declared_size_matches_the_file(self):
        """A stale width/height attribute reserves the wrong box and reintroduces
        the layout jump it was meant to prevent."""
        data = LOGO.read_bytes()
        index = 2
        found = None
        while index < len(data) - 9:
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                height = int.from_bytes(data[index + 5:index + 7], "big")
                width = int.from_bytes(data[index + 7:index + 9], "big")
                found = (width, height)
                break
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            index += 2 + int.from_bytes(data[index + 2:index + 4], "big")

        assert found == (535, 392), f"the artwork is {found}"

    def test_it_does_not_overlap_the_controls_beside_it(self):
        """The hero is a flex row ending in a spacer and a pill. The plate must
        not grow into them."""
        block = rule(self._css(), ".hero .emblem")
        assert "flex: 0 0 auto" in block, "the plate can be stretched by flex"

    def test_the_hero_plate_is_bounded_at_both_ends(self):
        """It scales with the window, which is the point - but a plate free to
        grow would drive the hero taller than its own title block on a wide
        monitor, and one free to shrink would put the ribbon back to a smear.

        The bound used to be a `max-width` on the image. It is the clamp
        ceiling now: the plate shrink-wraps the artwork, so a width cap could
        only letterbox the emblem inside its own badge.
        """
        block = rule(self._css(), ".hero .emblem")
        match = re.search(r"height:\s*clamp\(\s*(\d+)px\s*,[^,]+,\s*(\d+)px\s*\)",
                          block)
        assert match, f"the hero plate has no bounded height: {block.strip()}"
        floor, ceiling = int(match.group(1)), int(match.group(2))
        assert floor >= 64, "smaller than the size the ribbon was already a smear at"
        assert ceiling <= 120, "taller than the hero's own title block"
        assert floor < ceiling

    def test_the_plate_matches_the_artwork_ground(self):
        """The JPEG is on #f3f3f3, not white. On a white plate its edge showed
        as a hard grey rectangle inside the badge - a correctly-proportioned
        logo that looked wrong. Both plates paint the artwork's own ground, so
        there is no seam to see."""
        css = self._css()
        assert "--emblem-ground: #f3f3f3;" in css
        for selector in (".hero .emblem", ".brand .emblem"):
            assert "background: var(--emblem-ground)" in rule(css, selector), \
                selector

    def test_the_ground_is_the_colour_the_artwork_actually_uses(self):
        """Sampled, not assumed. If the artwork is ever replaced with one on a
        different ground, the seam comes back and this is what says so."""
        import pymupdf

        px = pymupdf.Pixmap(str(LOGO))
        corner = tuple(px.samples[:3])
        css = (ASSETS / "theme.css").read_text(encoding="utf-8")
        declared = re.search(r"--emblem-ground:\s*#([0-9a-fA-F]{6});", css)
        assert declared, "no --emblem-ground declared"
        want = tuple(int(declared.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        assert all(abs(a - b) <= 4 for a, b in zip(corner, want)), (
            f"the artwork's ground is {corner}, the stylesheet paints {want}")

    @pytest.mark.parametrize("selector", (".hero .emblem", ".brand .emblem"))
    def test_the_padding_is_even_on_every_side(self, selector):
        """The artwork bleeds to its own left and right edges, so the clearance
        the ribbon tails get from the rounded corner is entirely this padding.
        Uneven padding reads as a logo sitting off-centre in its badge."""
        block = rule(self._css(), selector)
        match = re.search(r"padding:\s*([^;]+);", block)
        assert match, f"{selector} declares no padding"
        values = match.group(1).split()
        assert len(set(values)) == 1, f"{selector} padding is uneven: {values}"
