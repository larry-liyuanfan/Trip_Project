"""Deterministic helpers for preparing human-annotation candidate pools."""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from src.evaluation.manifests import ManifestValidationError, perceptual_hash_distance


PRODUCT_CATEGORY_TERMS = {
    "hotel": {
        "hotels",
        "bed & breakfast",
        "resorts",
        "hostels",
        "guest houses",
        "vacation rentals",
    },
    "attraction": {
        "museums",
        "landmarks & historical buildings",
        "amusement parks",
        "zoos",
        "botanical gardens",
        "parks",
        "aquariums",
        "arts & entertainment",
        "art galleries",
        "beaches",
        "tours",
        "music venues",
        "wineries",
    },
    "restaurant": {"restaurants"},
}

AFTER_SALES_TERMS = {
    "hygiene_stain": {
        "mold",
        "mould",
        "dirty sheets",
        "dirty room",
        "stain",
        "filthy",
        "cockroach",
        "bed bug",
    },
    "facility_damage": {
        "broken",
        "damaged",
        "not working",
        "out of order",
        "leaking",
    },
    "attraction_closure": {
        "museum was closed",
        "park was closed",
        "attraction was closed",
        "venue was closed",
        "permanently closed",
        "ticket was cancelled",
    },
    "transport_delay": {
        "flight was delayed",
        "train was delayed",
        "bus was delayed",
        "transport was delayed",
        "flight cancelled",
        "train cancelled",
        "missed connection",
    },
}


@dataclass(frozen=True)
class ImageFingerprints:
    sha256: str
    perceptual_hash: str


@dataclass(frozen=True)
class SyntheticEvidenceRenderAudit:
    """Describe the deterministic template and measured text bounds."""

    template_name: str
    text_boxes: tuple[tuple[int, int, int, int], ...]


SYNTHETIC_CANVAS_SIZE = (960, 640)
SYNTHETIC_SAFE_MARGIN = 40
SYNTHETIC_TEMPLATE_NAMES = (
    "official_notice",
    "booking_status",
    "app_notification",
    "ticket_status",
)
VISUAL_AFTER_SALES_TYPES = {"hygiene_stain", "facility_damage"}


def image_fingerprints(root: Path, repository_relative_path: str) -> ImageFingerprints:
    """Return byte SHA-256 and a deterministic 64-bit difference hash."""
    root = Path(root).resolve()
    relative = Path(repository_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ManifestValidationError("candidate image path must be repository-relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ManifestValidationError("candidate image path escapes repository root") from exc
    if not path.is_file():
        raise ManifestValidationError(f"candidate image does not exist: {repository_relative_path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(grayscale.get_flattened_data())
    except (OSError, UnidentifiedImageError) as exc:
        raise ManifestValidationError(f"candidate image is unreadable: {repository_relative_path}") from exc
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return ImageFingerprints(digest.hexdigest(), f"{bits:016x}")


class CandidateDeduplicator:
    """Track cross-scene exact, grouped, and perceptual candidate collisions."""

    def __init__(self, *, max_perceptual_distance: int = 4) -> None:
        if isinstance(max_perceptual_distance, bool) or not 0 <= max_perceptual_distance <= 64:
            raise ManifestValidationError("max_perceptual_distance must be between 0 and 64")
        self.max_perceptual_distance = max_perceptual_distance
        self.source_ids: set[str] = set()
        self.group_ids: set[str] = set()
        self.image_hashes: set[str] = set()
        self.perceptual_hashes: list[str] = []

    def accept(
        self,
        *,
        source_id: str,
        group_id: str,
        image_hashes: Iterable[str],
        perceptual_hashes: Iterable[str],
    ) -> bool:
        image_hash_list = list(image_hashes)
        perceptual_hash_list = list(perceptual_hashes)
        if source_id in self.source_ids or group_id in self.group_ids:
            return False
        if len(set(image_hash_list)) != len(image_hash_list):
            return False
        if self.image_hashes.intersection(image_hash_list):
            return False
        all_fingerprints = self.perceptual_hashes + perceptual_hash_list
        for index, fingerprint in enumerate(all_fingerprints):
            for other in all_fingerprints[index + 1 :]:
                if perceptual_hash_distance(fingerprint, other) <= self.max_perceptual_distance:
                    return False
        self.source_ids.add(source_id)
        self.group_ids.add(group_id)
        self.image_hashes.update(image_hash_list)
        self.perceptual_hashes.extend(perceptual_hash_list)
        return True


def classify_product_coverage(categories: Iterable[str] | None) -> str | None:
    """Map explicit Yelp category labels to one unambiguous OTA coverage stratum."""
    normalized = {str(category).strip().lower() for category in categories or []}
    for coverage in ("hotel", "attraction", "restaurant"):
        if normalized.intersection(PRODUCT_CATEGORY_TERMS[coverage]):
            return coverage
    return None


def classify_after_sales_issue(text: str) -> str | None:
    """Conservatively route review text to a candidate stratum, not a gold label."""
    normalized = " ".join(text.lower().split())
    matches = [
        issue
        for issue, terms in AFTER_SALES_TERMS.items()
        if any(term in normalized for term in terms)
    ]
    return matches[0] if len(matches) == 1 else None


def _validate_synthetic_evidence_identity(issue_type: str, index: int) -> None:
    if issue_type not in {"attraction_closure", "transport_delay"}:
        raise ManifestValidationError("synthetic evidence only supports closure or delay")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ManifestValidationError("synthetic evidence index must be a non-negative integer")


def synthetic_evidence_template_name(issue_type: str, index: int) -> str:
    """Return the stable v2 business-document layout for one sample."""
    _validate_synthetic_evidence_identity(issue_type, index)
    return SYNTHETIC_TEMPLATE_NAMES[index % len(SYNTHETIC_TEMPLATE_NAMES)]


def _synthetic_palette(issue_type: str, digest: bytes) -> tuple[tuple[int, int, int], ...]:
    if issue_type == "attraction_closure":
        accent = (164 + digest[0] % 24, 45 + digest[1] % 18, 52 + digest[2] % 16)
        dark = (76, 31, 36)
        pale = (250, 239, 237)
    else:
        accent = (202 + digest[0] % 24, 117 + digest[1] % 28, 24 + digest[2] % 18)
        dark = (37, 61, 75)
        pale = (249, 242, 225)
    background = tuple(239 + byte % 10 for byte in digest[3:6])
    return accent, dark, pale, background


def _draw_checked_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font_size: int,
    fill: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    font = ImageFont.load_default(size=font_size)
    box = draw.textbbox(position, text, font=font)
    left, top, right, bottom = box
    width, height = SYNTHETIC_CANVAS_SIZE
    margin = SYNTHETIC_SAFE_MARGIN
    if left < margin or top < margin or right > width - margin or bottom > height - margin:
        raise ManifestValidationError(f"synthetic evidence text exceeds safe bounds: {text}")
    draw.text(position, text, font=font, fill=fill)
    return box


def _draw_security_watermark(
    draw: ImageDraw.ImageDraw,
    *,
    security_code: int,
    bounds: tuple[int, int, int, int],
    accent: tuple[int, int, int],
) -> None:
    """Add a deterministic three-row machine-readable verification band."""
    left, top, right, bottom = bounds
    available_width = right - left
    columns = 9
    rows = 3
    cell_width = available_width / columns
    band_top = top
    cell_height = (bottom - band_top) / rows
    draw.rounded_rectangle(
        (left, band_top, right, bottom),
        radius=12,
        fill=(248, 248, 246),
        outline=(225, 225, 221),
        width=2,
    )
    for row in range(rows):
        row_bits = (security_code >> ((2 - row) * 8)) & 0xFF
        levels = [0]
        for comparison in range(8):
            bit = (row_bits >> (7 - comparison)) & 1
            levels.append(levels[-1] - 1 if bit else levels[-1] + 1)
        minimum = min(levels)
        maximum = max(levels)
        span = max(maximum - minimum, 1)
        for column, raw_level in enumerate(levels):
            gray = 76 + round((raw_level - minimum) * 156 / span)
            fill = tuple((channel + gray * 3) // 4 for channel in accent)
            x1 = round(left + column * cell_width)
            y1 = round(band_top + row * cell_height)
            x2 = round(left + (column + 1) * cell_width)
            y2 = round(band_top + (row + 1) * cell_height)
            inset = 3
            draw.rounded_rectangle(
                (x1 + inset, y1 + inset, x2 - inset, y2 - inset),
                radius=6,
                fill=fill,
            )


def _image_difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.get_flattened_data())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{bits:016x}"


@lru_cache(maxsize=None)
def _selected_security_variant(ordinal: int) -> tuple[int, str]:
    issue_type = "attraction_closure" if ordinal % 2 == 0 else "transport_delay"
    index = ordinal // 2
    previous_hashes = [
        _selected_security_variant(previous)[1] for previous in range(ordinal)
    ]
    for salt in range(2_000):
        security_code = int.from_bytes(
            hashlib.sha256(
                f"week3-security-variant-v2|{issue_type}|{index}|{salt}".encode(
                    "utf-8"
                )
            ).digest()[:3],
            "big",
        )
        image, _ = _render_synthetic_evidence_image_with_code(
            issue_type=issue_type,
            index=index,
            security_code=security_code,
        )
        perceptual_hash = _image_difference_hash(image)
        if all(
            (int(perceptual_hash, 16) ^ int(previous, 16)).bit_count() > 4
            for previous in previous_hashes
        ):
            return security_code, perceptual_hash
    raise ManifestValidationError(
        f"unable to allocate a distinct synthetic security variant: {issue_type}:{index}"
    )


def _draw_official_notice(
    draw: ImageDraw.ImageDraw,
    *,
    accent: tuple[int, int, int],
    dark: tuple[int, int, int],
    pale: tuple[int, int, int],
    security_code: int,
    heading: str,
    status: str,
    detail: str,
) -> tuple[tuple[int, int, int, int], ...]:
    draw.rounded_rectangle((72, 70, 888, 570), radius=24, fill=(255, 255, 255), outline=(218, 214, 205), width=2)
    draw.rounded_rectangle((72, 70, 92, 570), radius=12, fill=accent)
    draw.rectangle((86, 70, 104, 570), fill=accent)
    _draw_security_watermark(
        draw,
        security_code=security_code,
        bounds=(106, 344, 876, 558),
        accent=accent,
    )
    draw.ellipse((790, 118, 838, 166), fill=pale, outline=accent, width=4)
    draw.line((790, 196, 840, 196), fill=(218, 214, 205), width=3)
    draw.line((790, 214, 858, 214), fill=(218, 214, 205), width=3)
    draw.line((120, 300, 840, 300), fill=(224, 220, 211), width=2)
    return (
        _draw_checked_text(draw, (122, 126), heading, font_size=38, fill=dark),
        _draw_checked_text(draw, (122, 232), status, font_size=64, fill=accent),
        _draw_checked_text(draw, (122, 310), detail, font_size=24, fill=(70, 70, 68)),
    )


def _draw_booking_status(
    draw: ImageDraw.ImageDraw,
    *,
    accent: tuple[int, int, int],
    dark: tuple[int, int, int],
    pale: tuple[int, int, int],
    security_code: int,
    heading: str,
    status: str,
    detail: str,
) -> tuple[tuple[int, int, int, int], ...]:
    draw.rounded_rectangle((55, 72, 905, 568), radius=28, fill=(255, 255, 255), outline=(218, 221, 223), width=2)
    draw.rounded_rectangle((55, 72, 238, 568), radius=28, fill=dark)
    draw.rectangle((210, 72, 238, 568), fill=dark)
    _draw_security_watermark(
        draw,
        security_code=security_code,
        bounds=(248, 344, 888, 554),
        accent=accent,
    )
    draw.ellipse((105, 130, 188, 213), outline=(255, 255, 255), width=7)
    draw.line((126, 173, 148, 195), fill=(255, 255, 255), width=7)
    draw.line((148, 195, 178, 151), fill=(255, 255, 255), width=7)
    draw.rounded_rectangle((286, 210, 820, 290), radius=20, fill=pale, outline=accent, width=3)
    return (
        _draw_checked_text(draw, (286, 126), heading, font_size=34, fill=dark),
        _draw_checked_text(draw, (326, 226), status, font_size=48, fill=accent),
        _draw_checked_text(draw, (286, 304), detail, font_size=24, fill=(69, 73, 76)),
    )


def _draw_app_notification(
    draw: ImageDraw.ImageDraw,
    *,
    accent: tuple[int, int, int],
    dark: tuple[int, int, int],
    pale: tuple[int, int, int],
    security_code: int,
    heading: str,
    status: str,
    detail: str,
) -> tuple[tuple[int, int, int, int], ...]:
    draw.rounded_rectangle((86, 42, 874, 598), radius=48, fill=(30, 35, 40))
    draw.rounded_rectangle((102, 58, 858, 582), radius=36, fill=(247, 248, 249))
    draw.rounded_rectangle((378, 72, 582, 92), radius=10, fill=(30, 35, 40))
    draw.ellipse((224, 128, 278, 182), fill=accent)
    draw.rounded_rectangle((222, 218, 738, 338), radius=24, fill=(255, 255, 255), outline=(225, 226, 228), width=2)
    draw.rounded_rectangle((246, 244, 690, 306), radius=18, fill=pale)
    _draw_security_watermark(
        draw,
        security_code=security_code,
        bounds=(116, 350, 844, 566),
        accent=accent,
    )
    return (
        _draw_checked_text(draw, (302, 134), heading, font_size=28, fill=dark),
        _draw_checked_text(draw, (276, 257), status, font_size=44, fill=accent),
        _draw_checked_text(draw, (206, 314), detail, font_size=24, fill=(68, 72, 76)),
    )


def _draw_ticket_status(
    draw: ImageDraw.ImageDraw,
    *,
    accent: tuple[int, int, int],
    dark: tuple[int, int, int],
    pale: tuple[int, int, int],
    security_code: int,
    heading: str,
    status: str,
    detail: str,
) -> tuple[tuple[int, int, int, int], ...]:
    draw.rounded_rectangle((66, 88, 894, 552), radius=26, fill=(255, 255, 255), outline=(210, 212, 210), width=2)
    draw.rectangle((66, 88, 894, 144), fill=dark)
    draw.rectangle((66, 144, 82, 552), fill=accent)
    _draw_security_watermark(
        draw,
        security_code=security_code,
        bounds=(92, 330, 868, 536),
        accent=accent,
    )
    draw.ellipse((42, 256, 90, 304), fill=pale)
    draw.ellipse((870, 256, 918, 304), fill=pale)
    for y in range(176, 492, 36):
        draw.line((720, y, 720, y + 18), fill=(196, 199, 198), width=3)
    draw.rounded_rectangle((116, 210, 650, 300), radius=18, fill=pale, outline=accent, width=3)
    draw.line((116, 294, 662, 294), fill=(218, 220, 218), width=2)
    return (
        _draw_checked_text(draw, (116, 174), heading, font_size=34, fill=dark),
        _draw_checked_text(draw, (148, 226), status, font_size=52, fill=accent),
        _draw_checked_text(draw, (116, 300), detail, font_size=24, fill=(65, 69, 69)),
    )


def render_synthetic_evidence_image(
    *,
    issue_type: str,
    index: int,
) -> tuple[Image.Image, SyntheticEvidenceRenderAudit]:
    """Build one deterministic v2 evidence image without touching the filesystem."""
    _validate_synthetic_evidence_identity(issue_type, index)
    issue_offset = 0 if issue_type == "attraction_closure" else 1
    security_code, _ = _selected_security_variant(index * 2 + issue_offset)
    return _render_synthetic_evidence_image_with_code(
        issue_type=issue_type,
        index=index,
        security_code=security_code,
    )


def _render_synthetic_evidence_image_with_code(
    *,
    issue_type: str,
    index: int,
    security_code: int,
) -> tuple[Image.Image, SyntheticEvidenceRenderAudit]:
    template_name = synthetic_evidence_template_name(issue_type, index)
    digest = hashlib.sha256(
        f"week3-after-sales-evidence-v2|{issue_type}|{index}".encode("utf-8")
    ).digest()
    accent, dark, pale, background = _synthetic_palette(issue_type, digest)
    image = Image.new("RGB", SYNTHETIC_CANVAS_SIZE, color=background)
    draw = ImageDraw.Draw(image)
    heading, status, detail = synthetic_evidence_text(
        issue_type=issue_type,
        index=index,
    )
    renderer = {
        "official_notice": _draw_official_notice,
        "booking_status": _draw_booking_status,
        "app_notification": _draw_app_notification,
        "ticket_status": _draw_ticket_status,
    }[template_name]
    text_boxes = renderer(
        draw,
        accent=accent,
        dark=dark,
        pale=pale,
        security_code=security_code,
        heading=heading,
        status=status,
        detail=detail,
    )
    audit = SyntheticEvidenceRenderAudit(
        template_name=template_name,
        text_boxes=text_boxes,
    )
    return image, audit


def render_synthetic_evidence(
    path: Path,
    *,
    issue_type: str,
    index: int,
) -> SyntheticEvidenceRenderAudit:
    """Render one deterministic, measured v2 business-synthetic evidence card."""
    image, audit = render_synthetic_evidence_image(
        issue_type=issue_type,
        index=index,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return audit


def synthetic_evidence_text(*, issue_type: str, index: int) -> tuple[str, str, str]:
    """Return the exact deterministic text rendered on one synthetic evidence card."""
    if issue_type == "attraction_closure":
        return (
            "ATTRACTION STATUS NOTICE",
            "CLOSED",
            f"Venue C-{index:03d} | Ticket T-{1000 + index} | Service unavailable",
        )
    if issue_type == "transport_delay":
        return (
            "TRANSPORT SERVICE UPDATE",
            "DELAYED",
            f"Route R-{index:03d} | Scheduled {8 + index % 10:02d}:00 | Delay {30 + index % 7 * 15} min",
        )
    raise ManifestValidationError("synthetic evidence only supports closure or delay")


def render_synthetic_visual_evidence(
    path: Path,
    *,
    issue_type: str,
    index: int,
) -> None:
    """Render a deterministic, label-free visual hygiene or damage scene."""
    if issue_type not in VISUAL_AFTER_SALES_TYPES:
        raise ManifestValidationError(
            "visual synthetic evidence supports hygiene_stain or facility_damage"
        )
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ManifestValidationError("synthetic evidence index must be non-negative")
    digest = hashlib.sha256(
        f"week3-after-sales-visual-v1|{issue_type}|{index}".encode("utf-8")
    ).digest()
    image = _patterned_scene_background(digest)
    draw = ImageDraw.Draw(image)
    if issue_type == "hygiene_stain":
        _draw_hygiene_scene(draw, digest, index)
    else:
        _draw_facility_damage_scene(draw, digest, index)
    _draw_checked_text(
        draw,
        (650, 570),
        "PROJECT-OWNED SYNTHETIC",
        font_size=12,
        fill=(88, 88, 84),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False)


def _patterned_scene_background(digest: bytes) -> Image.Image:
    image = Image.new("RGB", SYNTHETIC_CANVAS_SIZE, color=(226, 222, 211))
    draw = ImageDraw.Draw(image)
    for row in range(8):
        for column in range(9):
            bit_index = row * 9 + column
            byte = digest[bit_index % len(digest)]
            level = 194 + ((byte >> (bit_index % 5)) & 1) * 34
            tint = (level + byte % 8, level, max(level - byte % 7, 0))
            left = column * SYNTHETIC_CANVAS_SIZE[0] // 9
            top = row * SYNTHETIC_CANVAS_SIZE[1] // 8
            right = (column + 1) * SYNTHETIC_CANVAS_SIZE[0] // 9
            bottom = (row + 1) * SYNTHETIC_CANVAS_SIZE[1] // 8
            draw.rectangle((left, top, right, bottom), fill=tint)
    draw.rectangle((48, 42, 912, 598), fill=(242, 239, 231), outline=(88, 94, 92), width=5)
    return image


def _draw_hygiene_scene(draw: ImageDraw.ImageDraw, digest: bytes, index: int) -> None:
    """Draw a hotel bed or tiled bathroom with an unmistakable visible stain."""
    if index % 2 == 0:
        draw.rectangle((120, 250, 840, 540), fill=(231, 226, 215), outline=(95, 91, 83), width=5)
        draw.rounded_rectangle((150, 280, 810, 520), radius=24, fill=(248, 247, 241), outline=(170, 164, 150), width=4)
        draw.rounded_rectangle((180, 225, 430, 330), radius=22, fill=(252, 251, 247), outline=(180, 176, 167), width=3)
        draw.rounded_rectangle((530, 225, 780, 330), radius=22, fill=(252, 251, 247), outline=(180, 176, 167), width=3)
        center_x = 350 + digest[0] % 260
        center_y = 360 + digest[1] % 90
    else:
        for x in range(120, 850, 90):
            draw.line((x, 120, x, 550), fill=(174, 178, 176), width=3)
        for y in range(120, 560, 86):
            draw.line((110, y, 850, y), fill=(174, 178, 176), width=3)
        center_x = 300 + digest[0] % 360
        center_y = 230 + digest[1] % 220
    stain = (112 + digest[2] % 40, 61 + digest[3] % 30, 32 + digest[4] % 20)
    for offset in range(7):
        dx = (digest[5 + offset] % 95) - 47
        dy = (digest[12 + offset] % 75) - 37
        radius_x = 24 + digest[19 + offset] % 42
        radius_y = 16 + digest[(26 + offset) % len(digest)] % 30
        draw.ellipse(
            (center_x + dx - radius_x, center_y + dy - radius_y,
             center_x + dx + radius_x, center_y + dy + radius_y),
            fill=stain,
        )
    draw.ellipse((center_x - 110, center_y - 80, center_x + 110, center_y + 80), outline=(91, 42, 24), width=8)


def _draw_facility_damage_scene(draw: ImageDraw.ImageDraw, digest: bytes, index: int) -> None:
    """Draw a damaged sink, chair, window, or pipe with explicit break geometry."""
    variant = index % 4
    if variant == 0:
        draw.rectangle((160, 160, 800, 500), fill=(216, 224, 224), outline=(70, 82, 85), width=6)
        draw.ellipse((260, 270, 700, 500), fill=(247, 247, 241), outline=(92, 101, 102), width=7)
        draw.arc((400, 170, 560, 350), 180, 360, fill=(75, 82, 83), width=16)
        origin = (480, 360)
    elif variant == 1:
        draw.rectangle((300, 170, 660, 430), fill=(152, 103, 68), outline=(70, 46, 35), width=8)
        draw.rectangle((330, 420, 380, 555), fill=(119, 77, 52))
        draw.rectangle((580, 420, 630, 555), fill=(119, 77, 52))
        draw.line((380, 555, 520, 455), fill=(45, 35, 30), width=22)
        origin = (510, 450)
    elif variant == 2:
        draw.rectangle((180, 110, 780, 540), fill=(178, 211, 225), outline=(62, 72, 76), width=10)
        draw.line((480, 110, 480, 540), fill=(86, 92, 94), width=7)
        draw.line((180, 325, 780, 325), fill=(86, 92, 94), width=7)
        origin = (500, 300)
    else:
        draw.rectangle((160, 200, 800, 520), fill=(221, 218, 207), outline=(83, 82, 76), width=6)
        draw.line((250, 330, 710, 330), fill=(95, 105, 108), width=45)
        draw.line((510, 330, 510, 520), fill=(95, 105, 108), width=45)
        draw.ellipse((475, 470, 545, 560), fill=(65, 132, 178))
        origin = (510, 330)
    x, y = origin
    crack = [(x, y)]
    for step in range(6):
        x += (digest[step] % 75) - 28
        y += 28 + digest[step + 6] % 35
        crack.append((x, y))
    draw.line(crack, fill=(28, 28, 27), width=12, joint="curve")
    for point_x, point_y in crack[1:-1]:
        draw.line((point_x, point_y, point_x + 45, point_y - 30), fill=(28, 28, 27), width=7)


def retain_best_group_row(
    grouped: dict[str, tuple[int, dict]],
    *,
    group_id: str,
    rank: int,
    row: dict,
) -> None:
    """Retain the lowest deterministic rank for one leakage group."""
    current = grouped.get(group_id)
    if current is None or rank < current[0]:
        grouped[group_id] = (rank, row)
