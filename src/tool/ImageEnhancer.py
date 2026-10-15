import io
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0,
                  smooth=0, whiten=0, grayscale=False):
    """Basic color adjustment plus light skin smoothing / whitening.

    brightness/contrast/saturation/sharpness are PIL factors (1.0 = unchanged).
    smooth and whiten are strengths in 0-100.
    """
    img = image.convert("RGB")
    if smooth > 0:
        arr = np.array(img)
        strength = smooth / 100.0
        filtered = cv2.bilateralFilter(arr, d=9, sigmaColor=40 + 60 * strength, sigmaSpace=9)
        img = Image.fromarray(cv2.addWeighted(filtered, strength, arr, 1 - strength, 0))
    if whiten > 0:
        arr = np.array(img).astype(np.float32)
        gain = whiten / 100.0
        arr = arr + (255.0 - arr) * 0.25 * gain
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    if grayscale:
        img = ImageOps.grayscale(img).convert("RGB")
    return img


def auto_enhance(image):
    """Auto levels: stretch histogram, ignoring 1% outliers."""
    return ImageOps.autocontrast(image.convert("RGB"), cutoff=1)


def add_watermark(image, text, opacity=0.35, font_size=0, angle=30, tiled=True, color=(128, 128, 128)):
    """Add a text watermark, tiled across the photo or centered."""
    base = image.convert("RGBA")
    if not text:
        return base.convert("RGB")
    w, h = base.size
    size = font_size if font_size > 0 else max(16, min(w, h) // 12)
    font = _load_font(size)
    alpha = int(255 * max(0.0, min(1.0, opacity)))

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    tw, th = right - left, bottom - top

    diag = int(math.hypot(w, h))
    layer = Image.new("RGBA", (diag, diag), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = (*color, alpha)
    if tiled:
        step_x, step_y = int(tw * 1.8) + 1, int(th * 3) + 1
        for row, y in enumerate(range(0, diag, step_y)):
            offset = (step_x // 2) if row % 2 else 0
            for x in range(-step_x, diag, step_x):
                draw.text((x + offset - left, y - top), text, font=font, fill=fill)
    else:
        draw.text(((diag - tw) // 2 - left, (diag - th) // 2 - top), text, font=font, fill=fill)

    layer = layer.rotate(angle, resample=Image.BICUBIC)
    box = ((diag - w) // 2, (diag - h) // 2, (diag - w) // 2 + w, (diag - h) // 2 + h)
    base.alpha_composite(layer.crop(box))
    return base.convert("RGB")


def transform_image(image, rotate_angle=0, flip_h=False, flip_v=False,
                    border=0, border_color=(255, 255, 255)):
    """Rotate by an arbitrary angle (counter-clockwise), flip, and add a border."""
    img = image.convert("RGB")
    if rotate_angle % 360:
        img = img.rotate(rotate_angle, resample=Image.BICUBIC, expand=True, fillcolor=tuple(border_color))
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    if border > 0:
        img = ImageOps.expand(img, border=int(border), fill=tuple(border_color))
    return img


def resize_image(image, width, height, keep_ratio=True):
    """Resize to width x height; with keep_ratio the image is fit inside the box."""
    img = image.convert("RGB")
    if width <= 0 or height <= 0:
        return img
    if keep_ratio:
        img = ImageOps.contain(img, (int(width), int(height)), Image.LANCZOS)
    else:
        img = img.resize((int(width), int(height)), Image.LANCZOS)
    return img


def encode_image(image, fmt="JPEG", target_kb=0, dpi=300):
    """Encode an image to bytes. For JPEG with target_kb > 0, pick the highest
    quality whose output is within the target size (best effort)."""
    fmt = fmt.upper()
    img = image.convert("RGB")

    def _encode(quality):
        buf = io.BytesIO()
        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=quality, dpi=(dpi, dpi), optimize=True)
        else:
            img.save(buf, format=fmt, dpi=(dpi, dpi))
        return buf.getvalue()

    if fmt != "JPEG" or target_kb <= 0:
        return _encode(95)

    limit = target_kb * 1024
    lo, hi, best = 5, 95, None
    while lo <= hi:
        mid = (lo + hi) // 2
        data = _encode(mid)
        if len(data) <= limit:
            best, lo = data, mid + 1
        else:
            hi = mid - 1
    return best if best is not None else _encode(5)


def crop_margins(image, top=0, bottom=0, left=0, right=0):
    """Crop a percentage (0-45) off each side of a PIL image."""
    w, h = image.size
    clamp = lambda v: max(0.0, min(45.0, float(v or 0)))
    box = (int(w * clamp(left) / 100), int(h * clamp(top) / 100),
           w - int(w * clamp(right) / 100), h - int(h * clamp(bottom) / 100))
    return image.crop(box) if box != (0, 0, w, h) else image


def apply_post_bgr(bgr, opts):
    """Apply the optional beautify / watermark options to a BGR numpy image
    (the final ID photo). Returns a BGR numpy image of the same size."""
    if not opts:
        return bgr
    needs_color = (opts.get("auto") or opts.get("gray") or opts.get("smooth") or opts.get("whiten")
                   or any(abs(opts.get(k, 1.0) - 1.0) > 1e-6
                          for k in ("brightness", "contrast", "saturation", "sharpness")))
    needs_wm = bool(opts.get("wm_text"))
    needs_tf = bool(opts.get("rotate") or opts.get("flip_h") or opts.get("flip_v") or opts.get("border"))
    if not needs_color and not needs_wm and not needs_tf:
        return bgr
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if needs_tf:
        # Keep the standard photo size: rotation is not expanded and the border is drawn inside the edge.
        color = tuple(opts.get("border_color") or (255, 255, 255))
        if opts.get("rotate"):
            img = img.rotate(opts["rotate"], resample=Image.BICUBIC, expand=False, fillcolor=color)
        if opts.get("flip_h"):
            img = ImageOps.mirror(img)
        if opts.get("flip_v"):
            img = ImageOps.flip(img)
        b = int(opts.get("border") or 0)
        if b > 0:
            d = ImageDraw.Draw(img)
            w, h = img.size
            for i in range(min(b, min(w, h) // 2)):
                d.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    if opts.get("auto"):
        img = auto_enhance(img)
    if needs_color:
        img = enhance_image(img, opts.get("brightness", 1.0), opts.get("contrast", 1.0),
                            opts.get("saturation", 1.0), opts.get("sharpness", 1.0),
                            opts.get("smooth", 0), opts.get("whiten", 0), opts.get("gray", False))
    if needs_wm:
        img = add_watermark(img, opts["wm_text"], opts.get("wm_opacity", 0.35), int(opts.get("wm_size", 0)),
                            opts.get("wm_angle", 30), opts.get("wm_tiled", True))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
