#!/usr/bin/env python3
"""Generates AppIcon.icns for Mac Sound Lab.

Concept: a macOS-style rounded square with an indigo→violet gradient and a
seven-bar waveform whose centre bar is the accent colour, i.e. "sound placed
around the head" reduced to something legible at 16 px.
"""

import math
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter

BASE = 1024
SCALE = 2                     # draw at 2x, then downsample for clean edges
S = BASE * SCALE
MARGIN = int(100 * SCALE)     # Big Sur icon grid: 824/1024
RADIUS = int(185 * SCALE)


def gradient(size, top, bottom):
    image = Image.new("RGB", (1, size))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        t = y / max(1, size - 1)
        eased = t * t * (3 - 2 * t)
        colour = tuple(int(top[i] + (bottom[i] - top[i]) * eased) for i in range(3))
        draw.point((0, y), fill=colour)
    return image.resize((size, size), Image.BICUBIC)


def rounded_mask(size, radius):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def build():
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # Body
    body_size = S - 2 * MARGIN
    body = gradient(body_size, (37, 41, 98), (96, 44, 150)).convert("RGBA")
    body.putalpha(rounded_mask(body_size, RADIUS))
    canvas.alpha_composite(body, (MARGIN, MARGIN))

    # Soft glow behind the waveform
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(
        (S * 0.22, S * 0.30, S * 0.78, S * 0.70),
        fill=(120, 220, 255, 60),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(70 * SCALE))
    canvas.alpha_composite(Image.composite(glow, Image.new("RGBA", (S, S), (0, 0, 0, 0)),
                                           rounded_mask(S, RADIUS + MARGIN).resize((S, S))))

    # Thin full ring: the "room" around the sound. Fully inside the body so it
    # never reads as a clipped line.
    ring = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring)
    ring_radius = 0.365 * S
    ring_draw.ellipse(
        (S / 2 - ring_radius, S / 2 - ring_radius, S / 2 + ring_radius, S / 2 + ring_radius),
        outline=(255, 255, 255, 40), width=int(0.005 * S))
    canvas.alpha_composite(ring)

    # Waveform: seven rounded bars, symmetric, centre bar accented
    heights = [0.17, 0.30, 0.46, 0.66, 0.46, 0.30, 0.17]
    bar_width = int(0.052 * S)
    gap = int(0.030 * S)
    total = len(heights) * bar_width + (len(heights) - 1) * gap
    start_x = (S - total) / 2
    centre_y = S / 2
    draw = ImageDraw.Draw(canvas)
    for index, fraction in enumerate(heights):
        height = int(fraction * S)
        x0 = start_x + index * (bar_width + gap)
        x1 = x0 + bar_width
        y0 = centre_y - height / 2
        y1 = centre_y + height / 2
        colour = (94, 234, 212) if index == 3 else (255, 255, 255, 236)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=bar_width / 2, fill=colour)

    return canvas


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "build/icon"
    os.makedirs(out_dir, exist_ok=True)
    icon = build().resize((BASE, BASE), Image.LANCZOS)

    iconset = os.path.join(out_dir, "AppIcon.iconset")
    os.makedirs(iconset, exist_ok=True)
    for size in (16, 32, 64, 128, 256, 512, 1024):
        icon.resize((size, size), Image.LANCZOS).save(
            os.path.join(iconset, f"icon_{size}x{size}.png"))
    for size in (16, 32, 128, 256, 512):
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(
            os.path.join(iconset, f"icon_{size}x{size}@2x.png"))

    icns = os.path.join(out_dir, "AppIcon.icns")
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", icns], check=True)
    icon.save(os.path.join(out_dir, "AppIcon.png"))
    print(f"icon: {icns}")


if __name__ == "__main__":
    main()
