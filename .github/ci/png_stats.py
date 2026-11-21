"""Checks the viewer's screenshots in CI without any third-party packages (#47).

    python3 png_stats.py glow.png plain.png

Reads two 8-bit RGB/RGBA PNGs (what raylib's ExportImage writes) and compares the
picture area, leaving out the stats panel (top left) and the key help (bottom line):
  - the glow image must not be black (something was drawn);
  - the plain image (opaque squares of one color) must be nearly uniform: most lit pixels
    within a few levels of the most common brightness;
  - the glow image must not be: soft sprites that add up give a spread of brightness.
That fails if the glow pass draws nothing, draws flat squares, or plain mode draws glow.
Exits 1 with a message if a check fails.

Not checked: whether bloom adds light. The CI scene (smoke.cfg) is two cold planets, and
bloom only lights up bright, hot material: measured, it adds under 1% to that picture, too
little to tell from frame-to-frame differences. A bloom pass that failed to compile is
still caught, by the "bloom disabled" check in the CI job.
"""

import struct
import sys
import zlib


def read_png(path):
    """Returns (width, height, rows) with rows a list of lists of (r, g, b) tuples."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos, idat, width = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or color not in (2, 6) or interlace != 0:
                raise ValueError(f"{path}: expected 8-bit RGB/RGBA, not interlaced")
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, prev = [], bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for x in range(stride):  # undo the PNG row filter (spec section 9)
            a = line[x - channels] if x >= channels else 0
            b = prev[x]
            c = prev[x - channels] if x >= channels else 0
            if kind == 1:
                line[x] = (line[x] + a) & 255
            elif kind == 2:
                line[x] = (line[x] + b) & 255
            elif kind == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 255
        rows.append([tuple(line[i:i + 3]) for i in range(0, stride, channels)])
        prev = line
    return width, height, rows


def stats(path):
    """(lit pixel count, brightest luma, share of lit pixels within 12 levels of the most
    common lit brightness, summed luma) over the picture area."""
    w, h, rows = read_png(path)
    lit, peak, total = 0, 0.0, 0.0
    counts = [0] * 256
    for y in range(h):
        if y > h - 70:  # key help line
            continue
        for x in range(w):
            if x < 350 and y < 240:  # stats panel
                continue
            r, g, b = rows[y][x]
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            peak = max(peak, luma)
            total += luma
            if luma > 40:  # clearly above the dark background (about 6)
                lit += 1
                counts[int(luma)] += 1
    mode = max(range(256), key=lambda k: counts[k])
    near = sum(counts[k] for k in range(max(0, mode - 12), min(256, mode + 13)))
    return lit, peak, (near / lit if lit else 0.0), total


def main(argv):
    glow_lit, glow_peak, glow_uniform, _ = stats(argv[1])
    plain_lit, _, plain_uniform, _ = stats(argv[2])
    print(f"glow: {glow_lit} lit pixels, brightest {glow_peak:.0f}, {glow_uniform:.0%} uniform; "
          f"plain: {plain_lit} lit pixels, {plain_uniform:.0%} uniform")
    if glow_lit == 0 or glow_peak < 60:
        print("the glow image is (nearly) black: nothing was drawn")
        return 1
    if plain_lit == 0 or plain_uniform < 0.9:
        print("the plain image isn't flat squares of one color")
        return 1
    # Measured 37% to 52% (it depends on which frame the screenshot catches); plain is 100%.
    if glow_uniform > 0.75:
        print("the glow image is as flat as plain squares: no soft, adding-up sprites")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
