"""Quick-look video of a run: planet_sim --out frames -> MP4, colored by planet.

    python scripts/render.py frames/run1 -o run1.mp4 [--fps 30] [--every k] [--still last.png]

A debug tool (#11): the polished videos come from the viewer's replay (#49).
Needs scripts/requirements.txt (numpy, matplotlib, and imageio-ffmpeg, which ships
its own ffmpeg binary). Exit codes: 0 ok, 1 bad input, 2 bad command line.
"""

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

HEADER = "id,x,y,z,body"  # what planet_sim's write_frame() writes (src/io.cpp)
FRAME_NAME = re.compile(r"^frame_(\d+)\.csv$")


def list_frames(folder):
    """The frame_*.csv files in folder, sorted by frame number (so 99999 comes before 100000)."""
    found = []
    for name in os.listdir(folder):
        m = FRAME_NAME.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(folder, name)))
    found.sort()
    return [path for _, path in found]


def load_frame(path):
    """Reads one frame. Returns (xyz, body): an (n, 3) float array and an (n,) int array.
    Raises ValueError, naming the file, if it isn't in write_frame()'s format."""
    with open(path, newline="") as f:
        header = f.readline().rstrip("\r\n")
        rows = f.read().splitlines()  # handles \n and \r\n
    if header != HEADER:
        raise ValueError(f"{path}: expected header '{HEADER}', got '{header}'")
    if rows and rows[-1] == "":
        rows.pop()  # one blank line at the end is fine (read_frame allows it too)
    if "" in rows:
        raise ValueError(f"{path}: blank line in the middle of the frame")
    if not rows:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    try:
        # comments=None: a '#' line is an error, as in read_frame, not something to skip.
        data = np.loadtxt(rows, delimiter=",", ndmin=2, comments=None)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if data.shape[1] != 5:
        raise ValueError(f"{path}: expected 5 columns, got {data.shape[1]}")
    if not np.array_equal(data[:, 0], np.arange(len(data))):
        raise ValueError(f"{path}: ids are not 0, 1, 2, ... in order")
    body = data[:, 4]
    if not np.all(body == np.round(body)):
        raise ValueError(f"{path}: body ids must be whole numbers")
    return data[:, 1:4], body.astype(int)


def bounds(paths):
    """One cube (center, half_width) for the whole video, so the camera stays still.
    It holds every frame's particles except the farthest 0.5% on each side of each
    axis: a few particles flung far away shouldn't shrink the planets to a dot. Those
    are clipped from the picture. Reads the frames one at a time."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for path in paths:
        xyz, _ = load_frame(path)
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]  # a blown-up run can have NaN/inf
        if len(xyz) > 0:
            # inverted_cdf picks real particle positions (no interpolation), so frames
            # under 200 particles keep every particle in the box.
            lo = np.minimum(lo, np.percentile(xyz, 0.5, axis=0, method="inverted_cdf"))
            hi = np.maximum(hi, np.percentile(xyz, 99.5, axis=0, method="inverted_cdf"))
    if not np.all(np.isfinite(lo)):
        return np.zeros(3), 1.0
    center = (lo + hi) / 2.0
    half = 1.05 * float(np.max(hi - lo)) / 2.0  # 5% margin
    if half == 0.0:  # one particle, or all at one point
        half = 1.0
    return center, half


def render(paths, out, box, fps=30, still=None):
    """Draws each frame as a 3D scatter inside box (from bounds()) and pipes the images
    into ffmpeg (H.264 MP4)."""
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")  # no window needed, so this runs over SSH and in CI
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    center, half = box
    fig = plt.figure(figsize=(6, 6), dpi=100)  # 600 x 600: even sizes, as H.264 needs
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    colors = plt.get_cmap("tab10")

    # yuv420p so the video plays in browsers and on phones, not just in VLC.
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    dots = None
    with writer.saving(fig, out, dpi=100):
        for k, path in enumerate(paths):
            xyz, body = load_frame(path)
            if dots is not None:
                dots.remove()  # replace the particles, keep the axes
            dots = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c=colors(body % 10),
                              depthshade=False, axlim_clip=True)  # hide points outside the box
            ax.set_title(f"{os.path.basename(path)}  ({k + 1} / {len(paths)})")
            writer.grab_frame()
    if still is not None:
        fig.savefig(still, dpi=100)
    plt.close(fig)


def main(argv):
    parser = argparse.ArgumentParser(description="Render planet_sim frames to an MP4.")
    parser.add_argument("frames", help="folder written by planet_sim --out")
    parser.add_argument("-o", "--out", required=True, help="output video, e.g. run.mp4")
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    parser.add_argument("--every", type=int, default=1, help="use every k-th frame (default 1)")
    parser.add_argument("--still", help="also save the last frame as this PNG")
    args = parser.parse_args(argv)
    if args.fps < 1 or args.every < 1:
        parser.error("--fps and --every need a whole number >= 1")  # exits 2
    if not os.path.isdir(args.frames):
        print(f"render.py: '{args.frames}' is not a folder", file=sys.stderr)
        return 1
    # Check where the outputs go before spending minutes on the video.
    for path in [args.out, args.still]:
        if path is not None and not os.path.isdir(os.path.dirname(os.path.abspath(path))):
            print(f"render.py: the folder for '{path}' doesn't exist", file=sys.stderr)
            return 1
    paths = list_frames(args.frames)[:: args.every]
    if not paths:
        print(f"render.py: no frame_*.csv files in '{args.frames}'", file=sys.stderr)
        return 1
    try:
        box = bounds(paths)  # reads every frame first, so a bad one fails before any output
    except (OSError, ValueError) as e:
        print(f"render.py: {e}", file=sys.stderr)
        return 1
    t0 = time.time()
    try:
        render(paths, args.out, box, fps=args.fps, still=args.still)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"render.py: writing the video failed: {e}", file=sys.stderr)
        return 1
    seconds = time.time() - t0
    print(f"wrote {args.out}: {len(paths)} frames in {seconds:.1f} s "
          f"({len(paths) / seconds:.1f} frames/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
