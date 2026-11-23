"""Turns planet_view --export images into a video (#49).

    python scripts/make_video.py PNG_DIR out.mp4 [--fps 60] [--gif out.gif] [--gif-width 640]
                                                 [--gif-fps 20]

PNG_DIR holds frame_00000.png, frame_00001.png, ... as planet_view --replay RUN --export
PNG_DIR writes them. The MP4 is H.264 (yuv420p, so it plays in browsers and on phones).
--gif also writes a looping GIF, smaller and at a lower frame rate, for the README.
Uses the ffmpeg that ships with the pinned imageio-ffmpeg package (scripts/requirements.txt),
so nothing needs installing system-wide. Exit codes: 0 ok, 1 bad input or ffmpeg failed,
2 bad command line.
"""

import argparse
import os
import re
import subprocess
import sys

IMAGE_NAME = re.compile(r"^frame_(\d+)\.png$")


def numbered_images(folder):
    """The frame_*.png numbers in folder, sorted. Raises ValueError if there are none or
    they aren't 0, 1, 2, ... with no gaps (ffmpeg would silently stop at a gap)."""
    numbers = sorted(int(m.group(1)) for m in map(IMAGE_NAME.match, os.listdir(folder)) if m)
    if not numbers:
        raise ValueError(f"no frame_*.png images in '{folder}' (make them with "
                         f"planet_view --replay RUN --export '{folder}')")
    if numbers != list(range(len(numbers))):
        missing = next(k for k in range(len(numbers) + 1) if k not in set(numbers))
        raise ValueError(f"'{folder}': images must be frame_00000.png, frame_00001.png, ... "
                         f"with no gaps; frame {missing} is missing")
    return numbers


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args):
    """Runs ffmpeg quietly; raises RuntimeError with its last lines if it fails."""
    done = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error"] + args,
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + done.stderr.strip()[-500:])


def make_mp4(folder, out, fps):
    pattern = os.path.join(folder, "frame_%05d.png")
    # Even width and height, as H.264 with yuv420p needs (odd export sizes are cropped by
    # one pixel); crf 18 is close to visually lossless.
    run_ffmpeg(["-framerate", str(fps), "-i", pattern, "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out])


def make_gif(folder, out, fps, gif_fps, width):
    pattern = os.path.join(folder, "frame_%05d.png")
    # One palette made for the whole clip, then used for every frame: far better colors
    # than ffmpeg's default GIF palette, and smaller files.
    graph = (f"fps={gif_fps},scale={width}:-1:flags=lanczos,split[a][b];"
             f"[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a")
    run_ffmpeg(["-framerate", str(fps), "-i", pattern, "-filter_complex", graph, "-loop", "0", out])


def main(argv):
    parser = argparse.ArgumentParser(description="planet_view --export images to a video.")
    parser.add_argument("images", help="folder written by planet_view --export")
    parser.add_argument("out", help="output video, e.g. smash.mp4")
    parser.add_argument("--fps", type=float, default=60.0,
                        help="frame rate the images were exported at (default 60)")
    parser.add_argument("--gif", help="also write a looping GIF here")
    parser.add_argument("--gif-width", type=int, default=640, help="GIF width in pixels")
    parser.add_argument("--gif-fps", type=float, default=20.0, help="GIF frame rate")
    args = parser.parse_args(argv)
    if not args.fps > 0 or not args.gif_fps > 0 or args.gif_width < 16:
        parser.error("--fps and --gif-fps need a number above 0, --gif-width at least 16")
    if not os.path.isdir(args.images):
        print(f"make_video.py: '{args.images}' is not a folder", file=sys.stderr)
        return 1
    for path in [args.out, args.gif]:
        if path is not None and not os.path.isdir(os.path.dirname(os.path.abspath(path))):
            print(f"make_video.py: the folder for '{path}' doesn't exist", file=sys.stderr)
            return 1
    try:
        count = len(numbered_images(args.images))
        make_mp4(args.images, args.out, args.fps)
        print(f"wrote {args.out}: {count} images at {args.fps:g} fps "
              f"({count / args.fps:.1f} s)")
        if args.gif is not None:
            make_gif(args.images, args.gif, args.fps, args.gif_fps, args.gif_width)
            print(f"wrote {args.gif}: {os.path.getsize(args.gif) / 1e6:.1f} MB")
    except ImportError as e:
        print(f"make_video.py: needs imageio-ffmpeg ({e}); "
              f"pip install -r scripts/requirements.txt", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as e:
        print(f"make_video.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
