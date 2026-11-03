"""Tests for render.py. Run with: python -m unittest discover scripts"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def write_frame(folder, number, rows):
    """Writes a frame the way planet_sim's write_frame() does: header, then id,x,y,z,body."""
    path = os.path.join(folder, "frame_%05d.csv" % number)
    with open(path, "w", newline="\n") as f:
        f.write("id,x,y,z,body\n")
        for i, (x, y, z, body) in enumerate(rows):
            f.write("%d,%.17g,%.17g,%.17g,%d\n" % (i, x, y, z, body))
    return path


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write_text(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", newline="\n") as f:
            f.write(text)
        return path

    def test_loads_frame(self):
        path = write_frame(self.dir, 0, [(0.5, -1.0, 2.0, 0), (-0.56367743418916594, 0.0, 1e-300, 1)])
        xyz, body = render.load_frame(path)
        self.assertEqual(xyz.shape, (2, 3))
        self.assertEqual(xyz[0].tolist(), [0.5, -1.0, 2.0])
        self.assertEqual(xyz[1, 0], -0.56367743418916594)  # all 17 digits survive
        self.assertEqual(body.tolist(), [0, 1])

    def test_loads_single_particle_and_empty_frames(self):
        xyz, body = render.load_frame(write_frame(self.dir, 0, [(1.0, 2.0, 3.0, 4)]))
        self.assertEqual(xyz.shape, (1, 3))
        self.assertEqual(body.tolist(), [4])
        xyz, body = render.load_frame(write_frame(self.dir, 1, []))
        self.assertEqual(xyz.shape, (0, 3))

    def test_loads_what_read_frame_accepts(self):
        # The C++ reader (src/io.cpp) accepts these, so the script must too.
        good = {
            "crlf.csv": "id,x,y,z,body\r\n0,1.5,-2,3,1\r\n1,0,0,0,0\r\n",
            "no_final_newline.csv": "id,x,y,z,body\n0,1.5,-2,3,1\n1,0,0,0,0",
            "one_blank_line_at_end.csv": "id,x,y,z,body\n0,1.5,-2,3,1\n1,0,0,0,0\n\n",
        }
        for name, text in good.items():
            xyz, body = render.load_frame(self.write_text(name, text))
            self.assertEqual(xyz.shape, (2, 3), name)
            self.assertEqual(xyz[0].tolist(), [1.5, -2.0, 3.0], name)
            self.assertEqual(body.tolist(), [1, 0], name)
        # A run that blew up writes nan/inf; its frames must still load.
        xyz, _ = render.load_frame(write_frame(self.dir, 0, [(float("nan"), float("inf"), -float("inf"), 0)]))
        self.assertTrue(np.isnan(xyz[0, 0]))
        self.assertEqual(xyz[0, 1], float("inf"))

    def test_rejects_malformed(self):
        bad = {
            "empty.csv": "",
            "header.csv": "id,x,y,z,mass\n0,1,2,3,0\n",
            "columns.csv": "id,x,y,z,body\n0,1,2,3\n",
            "ids.csv": "id,x,y,z,body\n0,1,2,3,0\n2,1,2,3,0\n",
            "number.csv": "id,x,y,z,body\n0,1,abc,3,0\n",
            "body.csv": "id,x,y,z,body\n0,1,2,3,0.5\n",
            # Rejected by read_frame too: a hand-edited frame should fail, not render.
            "comment.csv": "id,x,y,z,body\n# edited\n0,1,2,3,0\n",
            "blank_middle.csv": "id,x,y,z,body\n0,1,2,3,0\n\n1,1,2,3,0\n",
            "two_blank_at_end.csv": "id,x,y,z,body\n0,1,2,3,0\n\n\n",
        }
        for name, text in bad.items():
            path = self.write_text(name, text)
            with self.assertRaises(ValueError, msg=name) as caught:
                render.load_frame(path)
            self.assertIn(name, str(caught.exception))  # the message says which file

    def test_frames_sorted_by_number(self):
        for n in [100000, 2, 99999, 10]:
            write_frame(self.dir, n, [(0.0, 0.0, 0.0, 0)])
        self.write_text("energy.csv", "step,t\n")  # other files in the folder are ignored
        self.write_text("frame_notes.txt", "")
        self.write_text("frame_00001.csv.bak", "")  # an editor's backup isn't a frame
        self.write_text("old_frame_00003.csv", "")
        names = [os.path.basename(p) for p in render.list_frames(self.dir)]
        self.assertEqual(names, ["frame_00002.csv", "frame_00010.csv", "frame_99999.csv",
                                 "frame_100000.csv"])

    def test_bounds_cover_every_frame(self):
        # Debris flies out in the last frame: the box must hold it, not just frame 0.
        a = write_frame(self.dir, 0, [(0.0, 0.0, 0.0, 0), (1.0, 1.0, 1.0, 0)])
        b = write_frame(self.dir, 1, [(-4.0, 0.5, 0.5, 0), (1.0, 3.0, 0.2, 1)])
        center, half = render.bounds([a, b])
        for path in [a, b]:
            xyz, _ = render.load_frame(path)
            for p in xyz:
                for k in range(3):  # a cube: the same half-width on every axis
                    self.assertLessEqual(abs(p[k] - center[k]), half)
        self.assertGreater(half, 0.0)
        # A single point still gets a box with some size.
        c = write_frame(self.dir, 2, [(5.0, 5.0, 5.0, 0)])
        center, half = render.bounds([c])
        self.assertGreater(half, 0.0)

    def test_bounds_use_the_widest_axis(self):
        # Widest along z: the cube must be sized by z, not x.
        a = write_frame(self.dir, 0, [(0.0, 0.0, -5.0, 0), (1.0, 1.0, 5.0, 0)])
        center, half = render.bounds([a])
        self.assertGreaterEqual(half, 5.0)
        self.assertAlmostEqual(center[2], 0.0)

    def test_bounds_ignore_a_few_stragglers(self):
        # 1,000 particles spread evenly through a unit cube, plus 3 flung far away: the
        # box should frame the planet, not shrink it to a dot to fit the stragglers.
        pts = np.random.default_rng(7).random((1000, 3))
        rows = [(p[0], p[1], p[2], 0) for p in pts]
        rows += [(1000.0, 0.0, 0.0, 1), (0.0, -500.0, 0.0, 1), (0.0, 0.0, 800.0, 1)]
        center, half = render.bounds([write_frame(self.dir, 0, rows)])
        # Dropping the farthest 0.5% per side keeps about 99% of the width, plus the 5%
        # margin: 0.5 * 0.99 * 1.05 = 0.52. Cutting 5% per side would give about 0.47.
        self.assertGreater(half, 0.5)
        self.assertLess(half, 0.55)
        for k in range(3):
            self.assertLess(abs(center[k] - 0.5), 0.05)

    def test_bounds_skip_nan_and_inf(self):
        # One blown-up particle must not wreck the camera for the whole video.
        a = write_frame(self.dir, 0, [(0.0, 0.0, 0.0, 0), (2.0, 2.0, 2.0, 0)])
        b = write_frame(self.dir, 1, [(1.0, 1.0, 1.0, 0), (float("nan"), float("inf"), 0.0, 0)])
        center, half = render.bounds([a, b])
        self.assertEqual(center.tolist(), [1.0, 1.0, 1.0])
        self.assertAlmostEqual(half, 1.05)  # half of 2, plus the 5% margin
        # Nothing finite at all: a default box instead of a crash.
        c = write_frame(self.dir, 2, [(float("nan"), 0.0, 0.0, 0)])
        center, half = render.bounds([c])
        self.assertEqual(center.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(half, 1.0)

    @unittest.skipIf(imageio_ffmpeg is None, "scripts/requirements.txt not installed")
    def test_renders_video(self):
        frames = os.path.join(self.dir, "run")
        os.makedirs(frames)
        for k in range(3):
            write_frame(frames, k, [(0.1 * k, 0.0, 0.0, 0), (0.0, 0.2 * k, 0.0, 1), (0.0, 0.0, 1.0, 1)])
        out = os.path.join(self.dir, "out.mp4")
        still = os.path.join(self.dir, "last.png")
        self.assertEqual(render.main([frames, "-o", out, "--still", still, "--fps", "10"]), 0)
        n_frames, seconds = imageio_ffmpeg.count_frames_and_secs(out)
        self.assertEqual(n_frames, 3)
        self.assertAlmostEqual(seconds, 0.3, delta=0.05)  # 3 frames at --fps 10
        with open(still, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")  # a real PNG

    @unittest.skipIf(imageio_ffmpeg is None, "scripts/requirements.txt not installed")
    def test_every_skips_frames(self):
        frames = os.path.join(self.dir, "run")
        os.makedirs(frames)
        for k in range(5):
            write_frame(frames, k, [(0.1 * k, 0.0, 0.0, 0)])
        out = os.path.join(self.dir, "out.mp4")
        self.assertEqual(render.main([frames, "-o", out, "--every", "2"]), 0)
        n_frames, _ = imageio_ffmpeg.count_frames_and_secs(out)
        self.assertEqual(n_frames, 3)  # frames 0, 2, 4

    def test_main_errors(self):
        out = os.path.join(self.dir, "out.mp4")
        self.assertEqual(render.main([self.dir, "-o", out]), 1)  # no frames in the folder
        self.assertEqual(render.main([os.path.join(self.dir, "missing"), "-o", out]), 1)
        write_frame(self.dir, 0, [(0.0, 0.0, 0.0, 0)])
        self.write_text("frame_00001.csv", "id,x,y,z,mass\n")  # one bad frame stops the render
        self.assertEqual(render.main([self.dir, "-o", out]), 1)
        self.assertFalse(os.path.exists(out))

    def test_bad_output_paths_fail_before_rendering(self):
        frames = os.path.join(self.dir, "run")
        os.makedirs(frames)
        write_frame(frames, 0, [(0.0, 0.0, 0.0, 0)])
        missing = os.path.join(self.dir, "no_such_folder")
        out = os.path.join(self.dir, "out.mp4")
        self.assertEqual(render.main([frames, "-o", os.path.join(missing, "x.mp4")]), 1)
        # A bad --still folder is found before the video is encoded, not after.
        self.assertEqual(render.main([frames, "-o", out, "--still", os.path.join(missing, "s.png")]), 1)
        self.assertFalse(os.path.exists(out))

    @unittest.skipIf(imageio_ffmpeg is None, "scripts/requirements.txt not installed")
    def test_ffmpeg_failure_is_reported(self):
        frames = os.path.join(self.dir, "run")
        os.makedirs(frames)
        write_frame(frames, 0, [(0.0, 0.0, 0.0, 0)])
        # No extension: ffmpeg can't tell which format to write and fails.
        self.assertEqual(render.main([frames, "-o", os.path.join(self.dir, "noext")]), 1)

    def test_bad_numbers_are_usage_errors(self):
        for flag in ["--fps", "--every"]:
            with self.assertRaises(SystemExit) as caught:
                render.main([self.dir, "-o", "out.mp4", flag, "0"])
            self.assertEqual(caught.exception.code, 2, flag)  # exit 2 = bad command line


if __name__ == "__main__":
    unittest.main()
