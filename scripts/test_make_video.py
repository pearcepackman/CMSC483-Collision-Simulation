"""Tests for make_video.py. Run with: python -m unittest discover scripts"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_video  # noqa: E402

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def run_main(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = make_video.main(args)
    return code, out.getvalue(), err.getvalue()


class MakeVideoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.images = os.path.join(self.dir, "images")
        os.mkdir(self.images)

    def tearDown(self):
        self.tmp.cleanup()

    def write_images(self, numbers, size=(48, 64)):
        import matplotlib.image

        for k in numbers:
            pixels = np.zeros(size + (3,))
            pixels[:, : 4 * (k + 1)] = [1.0, 0.5, 0.2]  # a bar that grows each frame
            matplotlib.image.imsave(os.path.join(self.images, "frame_%05d.png" % k), pixels)

    def test_numbering(self):
        self.write_images([0, 1, 2])
        with open(os.path.join(self.images, "notes.txt"), "w") as f:
            f.write("other files are ignored")
        self.assertEqual(make_video.numbered_images(self.images), [0, 1, 2])

    def test_gap_or_no_images_exits_1(self):
        code, _, err = run_main([self.images, os.path.join(self.dir, "v.mp4")])
        self.assertEqual(code, 1)
        self.assertIn("no frame_*.png", err)
        self.write_images([0, 1, 3])
        code, _, err = run_main([self.images, os.path.join(self.dir, "v.mp4")])
        self.assertEqual(code, 1)
        self.assertIn("frame 2 is missing", err)
        code, _, err = run_main([os.path.join(self.dir, "nope"), os.path.join(self.dir, "v.mp4")])
        self.assertEqual(code, 1)

    def test_bad_command_line_exits_2(self):
        with self.assertRaises(SystemExit) as e:
            run_main([self.images, "v.mp4", "--fps", "0"])
        self.assertEqual(e.exception.code, 2)

    @unittest.skipIf(imageio_ffmpeg is None, "imageio-ffmpeg not installed")
    def test_makes_mp4_and_gif(self):
        self.write_images([0, 1, 2, 3, 4], size=(49, 65))  # odd sizes: cropped to even
        mp4 = os.path.join(self.dir, "v.mp4")
        gif = os.path.join(self.dir, "v.gif")
        code, out, err = run_main([self.images, mp4, "--fps", "10", "--gif", gif,
                                   "--gif-width", "32", "--gif-fps", "10"])
        self.assertEqual(code, 0, err)
        n, _ = imageio_ffmpeg.count_frames_and_secs(mp4)
        self.assertEqual(n, 5)
        with open(gif, "rb") as f:
            self.assertEqual(f.read(6), b"GIF89a")
        self.assertIn("5 images at 10 fps (0.5 s)", out)


if __name__ == "__main__":
    unittest.main()
