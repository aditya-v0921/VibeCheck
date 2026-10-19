import unittest

import numpy as np

from vibe_core import MotionEngine


class GridValidationTests(unittest.TestCase):
    def test_overflowing_grid_products_are_rejected(self):
        for height, width in ((65536, 65536), (2, 2147483647), (2147483647, 2)):
            with self.subTest(height=height, width=width):
                with self.assertRaisesRegex(ValueError, "cell count"):
                    MotionEngine(height, width)

    def test_nonpositive_grid_dimensions_are_rejected(self):
        for height, width in ((0, 8), (8, 0), (-1, 8), (8, -1)):
            with self.subTest(height=height, width=width):
                with self.assertRaisesRegex(ValueError, "positive"):
                    MotionEngine(height, width)

    def test_rectangular_grids_keep_their_shape(self):
        engine = MotionEngine(3, 5)
        frame = np.zeros((24, 40, 3), dtype=np.uint8)
        engine.process(frame)
        heatmap, energy = engine.process(frame)
        self.assertEqual(heatmap.shape, (3, 5))
        self.assertEqual(energy, 0)
        self.assertEqual(engine.get_stats()["grid_size"], (3, 5))


class FrameValidationTests(unittest.TestCase):
    def test_empty_frames_are_rejected(self):
        for shape in ((0, 32, 3), (32, 0, 3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "positive"):
                    MotionEngine().process(np.zeros(shape, dtype=np.uint8))

    def test_unsupported_strides_are_rejected(self):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        for view in (frame[:, ::2], frame[:, ::-1], frame[::-1], frame[:, :, ::-1]):
            with self.subTest(strides=view.strides):
                with self.assertRaisesRegex(ValueError, "memory layout"):
                    MotionEngine().process(view)

    def test_padded_rows_match_contiguous_frames(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
        padded_engine = MotionEngine()
        contiguous_engine = MotionEngine()
        for image in (frame, np.roll(frame, 2, axis=1)):
            padded = padded_engine.process(image[:, 8:40])
            contiguous = contiguous_engine.process(image[:, 8:40].copy())
            np.testing.assert_allclose(padded[0], contiguous[0])
            self.assertAlmostEqual(padded[1], contiguous[1])


class FrameResizeTests(unittest.TestCase):
    def test_resize_restarts_flow_and_clears_smoothing(self):
        rng = np.random.default_rng(7)
        original = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
        for shape in ((48, 64, 3), (32, 80, 3), (48, 80, 3)):
            with self.subTest(shape=shape):
                engine = MotionEngine()
                engine.process(original)
                _, energy = engine.process(np.roll(original, 3, axis=1))
                self.assertGreater(energy, 0)
                resized = rng.integers(0, 256, shape, dtype=np.uint8)
                heatmap, energy = engine.process(resized)
                np.testing.assert_array_equal(heatmap, np.zeros((8, 8)))
                self.assertEqual(energy, 0)
                self.assertEqual(engine.get_stats()["frame_count"], 1)
                fresh_engine = MotionEngine()
                fresh_engine.process(resized)
                expected_heatmap, expected_energy = fresh_engine.process(resized)
                heatmap, energy = engine.process(resized)
                np.testing.assert_allclose(heatmap, expected_heatmap)
                self.assertAlmostEqual(energy, expected_energy)
                self.assertEqual(engine.get_stats()["frame_count"], 2)


class SmallFrameTests(unittest.TestCase):
    def test_frames_smaller_than_grid_have_empty_excess_cells(self):
        rng = np.random.default_rng(19)
        for height, width in ((1, 1), (3, 5), (7, 16), (16, 7)):
            with self.subTest(height=height, width=width):
                engine = MotionEngine()
                frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                engine.process(frame)
                heatmap, energy = engine.process(np.roll(frame, 1, axis=1))
                self.assertEqual(heatmap.shape, (8, 8))
                self.assertTrue(np.isfinite(heatmap).all())
                self.assertTrue(np.isfinite(energy))
                self.assertTrue((heatmap >= 0).all())
                self.assertAlmostEqual(energy, float(heatmap.mean()), places=6)
                np.testing.assert_array_equal(heatmap[height:, :], 0)
                np.testing.assert_array_equal(heatmap[:, width:], 0)


if __name__ == "__main__":
    unittest.main()
