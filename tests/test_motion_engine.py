import unittest

import numpy as np

from vibe_core import MotionEngine


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


if __name__ == "__main__":
    unittest.main()
