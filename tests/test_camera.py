import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, call, patch


class CameraDetectionTests(unittest.TestCase):
    def setUp(self):
        # Load the camera helper without starting database or model setup.
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "open_camera")
        self.cv2 = Mock()
        namespace = {"cv2": self.cv2, "PREFERRED_RESOLUTION": (1280, 720),
                     "PREFERRED_FPS": 30}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"),
             namespace)
        self.open_camera = namespace["open_camera"]
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def capture(self, opened, readable=False):
        capture = Mock()
        capture.isOpened.return_value = opened
        capture.read.return_value = (readable, None)
        capture.get.return_value = 640
        return capture

    def test_no_readable_camera_releases_all_probes(self):
        captures = [self.capture(i % 2 == 0) for i in range(10)]
        self.cv2.VideoCapture.side_effect = captures
        with self.assertRaisesRegex(RuntimeError, "available camera"):
            self.open_camera("auto")
        self.assertEqual(self.cv2.VideoCapture.call_count, 10)
        for capture in captures:
            capture.release.assert_called_once_with()

    def test_successful_probe_is_released_before_reopening(self):
        captures = [self.capture(False), self.capture(True),
                    self.capture(True, True), self.capture(True, True)]
        self.cv2.VideoCapture.side_effect = captures
        self.assertIs(self.open_camera("auto"), captures[-1])
        self.assertEqual(self.cv2.VideoCapture.call_args_list,
                         [call(i, self.cv2.CAP_AVFOUNDATION) for i in (0, 1, 2, 2)])
        for capture in captures[:-1]:
            capture.release.assert_called_once_with()
        captures[-1].release.assert_not_called()

    def test_probe_read_error_releases_handle(self):
        capture = self.capture(True)
        capture.read.side_effect = RuntimeError("Camera read failed")
        self.cv2.VideoCapture.return_value = capture
        with self.assertRaisesRegex(RuntimeError, "Camera read failed"):
            self.open_camera("auto")
        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
