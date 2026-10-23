import ast
import asyncio
import base64
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np


class EndOfFrames(Exception):
    pass


class VisionLoopTests(unittest.IsolatedAsyncioTestCase):
    async def run_frames(self, pose_results, frame_count, startup_failure=None,
                         encoding_results=None, emit_rate=20):
        # Run the actual loop without importing database or model startup.
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name in {"vision_loop", "calculate_crowd_hype", "classify_zones"}]
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        camera = Mock()
        camera.read.side_effect = [(True, frame.copy()) for _ in range(frame_count + 1)] + [EndOfFrames()]
        model = Mock()
        model.predict.side_effect = [Mock()] + [[result] for result in pose_results]
        engine = Mock()
        engine.process.return_value = (np.zeros((8, 8)), 3.0)
        cv2 = Mock()
        cv2.imencode.return_value = (True, b"jpeg")
        if encoding_results is not None:
            cv2.imencode.side_effect = encoding_results
        socket = SimpleNamespace(emit=AsyncMock())
        clock = Mock()
        clock.time.side_effect = itertools.count()
        metrics = {"hands_up": 1.0, "jumping": 0.0, "arm_movement": 0.0,
                   "is_hyped": True}
        namespace = {
            "np": np, "asyncio": asyncio, "base64": base64, "time": clock,
            "cv2": cv2, "sio": socket, "open_camera": Mock(return_value=camera),
            "vibe_core": SimpleNamespace(MotionEngine=Mock(return_value=engine)),
            "YOLO": Mock(return_value=model), "CAM_SOURCE": 1,
            "GRID_H": 8, "GRID_W": 8, "CONF_THRESH": 0.35,
            "POSE_EVERY_N_FRAMES": 2, "EMIT_RATE_HZ": emit_rate, "JPEG_QUALITY": 70,
            "MIN_PEOPLE_FOR_HYPE": 1, "FULL_CROWD_SIZE": 3,
            "WEIGHT_HANDS_UP": 0.4, "WEIGHT_JUMPING": 0.3, "WEIGHT_MOVEMENT": 0.35,
            "INFLUX_ENABLED": False, "influx_client": Mock(),
            "compute_person_metrics": Mock(return_value=metrics),
            "draw_skeleton": Mock(side_effect=lambda image, *args: image),
        }
        if startup_failure == "engine":
            namespace["vibe_core"].MotionEngine.side_effect = EndOfFrames()
        elif startup_failure == "model":
            namespace["YOLO"].side_effect = EndOfFrames()
        elif startup_failure == "warmup":
            model.predict.side_effect = EndOfFrames()
        elif startup_failure == "read":
            camera.read.side_effect = EndOfFrames()
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        with patch("builtins.print"), self.assertRaises(EndOfFrames):
            await namespace["vision_loop"]()
        camera.release.assert_called_once_with()
        namespace["influx_client"].close.assert_called_once_with()
        return [call.args[1] for call in socket.emit.await_args_list], namespace

    def pose(self):
        data = Mock()
        data.cpu.return_value.numpy.return_value = np.ones((1, 17, 3))
        return SimpleNamespace(keypoints=SimpleNamespace(data=data))

    async def test_startup_failures_release_camera_and_database(self):
        for stage in ("engine", "model", "warmup", "read"):
            with self.subTest(stage=stage):
                payloads, _ = await self.run_frames([], 0, startup_failure=stage)
                self.assertEqual(payloads, [])

    async def test_skipped_pose_frame_keeps_latest_detection(self):
        payloads, _ = await self.run_frames([self.pose()], 2)
        self.assertEqual(len(payloads), 2)
        self.assertEqual([payload["peopleCount"] for payload in payloads], [1, 1])
        self.assertEqual(payloads[0]["keypoints"], payloads[1]["keypoints"])
        self.assertEqual(len(payloads[1]["keypoints"]), 1)
        self.assertEqual([payload["hypedPeople"] for payload in payloads], [1, 1])
        self.assertEqual(payloads[0]["hypeScore"], payloads[1]["hypeScore"])

    async def test_failed_encoding_skips_frame_and_recovers(self):
        payloads, namespace = await self.run_frames(
            [self.pose()], 2, encoding_results=[(False, None), (True, b"jpeg")])
        self.assertEqual(namespace["cv2"].imencode.call_count, 2)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["frame"], base64.b64encode(b"jpeg").decode())
        self.assertEqual(payloads[0]["peopleCount"], 1)

    async def test_throttled_frames_skip_encoding(self):
        payloads, namespace = await self.run_frames([self.pose()], 2, emit_rate=0.000001)
        self.assertEqual(payloads, [])
        namespace["cv2"].imencode.assert_not_called()
        self.assertEqual(namespace["vibe_core"].MotionEngine.return_value.process.call_count, 2)

    async def test_encoding_resumes_when_emission_is_due(self):
        payloads, namespace = await self.run_frames([self.pose()], 2, emit_rate=0.2)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(namespace["cv2"].imencode.call_count, 1)
        self.assertEqual(payloads[0]["peopleCount"], 1)
        self.assertEqual(payloads[0]["frame"], base64.b64encode(b"jpeg").decode())

    async def test_missing_pose_clears_overlay_and_movement_history(self):
        empty_data = Mock()
        empty_data.cpu.return_value.numpy.return_value = np.empty((0, 17, 3))
        missing_results = [SimpleNamespace(keypoints=None),
                           SimpleNamespace(keypoints=SimpleNamespace(data=None)),
                           SimpleNamespace(keypoints=SimpleNamespace(data=empty_data))]
        for missing in missing_results:
            with self.subTest(missing=missing):
                payloads, namespace = await self.run_frames([self.pose(), missing, self.pose()], 5)
                self.assertEqual(len(payloads), 5)
                for payload in payloads[2:4]:
                    self.assertEqual(payload["peopleCount"], 0)
                    self.assertEqual(payload["keypoints"], [])
                    self.assertEqual(payload["hypedPeople"], 0)
                    self.assertEqual(payload["crowdMultiplier"], 0)
                self.assertEqual(namespace["draw_skeleton"].call_count, 3)
                self.assertIsNone(namespace["compute_person_metrics"].call_args.args[1])
                self.assertEqual(payloads[4]["peopleCount"], 1)


if __name__ == "__main__":
    unittest.main()
