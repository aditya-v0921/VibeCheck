import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest

from aiohttp import web


class ServerStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text())
        handler = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                       and node.name == "index_handler")
        namespace = {"web": web, "time": SimpleNamespace(monotonic=lambda: 100.0)}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(path), "exec"), namespace)
        self.handler = namespace["index_handler"]

    async def test_running_task_keeps_existing_response(self):
        task = asyncio.get_running_loop().create_future()
        try:
            response = await self.handler(SimpleNamespace(app={
                "vision_task": task, "capture_state": {"last_frame_at": 99.0}}))
            self.assertEqual(response.status, 200)
            self.assertEqual(response.text, "Vibe-Check Server Running")
            self.assertEqual(response.content_type, "text/plain")
        finally:
            task.cancel()

    async def test_missing_task_is_unavailable(self):
        response = await self.handler(SimpleNamespace(app={}))
        self.assertEqual(response.status, 503)
        self.assertEqual(response.text, "Vibe-Check vision unavailable")

    async def test_running_task_without_recent_frames_is_unavailable(self):
        task = asyncio.get_running_loop().create_future()
        try:
            for state in ({}, {"last_frame_at": None}, {"last_frame_at": 94.9}):
                with self.subTest(state=state):
                    response = await self.handler(SimpleNamespace(app={
                        "vision_task": task, "capture_state": state}))
                    self.assertEqual(response.status, 503)
        finally:
            task.cancel()

    async def test_fresh_frames_restore_status_at_timeout_boundary(self):
        task = asyncio.get_running_loop().create_future()
        state = {"last_frame_at": 90.0}
        request = SimpleNamespace(app={"vision_task": task, "capture_state": state})
        try:
            self.assertEqual((await self.handler(request)).status, 503)
            for timestamp in (95.0, 100.0):
                state["last_frame_at"] = timestamp
                self.assertEqual((await self.handler(request)).status, 200)
        finally:
            task.cancel()

    async def test_finished_tasks_are_unavailable(self):
        for state in ("completed", "failed", "cancelled"):
            with self.subTest(state=state):
                task = asyncio.get_running_loop().create_future()
                if state == "completed":
                    task.set_result(None)
                elif state == "failed":
                    task.set_exception(RuntimeError("Camera unavailable"))
                    task.exception()
                else:
                    task.cancel()
                response = await self.handler(SimpleNamespace(app={"vision_task": task}))
                self.assertEqual(response.status, 503)
                self.assertEqual(response.text, "Vibe-Check vision unavailable")


if __name__ == "__main__":
    unittest.main()
