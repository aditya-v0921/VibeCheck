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
        namespace = {"web": web}
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(path), "exec"), namespace)
        self.handler = namespace["index_handler"]

    async def test_running_task_keeps_existing_response(self):
        task = asyncio.get_running_loop().create_future()
        try:
            response = await self.handler(SimpleNamespace(app={"vision_task": task}))
            self.assertEqual(response.status, 200)
            self.assertEqual(response.text, "Vibe-Check Server Running")
            self.assertEqual(response.content_type, "text/plain")
        finally:
            task.cancel()

    async def test_missing_task_is_unavailable(self):
        response = await self.handler(SimpleNamespace(app={}))
        self.assertEqual(response.status, 503)
        self.assertEqual(response.text, "Vibe-Check vision unavailable")

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
