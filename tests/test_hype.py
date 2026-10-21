import ast
from pathlib import Path
import unittest

import numpy as np


class HypeScoreTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(path.read_text())
        constants = {"MIN_PEOPLE_FOR_HYPE", "FULL_CROWD_SIZE", "WEIGHT_HANDS_UP",
                     "WEIGHT_JUMPING", "WEIGHT_MOVEMENT"}
        nodes = [node for node in tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name == "calculate_crowd_hype")
                 or (isinstance(node, ast.Assign) and any(
                     isinstance(target, ast.Name) and target.id in constants
                     for target in node.targets))]
        namespace = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        self.calculate = namespace["calculate_crowd_hype"]
        self.full_crowd = namespace["FULL_CROWD_SIZE"]

    def test_full_engagement_saturates_at_one(self):
        person = {"hands_up": 1.0, "jumping": 1.0, "arm_movement": 1.0,
                  "is_hyped": True}
        for count in (self.full_crowd, self.full_crowd * 2):
            with self.subTest(count=count):
                result = self.calculate([person] * count, count, 15.0)
                self.assertEqual(result["hype_score"], 1.0)
                self.assertEqual(result["hyped_people"], count)

    def test_low_energy_score_keeps_existing_scale(self):
        person = {"hands_up": 0.0, "jumping": 0.0, "arm_movement": 0.0,
                  "is_hyped": False}
        result = self.calculate([person] * self.full_crowd, self.full_crowd, 1.0)
        self.assertAlmostEqual(result["hype_score"], 0.0896)

    def test_empty_crowd_has_zero_hype(self):
        result = self.calculate([], 0, 15.0)
        self.assertEqual(result["hype_score"], 0)
        self.assertEqual(result["hyped_people"], 0)


if __name__ == "__main__":
    unittest.main()
