import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_scenario_variants as variants


BASE_CONFIG = {
    "scene_dir": "new_grscenes/example",
    "output_root": "new_outputs",
    "llm_config": "configs/llm_config.local.yaml",
    "prompt": "Generate a route",
}


class RunConfigTests(unittest.TestCase):
    def write_config(self, data):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "run.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.addCleanup(temp_dir.cleanup)
        return path

    def test_config_loads_with_defaults(self):
        path = self.write_config(BASE_CONFIG)
        args = variants.parse_args(["--config", str(path)])
        self.assertEqual(args.scene_dir, BASE_CONFIG["scene_dir"])
        self.assertEqual(args.count, 5)
        self.assertEqual(args.max_attempts, 3)
        self.assertEqual(args.pedestrians, 2)
        self.assertFalse(args.overwrite)
        self.assertFalse(args.dry_run)

    def test_command_line_overrides_config(self):
        config = {**BASE_CONFIG, "count": 6, "overwrite": True, "dry_run": True}
        path = self.write_config(config)
        args = variants.parse_args(
            ["--config", str(path), "--count", "1", "--no-overwrite", "--no-dry-run"]
        )
        self.assertEqual(args.count, 1)
        self.assertFalse(args.overwrite)
        self.assertFalse(args.dry_run)

    def test_original_command_line_usage_still_works(self):
        args = variants.parse_args(
            [
                "--scene-dir", "scene",
                "--output-root", "outputs",
                "--llm-config", "llm.yaml",
                "--prompt", "Generate",
                "--pedestrians", "1",
            ]
        )
        self.assertEqual(args.scene_dir, "scene")
        self.assertEqual(args.pedestrians, 1)
        self.assertEqual(args.count, 5)

    def test_unknown_config_key_is_rejected(self):
        path = self.write_config({**BASE_CONFIG, "typo": 1})
        with self.assertRaises(SystemExit):
            variants.parse_args(["--config", str(path)])


if __name__ == "__main__":
    unittest.main()
