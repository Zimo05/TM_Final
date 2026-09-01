import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SplitManifest import build_data_provenance, file_sha256, load_strict_manifest


class StrictManifestTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(root, data_path, splits):
        path = root / "split.json"
        path.write_text(json.dumps({
            "seed": 42,
            "data_sha256": file_sha256(data_path),
            "splits": splits,
        }), encoding="utf-8")
        return path

    def test_builds_complete_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            data_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            path = self._write_manifest(
                root, data_path,
                {"train": [0], "validation": [1], "test": [2]},
            )
            manifest = load_strict_manifest(
                path, data_path=data_path, available_source_ids=range(3)
            )
            provenance = build_data_provenance(manifest)
            self.assertEqual(provenance["evaluation_regime"], "strict_inductive")
            self.assertEqual(provenance["node_pool"], "train_only")
            self.assertEqual(provenance["train_source_ids"], [0])
            self.assertEqual(provenance["validation_source_ids"], [1])
            self.assertEqual(provenance["test_source_ids"], [2])

    def test_rejects_overlap(self):
        self._assert_rejected(
            {"train": [0, 2], "validation": [1], "test": [2]}, range(3)
        )

    def test_rejects_data_sha_mismatch(self):
        self._assert_rejected(
            {"train": [0], "validation": [1], "test": [2]},
            range(3), change_data=True,
        )

    def test_rejects_ambiguous_source_mapping(self):
        self._assert_rejected(
            {"train": [0], "validation": [1], "test": [2]}, range(4)
        )

    def _assert_rejected(self, splits, available, change_data=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            data_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
            path = self._write_manifest(root, data_path, splits)
            if change_data:
                data_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_strict_manifest(
                    path, data_path=data_path, available_source_ids=available
                )


if __name__ == "__main__":
    unittest.main()
