import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.rebuild_week8_yelp_sources import (
    Week8YelpRebuildError,
    extract_unique_member,
)


class Week8YelpRebuildTests(unittest.TestCase):
    def test_extracts_one_exact_suffix_without_unpacking_other_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/yelp_academic_dataset_business.json", '{"id": 1}\n')
                handle.writestr("nested/large-unused.json", "unused")
            destination = root / "raw" / "business.json"
            evidence = extract_unique_member(
                archive, "yelp_academic_dataset_business.json", destination
            )
            self.assertEqual(destination.read_text(encoding="utf-8"), '{"id": 1}\n')
            self.assertEqual(
                evidence["member"], "nested/yelp_academic_dataset_business.json"
            )
            self.assertFalse((root / "raw" / "large-unused.json").exists())

    def test_rejects_ambiguous_archive_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("a/photos.json", "{}\n")
                handle.writestr("b/photos.json", "{}\n")
            with self.assertRaisesRegex(Week8YelpRebuildError, "found 2"):
                extract_unique_member(archive, "photos.json", root / "photos.json")


if __name__ == "__main__":
    unittest.main()

