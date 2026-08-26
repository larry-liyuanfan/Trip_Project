import tempfile
import io
import tarfile
import unittest
import zipfile
from pathlib import Path

from scripts.rebuild_week8_yelp_sources import (
    Week8YelpRebuildError,
    extract_unique_tar_member_from_zip,
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

    def test_streams_one_member_from_official_nested_tar_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested_bytes = io.BytesIO()
            with tarfile.open(fileobj=nested_bytes, mode="w") as nested:
                for name, content in (
                    ("dataset/yelp_academic_dataset_business.json", b'{"id": 1}\n'),
                    ("dataset/large-unused.json", b"unused"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    nested.addfile(info, io.BytesIO(content))
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("Yelp JSON/yelp_dataset.tar", nested_bytes.getvalue())
                handle.writestr("__MACOSX/Yelp JSON/._yelp_dataset.tar", b"metadata")
            destination = root / "raw" / "business.json"
            evidence = extract_unique_tar_member_from_zip(
                archive,
                tar_basename="yelp_dataset.tar",
                member_basename="yelp_academic_dataset_business.json",
                destination=destination,
            )
            self.assertEqual(destination.read_bytes(), b'{"id": 1}\n')
            self.assertEqual(evidence["outer_member"], "Yelp JSON/yelp_dataset.tar")
            self.assertEqual(
                evidence["member"], "dataset/yelp_academic_dataset_business.json"
            )


if __name__ == "__main__":
    unittest.main()
