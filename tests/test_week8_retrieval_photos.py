import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.extract_week8_retrieval_photos import (
    Week8RetrievalPhotoError,
    extract_overlay,
    load_photo_ids,
)


class Week8RetrievalPhotoTests(unittest.TestCase):
    def test_load_photo_ids_binds_formal_source_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "image_id": "photo-1",
                        "source_image_path": "data/yelp/raw/photos/photo-1.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_photo_ids(metadata, 1), {"photo-1"})
            with self.assertRaisesRegex(Week8RetrievalPhotoError, "count mismatch"):
                load_photo_ids(metadata, 2)

    def test_extracts_complete_isolated_overlay_from_nested_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.jsonl"
            metadata.write_text(
                json.dumps(
                    {
                        "image_id": "photo-1",
                        "source_image_path": "data/yelp/raw/photos/photo-1.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tar_bytes = io.BytesIO()
            content = b"jpeg-bytes"
            with tarfile.open(fileobj=tar_bytes, mode="w:gz") as archive:
                info = tarfile.TarInfo("photos/photo-1.jpg")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            photos_zip = root / "photos.zip"
            with zipfile.ZipFile(photos_zip, "w") as archive:
                archive.writestr("Yelp Photos/yelp_photos.tar", tar_bytes.getvalue())
            output = root / "overlay"
            result = extract_overlay(
                metadata_path=metadata,
                photos_zip=photos_zip,
                output_root=output,
                expected_count=1,
                expected_zip_size=photos_zip.stat().st_size,
            )
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["extracted_photo_count"], 1)
            self.assertEqual(
                (output / "data/yelp/raw/photos/photo-1.jpg").read_bytes(), content
            )


if __name__ == "__main__":
    unittest.main()
