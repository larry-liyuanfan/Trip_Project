import unittest
from pathlib import Path


class _FakeDataType:
    INT64 = "INT64"
    VARCHAR = "VARCHAR"
    FLOAT_VECTOR = "FLOAT_VECTOR"
    FLOAT = "FLOAT"


class _FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class _FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _FakeClient:
    def __init__(self):
        self.schema = None
        self.index_params = None
        self.inserted = []
        self.search_kwargs = None
        self.delete_kwargs = None

    def has_collection(self, **kwargs):
        return False

    def create_schema(self, **kwargs):
        self.schema = _FakeSchema()
        return self.schema

    def create_collection(self, **kwargs):
        self.create_collection_kwargs = kwargs

    def prepare_index_params(self):
        self.index_params = _FakeIndexParams()
        return self.index_params

    def create_index(self, **kwargs):
        self.create_index_kwargs = kwargs

    def insert(self, **kwargs):
        self.inserted.extend(kwargs["data"])
        return {"insert_count": len(kwargs["data"])}

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [[]]

    def delete(self, **kwargs):
        self.delete_kwargs = kwargs
        return {"delete_count": 1}


class _FakeSDK:
    DataType = _FakeDataType


class MilvusVectorStoreTest(unittest.TestCase):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    def setUp(self):
        from src.retrieval.milvus_vectors import (
            OTAMilvusVectorStore,
            load_milvus_config,
        )

        self.config = load_milvus_config(
            self.PROJECT_ROOT / "configs/milvus_week4.yaml"
        )
        self.client = _FakeClient()
        self.store = OTAMilvusVectorStore(
            self.config,
            client=self.client,
            sdk=_FakeSDK,
        )
        self.vector = [0.0] * 511 + [1.0]
        self.entity = {
            "business_id": "business-1",
            "image_id": "image-1",
            "multimodal_vector": self.vector,
            "business_category": "restaurant",
            "city": "Shanghai",
            "star_rating": 4.5,
            "price_range": "mid_range",
            "image_type": "business_photo",
            "embedding_model": "openai/clip-vit-base-patch32",
        }

    def test_collection_schema_contains_fixed_fields_and_auto_primary_key(self):
        self.store.create_collection()

        fields = {field["field_name"]: field for field in self.client.schema.fields}
        self.assertEqual(
            set(fields),
            {
                "vector_id",
                "business_id",
                "image_id",
                "multimodal_vector",
                "business_category",
                "city",
                "star_rating",
                "price_range",
                "image_type",
                "embedding_model",
            },
        )
        self.assertTrue(fields["vector_id"]["auto_id"])
        self.assertEqual(fields["multimodal_vector"]["dim"], 512)

    def test_build_indexes_uses_configured_hnsw_cosine_and_scalar_indexes(self):
        self.store.build_indexes()

        vector_index = self.client.index_params.indexes[0]
        self.assertEqual(vector_index["index_type"], "HNSW")
        self.assertEqual(vector_index["metric_type"], "COSINE")
        self.assertEqual(vector_index["params"], {"M": 16, "efConstruction": 128})
        scalar_fields = {
            item["field_name"] for item in self.client.index_params.indexes[1:]
        }
        self.assertEqual(
            scalar_fields,
            set(self.config["index"]["scalar_fields"]),
        )

    def test_insert_search_and_delete_are_validated_and_filter_allowlisted(self):
        from src.retrieval.milvus_vectors import MilvusVectorError

        self.assertEqual(self.store.insert_one(self.entity)["insert_count"], 1)
        self.store.search(
            self.vector,
            top_k=3,
            filters={"city": "Shanghai", "business_category": ["restaurant", "hotel"]},
        )
        expression = self.client.search_kwargs["filter"]
        self.assertIn('city == "Shanghai"', expression)
        self.assertIn('business_category in ["restaurant", "hotel"]', expression)
        self.assertEqual(
            self.client.search_kwargs["search_params"]["params"]["ef"],
            64,
        )
        self.store.delete({"image_id": "image-1"})
        self.assertEqual(self.client.delete_kwargs["filter"], 'image_id == "image-1"')
        with self.assertRaises(MilvusVectorError):
            self.store.search(self.vector, filters={"raw_expression": "true"})

    def test_rejects_wrong_dimension_and_non_normalized_vectors(self):
        from src.retrieval.milvus_vectors import MilvusVectorError

        wrong = dict(self.entity)
        wrong["multimodal_vector"] = [1.0]
        with self.assertRaisesRegex(MilvusVectorError, "512"):
            self.store.insert_one(wrong)
        wrong["multimodal_vector"] = [1.0] * 512
        with self.assertRaisesRegex(MilvusVectorError, "normalized"):
            self.store.insert_one(wrong)


if __name__ == "__main__":
    unittest.main()
