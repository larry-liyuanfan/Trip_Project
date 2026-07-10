"""CLIP-based filtering for bounded Yelp image-review weak-alignment groups."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


DENOISED_PAIR_FIELDS = [
    "business_id",
    "photo_id",
    "image_path",
    "review_id",
    "review_text",
    "clip_similarity",
    "clip_model",
    "alignment_type",
]

ScoreCandidates = Callable[[list[dict[str, Any]], dict[str, Any]], list[float | None]]


def run_clip_denoising(
    weak_pairs: list[dict[str, Any]],
    config: dict[str, Any],
    score_candidates: ScoreCandidates | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score weak pairs in memory for tests and small local runs.

    The command-line runner uses :func:`stream_clip_denoising` to avoid keeping
    a full retained table in memory during the full Yelp run.
    """
    rows: list[dict[str, Any]] = []
    summary = stream_clip_denoising(weak_pairs, config, rows.append, score_candidates)
    return summary, rows


def stream_clip_denoising(
    weak_pairs: Iterable[dict[str, Any]],
    config: dict[str, Any],
    row_sink: Callable[[dict[str, Any]], None],
    score_candidates: ScoreCandidates | None = None,
) -> dict[str, Any]:
    """Expand, score, and retain candidates without accumulating output rows.

    Each weak business group is capped upstream, but its image-review Cartesian
    product can still contain hundreds of thousands of rows across Yelp.  This
    function keeps only one configurable candidate batch in memory.
    """
    threshold = float(config.get("threshold", 0.25))
    model_id = str(config.get("model_id", "openai/clip-vit-base-patch32"))

    if not config.get("enabled", False):
        input_groups, candidate_count = _count_groups_and_candidates(weak_pairs)
        return _skipped_summary(
            "clip_denoising.enabled is false",
            input_groups,
            candidate_count,
            threshold,
            model_id,
        )

    if score_candidates is None:
        try:
            scorer = _TransformersClipScorer(config)
        except ImportError as exc:
            input_groups, candidate_count = _count_groups_and_candidates(weak_pairs)
            return _skipped_summary(
                f"CLIP dependencies unavailable: {exc}",
                input_groups,
                candidate_count,
                threshold,
                model_id,
            )
        device = scorer.device
        score_candidates = lambda candidates, _: scorer(candidates)
    else:
        device = "injected"

    input_groups = 0
    candidate_count = 0
    retained_count = 0
    scored_count = 0
    score_sum = 0.0
    score_min: float | None = None
    score_max: float | None = None
    skipped_candidates = 0
    batch_size = int(config.get("candidate_batch_size", 256))

    def counted_candidate_rows() -> Iterable[dict[str, Any]]:
        nonlocal input_groups
        for group in weak_pairs:
            input_groups += 1
            yield from _iter_candidate_rows([group])

    for candidates in _batched(counted_candidate_rows(), batch_size):
        scores = score_candidates(candidates, config)
        if len(scores) != len(candidates):
            raise ValueError("CLIP scorer must return exactly one score per candidate")
        for candidate, score in zip(candidates, scores):
            candidate_count += 1
            if score is None:
                skipped_candidates += 1
                continue
            similarity = float(score)
            scored_count += 1
            score_sum += similarity
            score_min = similarity if score_min is None else min(score_min, similarity)
            score_max = similarity if score_max is None else max(score_max, similarity)
            if similarity < threshold:
                continue
            row_sink(
                {
                    **candidate,
                    "clip_similarity": similarity,
                    "clip_model": model_id,
                    "alignment_type": "weak_denoised",
                }
            )
            retained_count += 1

    return {
        "status": "completed",
        "reason": "CLIP scoring completed",
        "input_groups": input_groups,
        "input_pairs": candidate_count,
        "scored_pairs": scored_count,
        "skipped_candidates": skipped_candidates,
        "retained_pairs": retained_count,
        "threshold": threshold,
        "model_id": model_id,
        "device": device,
        "candidate_batch_size": batch_size,
        "similarity_distribution": _similarity_distribution(scored_count, score_sum, score_min, score_max),
    }


def _iter_candidate_rows(weak_pairs: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Produce the bounded image-review Cartesian product for each business."""
    for group in weak_pairs:
        business_id = group.get("business_id")
        photo_ids = group.get("photo_ids") or []
        image_paths = group.get("image_paths") or []
        review_ids = group.get("review_ids") or []
        review_texts = group.get("review_texts") or []
        for photo_id, image_path in zip(photo_ids, image_paths):
            if not photo_id or not image_path:
                continue
            for review_id, review_text in zip(review_ids, review_texts):
                text = str(review_text or "").strip()
                if not review_id or not text:
                    continue
                yield {
                    "business_id": business_id,
                    "photo_id": str(photo_id),
                    "image_path": str(image_path),
                    "review_id": str(review_id),
                    "review_text": text,
                }


def _count_groups_and_candidates(weak_pairs: Iterable[dict[str, Any]]) -> tuple[int, int]:
    group_count = 0
    candidate_count = 0
    for group in weak_pairs:
        group_count += 1
        candidate_count += sum(1 for _ in _iter_candidate_rows([group]))
    return group_count, candidate_count


def _batched(rows: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("candidate_batch_size must be at least 1")
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _skipped_summary(
    reason: str,
    input_groups: int,
    candidate_count: int,
    threshold: float,
    model_id: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "input_groups": input_groups,
        "input_pairs": candidate_count,
        "scored_pairs": 0,
        "skipped_candidates": 0,
        "retained_pairs": 0,
        "threshold": threshold,
        "model_id": model_id,
        "device": "not_started",
        "similarity_distribution": "not_available",
    }


def _similarity_distribution(
    count: int,
    total: float,
    minimum: float | None,
    maximum: float | None,
) -> dict[str, float | int] | str:
    if not count or minimum is None or maximum is None:
        return "not_available"
    return {
        "count": count,
        "min": minimum,
        "mean": total / count,
        "max": maximum,
    }


class _TransformersClipScorer:
    """Batch CLIP scorer that deduplicates image and text encodes per batch."""

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        requested_device = str(config.get("device", "cuda"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CLIP requested CUDA, but no CUDA device is available")
        self.torch = torch
        self.device = requested_device if requested_device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_batch_size = int(config.get("image_batch_size", 32))
        self.text_batch_size = int(config.get("text_batch_size", 128))
        model_id = str(config.get("model_id", "openai/clip-vit-base-patch32"))
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, candidates: list[dict[str, Any]]) -> list[float | None]:
        image_paths = list(dict.fromkeys(row["image_path"] for row in candidates))
        texts = list(dict.fromkeys(row["review_text"] for row in candidates))
        image_embeddings = self._image_embeddings(image_paths)
        text_embeddings = self._text_embeddings(texts)
        scores: list[float | None] = []
        for candidate in candidates:
            image_embedding = image_embeddings.get(candidate["image_path"])
            text_embedding = text_embeddings.get(candidate["review_text"])
            if image_embedding is None or text_embedding is None:
                scores.append(None)
                continue
            scores.append(float((image_embedding * text_embedding).sum().item()))
        return scores

    def _image_embeddings(self, image_paths: list[str]) -> dict[str, Any]:
        from PIL import Image

        embeddings: dict[str, Any] = {}
        for batch_paths in _plain_batches(image_paths, self.image_batch_size):
            images: list[Any] = []
            usable_paths: list[str] = []
            for image_path in batch_paths:
                try:
                    with Image.open(Path(image_path)) as image:
                        images.append(image.convert("RGB"))
                    usable_paths.append(image_path)
                except (OSError, ValueError):
                    continue
            if not images:
                continue
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                vectors = self.model.get_image_features(**inputs)
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1).cpu()
            embeddings.update(zip(usable_paths, vectors))
        return embeddings

    def _text_embeddings(self, texts: list[str]) -> dict[str, Any]:
        embeddings: dict[str, Any] = {}
        for batch_texts in _plain_batches(texts, self.text_batch_size):
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.inference_mode():
                vectors = self.model.get_text_features(**inputs)
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1).cpu()
            embeddings.update(zip(batch_texts, vectors))
        return embeddings


def _plain_batches(values: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size < 1:
        raise ValueError("CLIP batch sizes must be at least 1")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]
