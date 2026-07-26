"""Gold-independent deterministic coding for minimal-baseline text outputs."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


SUPPORTED_SCENARIOS = frozenset(
    {"image_product_search", "after_sales", "itinerary_planning"}
)


class SemanticCodingConfigurationError(ValueError):
    """Raised when the immutable semantic-coding contract is invalid."""


class BaselineSemanticCoder:
    """Map raw baseline text to fields using only a fixed checked codebook."""

    def __init__(self, config: dict[str, Any], *, codebook_sha256: str):
        self._config = _validate_config(config)
        self.version = self._config["version"]
        self.allowed_prompt_version = self._config["allowed_prompt_version"]
        self.codebook_sha256 = codebook_sha256

    @classmethod
    def from_path(cls, path: Path) -> "BaselineSemanticCoder":
        raw = Path(path).read_bytes()
        try:
            config = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise SemanticCodingConfigurationError(
                f"invalid semantic coding config: {exc}"
            ) from exc
        # Git 文本检出可能使用 CRLF；codebook 身份按仓库 LF 内容稳定计算。
        canonical_raw = raw.replace(b"\r\n", b"\n")
        return cls(
            config,
            codebook_sha256=hashlib.sha256(canonical_raw).hexdigest(),
        )

    def encode(self, *, scenario: str, raw_output: str) -> dict[str, Any]:
        """Return a prediction without accepting sample metadata or human gold."""
        if scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(f"unsupported semantic coding scenario: {scenario}")
        if not isinstance(raw_output, str):
            raise TypeError("raw_output must be text")
        text = _normalize(raw_output)
        scenario_config = self._config["scenarios"][scenario]
        prediction: dict[str, Any] = {}

        for field, labels in scenario_config.get("scalar_fields", {}).items():
            matched = _matched_labels(text, labels)
            prediction[field] = matched[0] if len(matched) == 1 else "unknown"
        for field, labels in scenario_config.get("multilabel_fields", {}).items():
            prediction[field] = _matched_labels(text, labels)

        if scenario == "after_sales":
            ocr_config = scenario_config["ocr"]
            prediction["ocr_text"] = (
                None
                if any(
                    _contains_term(text, _normalize(term))
                    for term in ocr_config["negative_terms"]
                )
                else _extract_ascii_tokens(
                    raw_output,
                    pattern=ocr_config["ascii_token_pattern"],
                    max_items=ocr_config["max_items"],
                )
            )
        elif scenario == "itinerary_planning":
            prediction["constraint_check"] = []
        return prediction


def _validate_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise SemanticCodingConfigurationError("semantic coding config must be an object")
    if config.get("version") != "baseline_semantic_coding_v1":
        raise SemanticCodingConfigurationError(
            "semantic coding version must be baseline_semantic_coding_v1"
        )
    if config.get("allowed_prompt_version") != "baseline_minimal_v1":
        raise SemanticCodingConfigurationError(
            "semantic coding is restricted to baseline_minimal_v1"
        )
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != SUPPORTED_SCENARIOS:
        raise SemanticCodingConfigurationError(
            "semantic coding config must define exactly the three Week 3 scenarios"
        )
    for scenario, scenario_config in scenarios.items():
        if not isinstance(scenario_config, dict):
            raise SemanticCodingConfigurationError(
                f"scenario config must be an object: {scenario}"
            )
        for group_name in ("scalar_fields", "multilabel_fields"):
            group = scenario_config.get(group_name, {})
            if not isinstance(group, dict):
                raise SemanticCodingConfigurationError(
                    f"{scenario}.{group_name} must be an object"
                )
            for field, labels in group.items():
                if not isinstance(field, str) or not field or not isinstance(labels, dict):
                    raise SemanticCodingConfigurationError(
                        f"{scenario}.{group_name} fields must map to objects"
                    )
                for canonical, terms in labels.items():
                    if (
                        not isinstance(canonical, str)
                        or not canonical
                        or not isinstance(terms, list)
                        or not terms
                        or any(not isinstance(term, str) or not term for term in terms)
                    ):
                        raise SemanticCodingConfigurationError(
                            f"invalid lexical rule: {scenario}.{field}.{canonical}"
                        )
    ocr = scenarios["after_sales"].get("ocr")
    if (
        not isinstance(ocr, dict)
        or not isinstance(ocr.get("negative_terms"), list)
        or not ocr["negative_terms"]
        or any(
            not isinstance(term, str) or not term
            for term in ocr["negative_terms"]
        )
        or not isinstance(ocr.get("ascii_token_pattern"), str)
        or not isinstance(ocr.get("max_items"), int)
        or not 0 < ocr["max_items"] <= 100
    ):
        raise SemanticCodingConfigurationError("after_sales.ocr is invalid")
    try:
        re.compile(ocr["ascii_token_pattern"])
    except re.error as exc:
        raise SemanticCodingConfigurationError(
            f"after_sales.ocr pattern is invalid: {exc}"
        ) from exc
    return config


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _matched_labels(text: str, labels: dict[str, list[str]]) -> list[str]:
    return sorted(
        canonical
        for canonical, terms in labels.items()
        if any(_contains_term(text, _normalize(term)) for term in terms)
    )


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if all(character.isascii() for character in term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def _extract_ascii_tokens(raw_output: str, *, pattern: str, max_items: int) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for match in re.finditer(pattern, raw_output):
        token = match.group(0)
        normalized = token.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(token)
        if len(tokens) >= max_items:
            break
    return tokens


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")
