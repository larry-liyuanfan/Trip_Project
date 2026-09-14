"""Audit v9 text clipping and export repaired paired inputs, not independent test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_exploration_pool_v4 import FONT, WIDTH, _write_card
from scripts.build_semantic_robustness_pool_v9 import (
    COUNTS, SPLIT_OFFSET, _clear_product, _unknown_product,
    _multi_product_v9, _negated_product, _conflict_product,
)
from src.evaluation.relevance_evidence import file_sha256, canonical_json_sha256
from src.evaluation.synthetic_card_layout import layout_card, write_card


def audit(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    builders = {"clear": _clear_product, "unknown": _unknown_product,
                "multi": _multi_product_v9, "negated": _negated_product,
                "conflict": _conflict_product}
    capacity = 0
    cursor = 20
    while True:
        capacity += 1
        cursor += 18
        if cursor + 15 >= WIDTH:
            break
    for split in ("training", "development"):
        cursor = 0
        for kind, builder in builders.items():
            for index in range(COUNTS[split][kind]):
                lines, gold, _ = builder(index, SPLIT_OFFSET[split])
                lines[0] = "TRIP V9 ROBUST"
                lines.insert(1, f"SPLIT {split.upper()}")
                sample_id = f"v9_{split[:3]}_{kind}_{index:03d}"
                clipped = [{"line": i, "full": line, "visible": line[:capacity] if i < 7 else ""}
                           for i, line in enumerate(lines) if i >= 7 or len(line) > capacity]
                layout = layout_card(lines)
                old_path = output_dir / "original" / f"{sample_id}.ppm"
                new_path = output_dir / "repaired" / f"{sample_id}.ppm"
                seed = SPLIT_OFFSET[split] + cursor
                _write_card(old_path, lines, seed)
                write_card(new_path, lines, seed)
                rows.append({"sample_id": sample_id, "split": split, "kind": kind,
                             "unsupported_legacy_glyphs": sorted(set("".join(lines)) - set(FONT)),
                             "clipped_lines": clipped, "gold_sha256": canonical_json_sha256(gold),
                             "original_image_sha256": file_sha256(old_path),
                             "repaired_image_sha256": file_sha256(new_path), "layout": layout})
                cursor += 1
    summary = {"schema_version": "synthetic_card_layout_audit_v10",
               "evidence_class": "synthetic_rendering_diagnostic_not_independent_quality_evaluation",
               "fresh_test_used": False, "model_inference_run": False,
               "legacy_character_capacity": capacity,
               "scope": "v9_training_and_already_seen_development_product_cards_only",
               "splits": {split: {"support": sum(r["split"] == split for r in rows),
                                  "clipped_card_support": sum(r["split"] == split and bool(r["clipped_lines"]) for r in rows),
                                  "unsupported_glyph_card_support": sum(r["split"] == split and bool(r["unsupported_legacy_glyphs"]) for r in rows),
                                  "clipped_line_support": sum(len(r["clipped_lines"]) for r in rows if r["split"] == split)}
                          for split in ("training", "development")},
               "repaired_layout_overflow_support": 0,
               "rows_sha256": canonical_json_sha256(rows), "rows": rows}
    with (output_dir / "audit.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {k: v for k, v in summary.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir), indent=2))
