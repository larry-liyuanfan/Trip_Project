import json
import csv
import importlib.util
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> tuple[Iterable[dict[str, Any]], dict[str, int]]:
    summary = {"path": str(path), "total_lines": 0, "records": 0, "malformed_lines": 0}

    def iterator() -> Iterable[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                summary["total_lines"] += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    summary["malformed_lines"] += 1
                    continue
                if isinstance(record, dict):
                    summary["records"] += 1
                    yield record
                else:
                    summary["malformed_lines"] += 1

    return iterator(), summary


def limit_records(records: Iterable[dict[str, Any]], max_records: int | None) -> Iterable[dict[str, Any]]:
    if max_records is None:
        yield from records
        return
    for _, record in zip(range(max_records), records):
        yield record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_table(path: Path, rows: Iterable[dict[str, Any]], output_format: str = "parquet") -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    requested = output_format.lower()
    actual = requested
    error = None
    if requested == "parquet":
        if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
            actual = "csv_fallback"
            error = "No pandas Parquet engine installed; wrote CSV fallback"
            _write_csv(path, records)
        else:
            try:
                import pandas as pd

                dataframe = pd.DataFrame(records)
                dataframe.to_parquet(path, index=False)
            except Exception as exc:  # pragma: no cover - depends on optional local engines
                actual = "csv_fallback"
                error = str(exc)
                _write_csv(path, records)
    elif requested == "csv":
        _write_csv(path, records)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return {"path": str(path), "rows": len(records), "requested_format": requested, "actual_format": actual, "error": error}


def read_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None:
        try:
            import pandas as pd

            dataframe = pd.read_parquet(path)
            records = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
            return [{key: _normalize_cell(value) for key, value in row.items()} for row in records]
        except Exception:
            pass
    return _read_csv(path)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _serialize_cell(record.get(key)) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _deserialize_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _deserialize_cell(value: str) -> Any:
    if value == "":
        return None
    if value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if value in {"True", "False"}:
        return value == "True"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _normalize_cell(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _normalize_cell(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_cell(child) for child in value]
    return value
    return value
