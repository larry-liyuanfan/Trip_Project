"""JSONL and tabular I/O helpers with bounded Parquet/CSV fallback writes."""

import json
import csv
import importlib.util
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> tuple[Iterable[dict[str, Any]], dict[str, int]]:
    """Return a lazy JSONL iterator and counters updated as rows are consumed."""
    summary = {"path": str(path), "total_lines": 0, "records": 0, "malformed_lines": 0}

    def iterator() -> Iterable[dict[str, Any]]:
        """Yield mapping rows while counting malformed and non-object records."""
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
    """Apply an optional smoke-test cap without materializing the iterable."""
    if max_records is None:
        yield from records
        return
    for _, record in zip(range(max_records), records):
        yield record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 indented JSON artifact and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_table(path: Path, rows: Iterable[dict[str, Any]], output_format: str = "parquet") -> dict[str, Any]:
    """Write a bounded table and record any explicit CSV fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    requested = output_format.lower()
    actual = requested
    error = None
    if requested == "parquet":
        if importlib.util.find_spec("pyarrow") is not None:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pylist(records)
                pq.write_table(table, path)
            except Exception as exc:
                actual = "csv_fallback"
                error = str(exc)
                _write_csv(path, records)
        elif importlib.util.find_spec("fastparquet") is None:
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


class TableStreamWriter:
    """Buffer rows in bounded chunks while preserving one output schema."""
    def __init__(
        self,
        path: Path,
        output_format: str = "parquet",
        fieldnames: list[str] | None = None,
        chunk_size: int = 50000,
        parquet_schema: Any | None = None,
    ) -> None:
        """Configure output format, chunk size, field order, and optional schema."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format.lower()
        self.fieldnames = fieldnames
        self.chunk_size = chunk_size
        self.rows_written = 0
        self.buffer: list[dict[str, Any]] = []
        self.actual_format = self.output_format
        self.error: str | None = None
        self._csv_handle = None
        self._csv_writer = None
        self._parquet_writer = None
        self._parquet_schema = parquet_schema

        if self.output_format == "parquet" and (
            importlib.util.find_spec("pyarrow") is None
            or importlib.util.find_spec("pyarrow.parquet") is None
        ):
            self.actual_format = "csv_fallback"
            self.error = "No pyarrow Parquet engine installed; wrote CSV fallback"
        elif self.output_format not in {"parquet", "csv"}:
            raise ValueError(f"Unsupported output format: {output_format}")

    def write(self, row: dict[str, Any]) -> None:
        """Queue one row and flush when the configured chunk bound is reached."""
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        """Persist the current buffer using the selected storage backend."""
        if not self.buffer:
            return
        if self.actual_format == "parquet":
            self._flush_parquet()
        else:
            self._flush_csv()
        self.rows_written += len(self.buffer)
        self.buffer = []

    def close(self) -> dict[str, Any]:
        """Flush remaining rows, materialize empty schemas, and close handles."""
        self.flush()
        if self.rows_written == 0:
            self._write_empty_table()
        if self._csv_handle is not None:
            self._csv_handle.close()
        if self._parquet_writer is not None:
            self._parquet_writer.close()
        return {
            "path": str(self.path),
            "rows": self.rows_written,
            "requested_format": self.output_format,
            "actual_format": self.actual_format,
            "error": self.error,
        }

    def _write_empty_table(self) -> None:
        """Materialize an empty table so downstream schema validation can run."""
        if self.actual_format == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = self._parquet_schema or pa.schema([(fieldname, pa.string()) for fieldname in self.fieldnames or []])
            pq.write_table(pa.Table.from_pylist([], schema=schema), self.path)
        elif self.fieldnames is not None:
            self._csv_handle = self.path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.fieldnames)
            self._csv_writer.writeheader()

    def _flush_csv(self) -> None:
        """Append serialized rows to a single-header CSV stream."""
        if self.fieldnames is None:
            self.fieldnames = sorted({key for record in self.buffer for key in record.keys()})
        if self._csv_handle is None:
            self._csv_handle = self.path.open("w", encoding="utf-8", newline="")
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self.fieldnames)
            self._csv_writer.writeheader()
        assert self._csv_writer is not None
        for record in self.buffer:
            self._csv_writer.writerow({key: _serialize_cell(record.get(key)) for key in self.fieldnames})

    def _flush_parquet(self) -> None:
        """Append an Arrow table while enforcing the first or explicit schema."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.fieldnames is None:
            self.fieldnames = sorted({key for record in self.buffer for key in record.keys()})
        rows = [{key: record.get(key) for key in self.fieldnames} for record in self.buffer]
        table = pa.Table.from_pylist(rows, schema=self._parquet_schema)
        if self._parquet_schema is None:
            self._parquet_schema = table.schema
        if self._parquet_writer is None:
            self._parquet_writer = pq.ParquetWriter(self.path, table.schema)
        self._parquet_writer.write_table(table)


def read_table(path: Path) -> list[dict[str, Any]]:
    """Read Parquet when possible, otherwise decode the documented CSV fallback."""
    if not path.exists():
        return []
    if importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None:
        if path.suffix == ".parquet" and importlib.util.find_spec("pyarrow") is not None:
            try:
                import pyarrow.parquet as pq

                records = pq.read_table(path).to_pylist()
                return [{key: _normalize_cell(value) for key, value in row.items()} for row in records]
            except Exception:
                pass
        try:
            import pandas as pd

            dataframe = pd.read_parquet(path)
            records = dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")
            return [{key: _normalize_cell(value) for key, value in row.items()} for row in records]
        except Exception:
            pass
    return _read_csv(path)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write scalar and JSON-serialized structured cells to CSV."""
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
    """Read CSV rows and restore supported scalar and structured cell types."""
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: _deserialize_cell(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _serialize_cell(value: Any) -> Any:
    """Serialize nested cells as JSON and represent nulls as empty strings."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _deserialize_cell(value: str) -> Any:
    """Restore JSON, booleans, and numeric values from a CSV cell."""
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
    """Convert Arrow and NumPy containers to plain Python values recursively."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _normalize_cell(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_cell(child) for child in value]
    return value
