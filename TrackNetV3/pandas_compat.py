from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    class Series(list):
        @property
        def values(self):
            return np.array(self)


    class DataFrame:
        def __init__(self, rows):
            if isinstance(rows, dict):
                keys = list(rows.keys())
                values = [list(column) for column in rows.values()]
                self._rows = [dict(zip(keys, row)) for row in zip(*values, strict=False)]
            else:
                self._rows = [dict(row) for row in rows]
            self.columns = list(self._rows[0].keys()) if self._rows else []

        def __len__(self):
            return len(self._rows)

        def sort_values(self, by: str):
            return DataFrame(sorted(self._rows, key=lambda row: _coerce(row.get(by))))

        def fillna(self, value):
            return DataFrame([{key: (value if cell in (None, "") else cell) for key, cell in row.items()} for row in self._rows])

        def to_csv(self, path, index=False):
            if index:
                raise NotImplementedError("fallback DataFrame.to_csv does not support index=True")
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(self._rows)

        def __getitem__(self, key: str):
            return Series([_coerce(row.get(key)) for row in self._rows])


    class _FallbackPandas:
        DataFrame = DataFrame

        @staticmethod
        def read_csv(path, encoding="utf8"):
            with Path(path).open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                return DataFrame([dict(row) for row in reader])


    pd = _FallbackPandas()


def _coerce(value):
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    return value
