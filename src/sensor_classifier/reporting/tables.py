"""Small Markdown table formatter used for static machine-generated reports."""
from __future__ import annotations

import pandas as pd


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, n: int = 20) -> str:
    table = frame[columns] if columns else frame
    table = table.head(n)
    if table.empty:
        return "No observations."
    headers = list(table.columns)
    def clean(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(
        "| " + " | ".join(clean(v) for v in row) + " |" for row in table.itertuples(index=False, name=None))
