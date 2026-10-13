from typing import Any

import akshare as ak
import pandas as pd


def _to_python(value: Any):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def get_company_data(
    stock_code: str,
) -> dict[str, Any]:
    stock_code = str(stock_code).zfill(6)

    result = {
        "stock_code": stock_code,
    }

    try:
        df = ak.stock_profile_cninfo(
            symbol=stock_code
        )

        if df is None or df.empty:
            result["warning"] = (
                "CNINFO returned no company profile."
            )
            return result

        row = df.iloc[0]

        for column in df.columns:
            value = _to_python(
                row[column]
            )

            if value is not None:
                result[column] = value

    except Exception as exc:
        result["warning"] = (
            f"CNINFO company profile unavailable: {exc}"
        )

    return result