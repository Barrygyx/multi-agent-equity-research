import time
from typing import Any

import akshare as ak
import pandas as pd


_CACHE = {}
_CACHE_TTL = 300


def _cache_get(key):
    item = _CACHE.get(
        key
    )

    if item is None:
        return None

    saved_time, value = item

    if (
        time.time()
        - saved_time
        > _CACHE_TTL
    ):
        _CACHE.pop(
            key,
            None,
        )

        return None

    return value


def _cache_set(
    key,
    value,
):
    _CACHE[key] = (
        time.time(),
        value,
    )


def _to_number(value: Any):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def get_valuation_data(
    stock_code: str,
):
    stock_code = str(
        stock_code
    ).zfill(6)

    cache_key = (
        "valuation",
        stock_code,
    )

    cached = _cache_get(
        cache_key
    )

    if cached is not None:
        return cached

    result = {
        "pe": None,
        "pe_date": None,
        "source": None,
        "warnings": [],
    }

    try:
        df = (
            ak.stock_zh_valuation_baidu(
                symbol=stock_code,
                indicator="市盈率",
            )
        )

        if (
            df is not None
            and not df.empty
        ):
            df = df.copy()

            date_column = None
            value_column = None

            for candidate in [
                "date",
                "日期",
                "trade_date",
            ]:
                if candidate in df.columns:
                    date_column = candidate
                    break

            for candidate in [
                "value",
                "市盈率",
                "pe",
                "PE",
            ]:
                if candidate in df.columns:
                    value_column = candidate
                    break

            if value_column is None:
                for column in df.columns:
                    if column == date_column:
                        continue

                    numeric = pd.to_numeric(
                        df[column],
                        errors="coerce",
                    )

                    if numeric.notna().any():
                        value_column = column
                        break

            if date_column is not None:
                df[date_column] = (
                    pd.to_datetime(
                        df[date_column],
                        errors="coerce",
                    )
                )

                df = df.sort_values(
                    date_column
                )

            if value_column is not None:
                values = pd.to_numeric(
                    df[value_column],
                    errors="coerce",
                )

                valid = df[
                    values.notna()
                ].copy()

                if not valid.empty:
                    latest = (
                        valid.iloc[-1]
                    )

                    result["pe"] = (
                        _to_number(
                            latest[
                                value_column
                            ]
                        )
                    )

                    if (
                        date_column
                        is not None
                        and pd.notna(
                            latest[
                                date_column
                            ]
                        )
                    ):
                        result[
                            "pe_date"
                        ] = str(
                            latest[
                                date_column
                            ].date()
                        )

                    result[
                        "source"
                    ] = "百度股市通"

    except Exception as exc:
        result[
            "warnings"
        ].append(
            "PE unavailable: "
            f"{exc}"
        )

    _cache_set(
        cache_key,
        result,
    )

    return result