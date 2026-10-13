import re
import time
from typing import Any

import akshare as ak
import pandas as pd


_STOCK_CACHE = None
_CACHE_TIME = 0
_CACHE_SECONDS = 3600


def _normalize_text(text: str) -> str:
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def get_exchange(stock_code: str) -> str:
    code = str(stock_code).zfill(6)

    if code.startswith("6"):
        return "SSE"

    if code.startswith(("0", "2", "3")):
        return "SZSE"

    if code.startswith(("4", "8", "9")):
        return "BSE"

    return "UNKNOWN"


def get_market_symbol(stock_code: str) -> str:
    code = str(stock_code).zfill(6)

    exchange = get_exchange(code)

    if exchange == "SSE":
        return f"SH{code}"

    if exchange == "SZSE":
        return f"SZ{code}"

    if exchange == "BSE":
        return f"BJ{code}"

    return code


def _normalize_stock_list(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("Stock list is empty.")

    df = df.copy()

    if "code" in df.columns and "name" in df.columns:
        df = df.rename(
            columns={
                "code": "代码",
                "name": "名称",
            }
        )

    elif "代码" in df.columns and "名称" in df.columns:
        pass

    elif len(df.columns) >= 2:
        df = df.rename(
            columns={
                df.columns[0]: "代码",
                df.columns[1]: "名称",
            }
        )

    else:
        raise RuntimeError(
            f"Unexpected stock list columns: {list(df.columns)}"
        )

    df["代码"] = (
        df["代码"]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
    )

    df["名称"] = df["名称"].astype(str).str.strip()

    df = df.dropna(
        subset=["代码", "名称"]
    )

    df = df.drop_duplicates(
        subset=["代码"]
    )

    df["_normalized_name"] = (
        df["名称"].map(_normalize_text)
    )

    return df.reset_index(drop=True)


def get_stock_universe(
    force_refresh: bool = False,
) -> pd.DataFrame:
    global _STOCK_CACHE
    global _CACHE_TIME

    now = time.time()

    cache_valid = (
        _STOCK_CACHE is not None
        and not force_refresh
        and now - _CACHE_TIME < _CACHE_SECONDS
    )

    if cache_valid:
        return _STOCK_CACHE.copy()

    try:
        df = ak.stock_info_a_code_name()

        df = _normalize_stock_list(df)

    except Exception as exc:
        raise RuntimeError(
            "Failed to load the A-share stock list from AKShare. "
            f"Original error: {exc}"
        ) from exc

    _STOCK_CACHE = df
    _CACHE_TIME = now

    return df.copy()


def resolve_stock(
    query: str,
) -> dict[str, Any]:
    if not query or not str(query).strip():
        raise ValueError(
            "Stock query cannot be empty."
        )

    raw_query = str(query).strip()

    normalized_query = _normalize_text(
        raw_query
    )

    df = get_stock_universe()

    code_match = re.search(
        r"(\d{6})",
        normalized_query,
    )

    if code_match:
        stock_code = code_match.group(1)

        matched = df[
            df["代码"] == stock_code
        ]

        if matched.empty:
            raise ValueError(
                f"Could not find A-share stock code "
                f"{stock_code}."
            )

        row = matched.iloc[0]

    else:
        exact_match = df[
            df["_normalized_name"]
            == normalized_query
        ]

        if not exact_match.empty:
            row = exact_match.iloc[0]

        else:
            contains_match = df[
                df["_normalized_name"]
                .str.contains(
                    re.escape(
                        normalized_query
                    ),
                    na=False,
                )
            ]

            if contains_match.empty:
                raise ValueError(
                    f"Could not find an A-share "
                    f"company matching "
                    f"'{raw_query}'."
                )

            if len(contains_match) > 1:
                candidates = []

                for _, candidate in (
                    contains_match
                    .head(10)
                    .iterrows()
                ):
                    candidates.append(
                        f"{candidate['名称']} "
                        f"({candidate['代码']})"
                    )

                candidate_text = ", ".join(
                    candidates
                )

                raise ValueError(
                    f"'{raw_query}' matches "
                    f"multiple stocks: "
                    f"{candidate_text}. "
                    f"Please enter the full "
                    f"company name or stock code."
                )

            row = contains_match.iloc[0]

    stock_code = str(
        row["代码"]
    ).zfill(6)

    company_name = str(
        row["名称"]
    )

    exchange = get_exchange(
        stock_code
    )

    market_symbol = get_market_symbol(
        stock_code
    )

    return {
        "stock_code": stock_code,
        "company_name": company_name,
        "exchange": exchange,
        "market_symbol": market_symbol,
    }