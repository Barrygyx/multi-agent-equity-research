import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

import akshare as ak
import pandas as pd

from tools.price_tools import get_price_data
from tools.stock_search import get_market_symbol


_CACHE = {}
_CACHE_TTL = 300


def _cache_get(key):
    item = _CACHE.get(key)

    if item is None:
        return None

    saved_time, value = item

    if time.time() - saved_time > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None

    return value


def _cache_set(key, value):
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

    if isinstance(value, (int, float)):
        return float(value)

    text = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("%", "")
    )

    if not text or text in {
        "-",
        "--",
        "nan",
    }:
        return None

    try:
        return float(text)

    except ValueError:
        return None


def _normalize_name(
    name: str,
):
    text = str(name).strip()

    for word in [
        "制造业",
        "制造",
        "行业",
        "产业",
        "板块",
        "工业",
    ]:
        text = text.replace(
            word,
            "",
        )

    return (
        text.replace("、", "")
        .replace("/", "")
        .replace("-", "")
        .replace(" ", "")
    )


def _similarity(
    first: str,
    second: str,
):
    first = _normalize_name(
        first
    )

    second = _normalize_name(
        second
    )

    if not first or not second:
        return 0.0

    if (
        first in second
        or second in first
    ):
        return 1.0

    return SequenceMatcher(
        None,
        first,
        second,
    ).ratio()


def _calculate_return(
    prices: pd.Series,
    periods: int,
):
    if len(prices) <= periods:
        return None

    old_price = float(
        prices.iloc[
            -periods - 1
        ]
    )

    current_price = float(
        prices.iloc[-1]
    )

    if old_price == 0:
        return None

    return round(
        (
            current_price
            / old_price
            - 1
        )
        * 100,
        2,
    )


def _find_industry(
    profile_industry: str,
    available_names: list[str],
):
    best_name = None
    best_score = 0.0

    for name in available_names:
        score = _similarity(
            profile_industry,
            name,
        )

        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score


def _get_industry_snapshot(
    profile_industry: str,
):
    cache_key = (
        "industry_snapshot",
        profile_industry,
    )

    cached = _cache_get(
        cache_key
    )

    if cached is not None:
        return cached

    df = ak.stock_fund_flow_industry(
        symbol="即时"
    )

    if df is None or df.empty:
        return {}

    if "行业" not in df.columns:
        return {}

    names = (
        df["行业"]
        .dropna()
        .astype(str)
        .tolist()
    )

    industry_name, score = (
        _find_industry(
            profile_industry,
            names,
        )
    )

    if industry_name is None:
        return {}

    matched = df[
        df["行业"].astype(str)
        == industry_name
    ]

    if matched.empty:
        return {}

    row = matched.iloc[0]

    result = {
        "industry_name": industry_name,
        "match_score": round(
            score,
            3,
        ),
        "industry_index": _to_number(
            row.get("行业指数")
        ),
        "change_pct": _to_number(
            row.get("行业-涨跌幅")
        ),
        "inflow_billion": _to_number(
            row.get("流入资金")
        ),
        "outflow_billion": _to_number(
            row.get("流出资金")
        ),
        "net_flow_billion": _to_number(
            row.get("净额")
        ),
        "company_count": _to_number(
            row.get("公司家数")
        ),
        "leading_stock": row.get(
            "领涨股"
        ),
        "leading_stock_change_pct": (
            _to_number(
                row.get(
                    "领涨股-涨跌幅"
                )
            )
        ),
    }

    ranking = df.copy()

    if "净额" in ranking.columns:
        ranking["_net_flow"] = (
            pd.to_numeric(
                ranking["净额"],
                errors="coerce",
            )
        )

        ranking = (
            ranking.dropna(
                subset=["_net_flow"]
            )
            .sort_values(
                "_net_flow",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        rank_match = ranking[
            ranking["行业"].astype(str)
            == industry_name
        ]

        if not rank_match.empty:
            result["fund_flow_rank"] = (
                int(
                    rank_match.index[0]
                )
                + 1
            )

            result["industry_count"] = (
                len(ranking)
            )

    _cache_set(
        cache_key,
        result,
    )

    return result


def _get_industry_history(
    industry_name: str,
):
    cache_key = (
        "industry_history",
        industry_name,
    )

    cached = _cache_get(
        cache_key
    )

    if cached is not None:
        return cached

    end_date = datetime.now()

    start_date = (
        end_date
        - timedelta(days=150)
    )

    df = (
        ak.stock_board_industry_index_ths(
            symbol=industry_name,
            start_date=start_date.strftime(
                "%Y%m%d"
            ),
            end_date=end_date.strftime(
                "%Y%m%d"
            ),
        )
    )

    if df is None or df.empty:
        return {}

    date_column = None
    close_column = None

    for column in [
        "日期",
        "date",
    ]:
        if column in df.columns:
            date_column = column
            break

    for column in [
        "收盘价",
        "收盘",
        "close",
    ]:
        if column in df.columns:
            close_column = column
            break

    if (
        date_column is None
        or close_column is None
    ):
        return {}

    df = df.copy()

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[date_column]
        )
        .sort_values(
            date_column
        )
    )

    close = pd.to_numeric(
        df[close_column],
        errors="coerce",
    ).dropna()

    if close.empty:
        return {}

    result = {
        "industry_name": (
            industry_name
        ),
        "latest_date": str(
            df.iloc[-1][
                date_column
            ].date()
        ),
        "return_5d_pct": (
            _calculate_return(
                close,
                5,
            )
        ),
        "return_20d_pct": (
            _calculate_return(
                close,
                20,
            )
        ),
        "return_60d_pct": (
            _calculate_return(
                close,
                60,
            )
        ),
    }

    _cache_set(
        cache_key,
        result,
    )

    return result


def _convert_ratio_to_pct(
    value,
):
    number = _to_number(
        value
    )

    if number is None:
        return None

    if abs(number) <= 1.5:
        return round(
            number * 100,
            2,
        )

    return round(
        number,
        2,
    )


def _get_business_composition(
    stock_code: str,
):
    cache_key = (
        "business_composition",
        stock_code,
    )

    cached = _cache_get(
        cache_key
    )

    if cached is not None:
        return cached

    market_symbol = (
        get_market_symbol(
            stock_code
        )
    )

    df = ak.stock_zygc_em(
        symbol=market_symbol
    )

    if df is None or df.empty:
        return {}

    if "报告日期" not in df.columns:
        return {}

    df = df.copy()

    df["报告日期"] = pd.to_datetime(
        df["报告日期"],
        errors="coerce",
    )

    latest_date = (
        df["报告日期"].max()
    )

    latest = df[
        df["报告日期"]
        == latest_date
    ]

    result = {
        "report_date": str(
            latest_date.date()
        ),
        "product": [],
        "industry": [],
    }

    category_map = {
        "按产品分类": "product",
        "按行业分类": "industry",
    }

    for source, target in (
        category_map.items()
    ):
        subset = latest[
            latest["分类类型"]
            == source
        ].copy()

        if subset.empty:
            continue

        if "主营收入" in subset.columns:
            subset["_income"] = (
                pd.to_numeric(
                    subset["主营收入"],
                    errors="coerce",
                )
            )

            subset = (
                subset.sort_values(
                    "_income",
                    ascending=False,
                )
            )

        for _, row in (
            subset.head(5).iterrows()
        ):
            result[target].append(
                {
                    "name": str(
                        row.get(
                            "主营构成",
                            "",
                        )
                    ),
                    "revenue": _to_number(
                        row.get(
                            "主营收入"
                        )
                    ),
                    "revenue_share_pct": (
                        _convert_ratio_to_pct(
                            row.get(
                                "收入比例"
                            )
                        )
                    ),
                    "gross_margin_pct": (
                        _convert_ratio_to_pct(
                            row.get(
                                "毛利率"
                            )
                        )
                    ),
                }
            )

    _cache_set(
        cache_key,
        result,
    )

    return result


def _get_sector_signal(
    snapshot,
):
    change_pct = snapshot.get(
        "change_pct"
    )

    net_flow = snapshot.get(
        "net_flow_billion"
    )

    if (
        change_pct is None
        or net_flow is None
    ):
        return "数据不足"

    if (
        change_pct > 0
        and net_flow > 0
    ):
        return "板块偏强"

    if (
        change_pct < 0
        and net_flow < 0
    ):
        return "板块偏弱"

    return "板块表现分化"


def get_sector_data(
    stock_code: str,
    profile_industry: str | None = None,
    stock_price_summary: dict[str, Any] | None = None,
):
    stock_code = str(
        stock_code
    ).zfill(6)

    result = {
        "stock_code": stock_code,
        "industry_snapshot": {},
        "industry_performance": {},
        "relative_performance": {},
        "business_composition": {},
        "sector_signal": "数据不足",
        "warnings": [],
    }

    if not profile_industry:
        result["warnings"].append(
            "profile_industry was not provided."
        )

        return result

    try:
        snapshot = (
            _get_industry_snapshot(
                profile_industry
            )
        )

        result[
            "industry_snapshot"
        ] = snapshot

    except Exception as exc:
        result["warnings"].append(
            f"Industry snapshot unavailable: {exc}"
        )

        snapshot = {}

    industry_name = snapshot.get(
        "industry_name"
    )

    if industry_name:
        try:
            result[
                "industry_performance"
            ] = (
                _get_industry_history(
                    industry_name
                )
            )

        except Exception as exc:
            result["warnings"].append(
                "Industry history unavailable: "
                f"{exc}"
            )

    try:
        result[
            "business_composition"
        ] = (
            _get_business_composition(
                stock_code
            )
        )

    except Exception as exc:
        result["warnings"].append(
            "Business composition unavailable: "
            f"{exc}"
        )

    if stock_price_summary is None:
        try:
            stock_price_summary = (
                get_price_data(
                    stock_code
                )["summary"]
            )

        except Exception:
            stock_price_summary = {}

    industry_performance = result[
        "industry_performance"
    ]

    for period in [
        5,
        20,
        60,
    ]:
        stock_return = (
            stock_price_summary.get(
                f"return_{period}d_pct"
            )
        )

        sector_return = (
            industry_performance.get(
                f"return_{period}d_pct"
            )
        )

        if (
            stock_return is not None
            and sector_return is not None
        ):
            result[
                "relative_performance"
            ][
                f"{period}d"
            ] = {
                "stock_return_pct": (
                    stock_return
                ),
                "sector_return_pct": (
                    sector_return
                ),
                "excess_return_pct": round(
                    stock_return
                    - sector_return,
                    2,
                ),
            }

    result["sector_signal"] = (
        _get_sector_signal(
            snapshot
        )
    )

    return result