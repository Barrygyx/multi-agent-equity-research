from typing import Any

import akshare as ak
import pandas as pd

from tools.price_tools import get_price_data


def _to_python(value: Any):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


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

    text = str(value).strip()

    if not text or text in {
        "-",
        "--",
        "None",
        "nan",
    }:
        return None

    multiplier = 1.0

    if "亿" in text:
        multiplier = 100000000.0

    elif "万" in text:
        multiplier = 10000.0

    text = (
        text.replace(",", "")
        .replace("%", "")
        .replace("亿元", "")
        .replace("万元", "")
        .replace("亿", "")
        .replace("万", "")
        .replace("元", "")
        .strip()
    )

    try:
        return float(text) * multiplier

    except ValueError:
        return None


def _get_market(stock_code: str):
    code = str(stock_code).zfill(6)

    if code.startswith("6"):
        return "sh"

    if code.startswith(("0", "2", "3")):
        return "sz"

    if code.startswith(("4", "8", "9")):
        return "bj"

    return "sz"


def _get_turnover_summary(
    stock_code: str,
    price_history: list[dict[str, Any]] | None = None,
):
    try:
        if price_history is None:
            price_data = get_price_data(
                stock_code
            )

            price_history = price_data.get(
                "history",
                [],
            )

        if not price_history:
            return {}

        turnover_values = []

        for row in price_history:
            value = _to_number(
                row.get("turnover")
            )

            if value is not None:
                turnover_values.append(
                    value * 100
                )

        if not turnover_values:
            return {}

        current = turnover_values[-1]

        last_5 = turnover_values[-5:]
        last_20 = turnover_values[-20:]

        avg_5 = (
            sum(last_5)
            / len(last_5)
        )

        avg_20 = (
            sum(last_20)
            / len(last_20)
        )

        activity_ratio = None

        if avg_20 != 0:
            activity_ratio = round(
                current / avg_20,
                2,
            )

        return {
            "current_turnover_rate_pct": round(
                current,
                2,
            ),
            "avg_turnover_5d_pct": round(
                avg_5,
                2,
            ),
            "avg_turnover_20d_pct": round(
                avg_20,
                2,
            ),
            "turnover_activity_ratio": (
                activity_ratio
            ),
        }

    except Exception:
        return {}


def _window_flow(
    df: pd.DataFrame,
    days: int,
):
    if df is None or df.empty:
        return {}

    window = df.tail(days)

    flow_column = "主力净流入-净额"
    ratio_column = "主力净流入-净占比"

    if flow_column not in window.columns:
        return {}

    flows = pd.to_numeric(
        window[flow_column],
        errors="coerce",
    ).dropna()

    if flows.empty:
        return {}

    result = {
        "days": days,
        "main_net_inflow": round(
            float(flows.sum()),
            2,
        ),
    }

    if ratio_column in window.columns:
        ratios = pd.to_numeric(
            window[ratio_column],
            errors="coerce",
        ).dropna()

        if not ratios.empty:
            result[
                "avg_main_net_inflow_ratio_pct"
            ] = round(
                float(ratios.mean()),
                2,
            )

    return result


def _get_eastmoney_flow(
    stock_code: str,
):
    df = ak.stock_individual_fund_flow(
        stock=stock_code,
        market=_get_market(
            stock_code
        ),
    )

    if df is None or df.empty:
        raise RuntimeError(
            "Eastmoney returned no fund-flow data."
        )

    df = df.copy()

    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(
            df["日期"],
            errors="coerce",
        )

        df = (
            df.sort_values("日期")
            .reset_index(drop=True)
        )

    numeric_columns = [
        "收盘价",
        "涨跌幅",
        "主力净流入-净额",
        "主力净流入-净占比",
        "超大单净流入-净额",
        "大单净流入-净额",
        "中单净流入-净额",
        "小单净流入-净额",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    latest = df.iloc[-1]

    return {
        "source": "东方财富",
        "flow_type": "main_force",
        "latest": {
            "date": (
                str(
                    latest["日期"].date()
                )
                if "日期" in df.columns
                and pd.notna(
                    latest["日期"]
                )
                else None
            ),
            "close": _to_python(
                latest.get("收盘价")
            ),
            "change_pct": _to_python(
                latest.get("涨跌幅")
            ),
            "main_net_inflow": (
                _to_python(
                    latest.get(
                        "主力净流入-净额"
                    )
                )
            ),
            "main_net_inflow_ratio_pct": (
                _to_python(
                    latest.get(
                        "主力净流入-净占比"
                    )
                )
            ),
            "super_large_net_inflow": (
                _to_python(
                    latest.get(
                        "超大单净流入-净额"
                    )
                )
            ),
            "large_net_inflow": (
                _to_python(
                    latest.get(
                        "大单净流入-净额"
                    )
                )
            ),
            "medium_net_inflow": (
                _to_python(
                    latest.get(
                        "中单净流入-净额"
                    )
                )
            ),
            "small_net_inflow": (
                _to_python(
                    latest.get(
                        "小单净流入-净额"
                    )
                )
            ),
        },
        "flow_5d": _window_flow(
            df,
            5,
        ),
        "flow_10d": _window_flow(
            df,
            10,
        ),
    }


def _find_stock_row(
    df: pd.DataFrame,
    stock_code: str,
):
    if df is None or df.empty:
        return None

    code_column = None

    for candidate in [
        "股票代码",
        "代码",
    ]:
        if candidate in df.columns:
            code_column = candidate
            break

    if code_column is None:
        return None

    df = df.copy()

    df["_code"] = (
        df[code_column]
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False,
        )
        .str.zfill(6)
    )

    matched = df[
        df["_code"]
        == stock_code
    ]

    if matched.empty:
        return None

    return matched.iloc[0]


def _get_ths_flow(
    stock_code: str,
):
    df = ak.stock_fund_flow_individual(
        symbol="即时"
    )

    row = _find_stock_row(
        df,
        stock_code,
    )

    if row is None:
        raise RuntimeError(
            "THS returned no matching stock."
        )

    return {
        "source": "同花顺",
        "flow_type": "overall",
        "latest": {
            "latest_price": _to_number(
                row.get("最新价")
            ),
            "change_pct": _to_number(
                row.get("涨跌幅")
            ),
            "turnover_rate_pct": (
                _to_number(
                    row.get("换手率")
                )
            ),
            "inflow": _to_number(
                row.get("流入资金")
            ),
            "outflow": _to_number(
                row.get("流出资金")
            ),
            "net_flow": _to_number(
                row.get("净额")
            ),
            "turnover_amount": (
                _to_number(
                    row.get("成交额")
                )
            ),
        },
    }


def _get_signal(
    fund_flow: dict[str, Any],
    turnover: dict[str, Any],
):
    if not fund_flow:
        return "资金数据不足"

    source = fund_flow.get(
        "source"
    )

    latest = fund_flow.get(
        "latest",
        {},
    )

    activity_ratio = turnover.get(
        "turnover_activity_ratio"
    )

    if source == "东方财富":
        latest_flow = latest.get(
            "main_net_inflow"
        )

        latest_ratio = latest.get(
            "main_net_inflow_ratio_pct"
        )

        five_day_flow = (
            fund_flow
            .get(
                "flow_5d",
                {},
            )
            .get(
                "main_net_inflow"
            )
        )

        ten_day_flow = (
            fund_flow
            .get(
                "flow_10d",
                {},
            )
            .get(
                "main_net_inflow"
            )
        )

        if latest_flow is None:
            return "主力资金数据不足"

        if (
            latest_ratio is not None
            and latest_ratio >= 15
        ):
            return "当日主力明显流入"

        if (
            latest_ratio is not None
            and latest_ratio <= -15
        ):
            return "当日主力明显流出"

        if (
            five_day_flow is not None
            and ten_day_flow is not None
        ):
            if (
                five_day_flow > 0
                and ten_day_flow > 0
            ):
                return "近期主力资金偏流入"

            if (
                five_day_flow < 0
                and ten_day_flow < 0
            ):
                return "近期主力资金偏流出"

        return "主力资金短期分化"

    if source == "同花顺":
        net_flow = latest.get(
            "net_flow"
        )

        if net_flow is None:
            return "整体资金数据不足"

        active = (
            activity_ratio is not None
            and activity_ratio >= 2
        )

        if net_flow > 0:
            if active:
                return (
                    "当日资金净流入，"
                    "成交明显放大"
                )

            return "当日整体资金净流入"

        if net_flow < 0:
            if active:
                return (
                    "当日资金净流出，"
                    "成交明显放大"
                )

            return "当日整体资金净流出"

        return "资金方向中性"

    return "资金数据不足"


def get_fund_flow_data(
    stock_code: str,
    price_history: list[dict[str, Any]] | None = None,
):
    stock_code = str(
        stock_code
    ).zfill(6)

    fund_flow = {}

    primary_available = False
    fallback_used = False

    source_note = None
    warnings = []

    eastmoney_error = None

    # First get fund-flow data.
    # Do this before price data because the THS interface
    # can occasionally return an abnormal table structure
    # after other network requests.
    try:
        fund_flow = (
            _get_eastmoney_flow(
                stock_code
            )
        )

        primary_available = True

        source_note = (
            "资金流数据来自东方财富，"
            "包含主力及不同大单类型资金流。"
        )

    except Exception as exc:
        eastmoney_error = str(exc)

    # Fall back to THS if Eastmoney is unavailable.
    if not fund_flow:
        try:
            fund_flow = (
                _get_ths_flow(
                    stock_code
                )
            )

            fallback_used = True

            source_note = (
                "东方财富资金流接口暂时不可用，"
                "已自动切换至同花顺。"
                "同花顺数据表示整体资金流，"
                "不等同于主力净流入。"
            )

        except Exception as exc:
            warnings.append(
                "Both fund-flow sources failed. "
                f"Eastmoney: {eastmoney_error}; "
                f"THS: {exc}"
            )

    # Calculate turnover after fund-flow retrieval.
    turnover = (
        _get_turnover_summary(
            stock_code=stock_code,
            price_history=price_history,
        )
    )

    signal = _get_signal(
        fund_flow=fund_flow,
        turnover=turnover,
    )

    return {
        "stock_code": stock_code,
        "fund_flow": fund_flow,
        "turnover": turnover,
        "signal": signal,
        "source_status": {
            "primary_source": "东方财富",
            "primary_available": (
                primary_available
            ),
            "fallback_source": "同花顺",
            "fallback_used": (
                fallback_used
            ),
        },
        "source_note": source_note,
        "warnings": warnings,
    }