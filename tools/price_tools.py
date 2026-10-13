from datetime import datetime, timedelta
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


def _calculate_return(
    prices: pd.Series,
    periods: int,
):
    if len(prices) <= periods:
        return None

    current_price = float(prices.iloc[-1])
    old_price = float(prices.iloc[-periods - 1])

    if old_price == 0:
        return None

    return round(
        (current_price / old_price - 1) * 100,
        2,
    )


def _get_tx_symbol(
    stock_code: str,
) -> str:
    code = str(stock_code).zfill(6)

    if code.startswith("6"):
        return f"sh{code}"

    if code.startswith(("0", "2", "3")):
        return f"sz{code}"

    if code.startswith(("4", "8", "9")):
        return f"bj{code}"

    return code


def get_price_data(
    stock_code: str,
    history_days: int = 450,
) -> dict[str, Any]:
    stock_code = str(stock_code).zfill(6)

    end_date = datetime.now()
    start_date = end_date - timedelta(
        days=history_days
    )

    tx_symbol = _get_tx_symbol(
        stock_code
    )

    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=tx_symbol,
            start_date=start_date.strftime(
                "%Y%m%d"
            ),
            end_date=end_date.strftime(
                "%Y%m%d"
            ),
            adjust="qfq",
        )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve historical "
            f"price data from Tencent: {exc}"
        ) from exc

    if df is None or df.empty:
        raise RuntimeError(
            f"No price data returned for "
            f"{stock_code}."
        )

    df = df.copy()

    rename_map = {
        "date": "日期",
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    }

    df = df.rename(
        columns=rename_map
    )

    df["日期"] = pd.to_datetime(
        df["日期"]
    )

    df = (
        df.sort_values("日期")
        .reset_index(drop=True)
    )

    close = pd.to_numeric(
        df["收盘"],
        errors="coerce",
    )

    valid_close = close.dropna()

    if valid_close.empty:
        raise RuntimeError(
            "Historical close prices are empty."
        )

    last_250 = df.tail(250)

    latest_row = df.iloc[-1]

    latest_close = float(
        latest_row["收盘"]
    )

    latest_open = float(
        latest_row["开盘"]
    )

    latest_high = float(
        latest_row["最高"]
    )

    latest_low = float(
        latest_row["最低"]
    )

    previous_close = None
    daily_change_pct = None

    if len(df) >= 2:
        previous_close = float(
            df.iloc[-2]["收盘"]
        )

        if previous_close != 0:
            daily_change_pct = round(
                (
                    latest_close
                    / previous_close
                    - 1
                )
                * 100,
                2,
            )

    summary = {
        "latest_trading_date": str(
            latest_row["日期"].date()
        ),
        "latest_close": latest_close,
        "latest_open": latest_open,
        "latest_high": latest_high,
        "latest_low": latest_low,
        "previous_close": previous_close,
        "daily_change_pct": daily_change_pct,
        "return_5d_pct": _calculate_return(
            valid_close,
            5,
        ),
        "return_20d_pct": _calculate_return(
            valid_close,
            20,
        ),
        "return_60d_pct": _calculate_return(
            valid_close,
            60,
        ),
        "return_120d_pct": _calculate_return(
            valid_close,
            120,
        ),
        "return_250d_pct": _calculate_return(
            valid_close,
            250,
        ),
        "52_week_high": float(
            pd.to_numeric(
                last_250["最高"],
                errors="coerce",
            ).max()
        ),
        "52_week_low": float(
            pd.to_numeric(
                last_250["最低"],
                errors="coerce",
            ).min()
        ),
    }

    history = []

    for _, row in df.tail(180).iterrows():
        history.append(
            {
                "date": str(
                    row["日期"].date()
                ),
                "open": _to_python(
                    row.get("开盘")
                ),
                "close": _to_python(
                    row.get("收盘")
                ),
                "high": _to_python(
                    row.get("最高")
                ),
                "low": _to_python(
                    row.get("最低")
                ),
                "volume": _to_python(
                    row.get("成交量")
                ),
                "turnover": _to_python(
                    row.get("换手率")
                ),
                "amount": _to_python(
                    row.get("成交额")
                ),
            }
        )

    return {
        "summary": summary,
        "history": history,
    }