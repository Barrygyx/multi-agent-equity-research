from typing import Any

import pandas as pd


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
    except (TypeError, ValueError):
        return None


def _prepare_history(
    price_history: list[dict[str, Any]],
):
    if not price_history:
        return pd.DataFrame()

    df = pd.DataFrame(price_history)

    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }

    df = df.rename(
        columns=column_map
    )

    required = [
        "close",
        "high",
        "low",
    ]

    for column in required:
        if column not in df.columns:
            return pd.DataFrame()

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "turnover",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = df.sort_values(
            "date"
        )

    df = df.reset_index(
        drop=True
    )

    return df


def _calculate_volume(
    df: pd.DataFrame,
):
    if (
        df.empty
        or "volume" not in df.columns
    ):
        return {}

    volume = (
        df["volume"]
        .dropna()
    )

    if volume.empty:
        return {}

    latest = float(
        volume.iloc[-1]
    )

    avg_5 = float(
        volume.tail(5).mean()
    )

    avg_20 = float(
        volume.tail(20).mean()
    )

    volume_ratio_20d = None

    if avg_20 > 0:
        volume_ratio_20d = round(
            latest / avg_20,
            2,
        )

    return {
        "latest_volume_shares": round(
            latest,
            0,
        ),
        "latest_volume_wan_shou": round(
            latest / 1_000_000,
            2,
        ),
        "avg_volume_5d_shares": round(
            avg_5,
            0,
        ),
        "avg_volume_5d_wan_shou": round(
            avg_5 / 1_000_000,
            2,
        ),
        "avg_volume_20d_shares": round(
            avg_20,
            0,
        ),
        "avg_volume_20d_wan_shou": round(
            avg_20 / 1_000_000,
            2,
        ),
        "volume_vs_20d_avg": (
            volume_ratio_20d
        ),
    }


def _calculate_kdj(
    df: pd.DataFrame,
    period: int = 9,
):
    if len(df) < period:
        return {}

    high = df["high"]
    low = df["low"]
    close = df["close"]

    lowest_low = low.rolling(
        window=period,
        min_periods=period,
    ).min()

    highest_high = high.rolling(
        window=period,
        min_periods=period,
    ).max()

    denominator = (
        highest_high
        - lowest_low
    )

    rsv = (
        (close - lowest_low)
        / denominator.replace(
            0,
            pd.NA,
        )
        * 100
    )

    k_values = []
    d_values = []
    j_values = []

    previous_k = 50.0
    previous_d = 50.0

    for value in rsv:
        if pd.isna(value):
            k_values.append(
                None
            )
            d_values.append(
                None
            )
            j_values.append(
                None
            )
            continue

        current_k = (
            2 / 3 * previous_k
            + 1 / 3 * float(value)
        )

        current_d = (
            2 / 3 * previous_d
            + 1 / 3 * current_k
        )

        current_j = (
            3 * current_k
            - 2 * current_d
        )

        k_values.append(
            current_k
        )

        d_values.append(
            current_d
        )

        j_values.append(
            current_j
        )

        previous_k = current_k
        previous_d = current_d

    valid = [
        (
            k,
            d,
            j,
        )
        for k, d, j in zip(
            k_values,
            d_values,
            j_values,
        )
        if (
            k is not None
            and d is not None
            and j is not None
        )
    ]

    if not valid:
        return {}

    k, d, j = valid[-1]

    if k >= 80 and d >= 80:
        zone = "高位"

    elif k <= 20 and d <= 20:
        zone = "低位"

    else:
        zone = "中性区间"

    if k > d:
        relation = "K线上方"

    elif k < d:
        relation = "K线下方"

    else:
        relation = "K与D接近"

    return {
        "period": period,
        "k": round(
            k,
            2,
        ),
        "d": round(
            d,
            2,
        ),
        "j": round(
            j,
            2,
        ),
        "zone": zone,
        "relation": relation,
    }


def _calculate_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_multiplier: float = 2.0,
):
    if len(df) < period:
        return {}

    close = df["close"]

    middle = close.rolling(
        window=period,
    ).mean()

    std = close.rolling(
        window=period,
    ).std(
        ddof=0
    )

    upper = (
        middle
        + std_multiplier * std
    )

    lower = (
        middle
        - std_multiplier * std
    )

    latest_close = _to_number(
        close.iloc[-1]
    )

    latest_middle = _to_number(
        middle.iloc[-1]
    )

    latest_upper = _to_number(
        upper.iloc[-1]
    )

    latest_lower = _to_number(
        lower.iloc[-1]
    )

    if None in {
        latest_close,
        latest_middle,
        latest_upper,
        latest_lower,
    }:
        return {}

    width = (
        latest_upper
        - latest_lower
    )

    position_pct = None

    if width > 0:
        position_pct = (
            (
                latest_close
                - latest_lower
            )
            / width
            * 100
        )

    if latest_close > latest_upper:
        position = "上轨上方"

    elif latest_close < latest_lower:
        position = "下轨下方"

    elif latest_close >= latest_middle:
        position = "中轨与上轨之间"

    else:
        position = "下轨与中轨之间"

    bandwidth_pct = None

    if latest_middle != 0:
        bandwidth_pct = (
            width
            / latest_middle
            * 100
        )

    return {
        "period": period,
        "upper": round(
            latest_upper,
            2,
        ),
        "middle": round(
            latest_middle,
            2,
        ),
        "lower": round(
            latest_lower,
            2,
        ),
        "close": round(
            latest_close,
            2,
        ),
        "position": position,
        "position_pct": (
            round(
                position_pct,
                2,
            )
            if position_pct
            is not None
            else None
        ),
        "bandwidth_pct": (
            round(
                bandwidth_pct,
                2,
            )
            if bandwidth_pct
            is not None
            else None
        ),
    }


def get_technical_data(
    price_history: list[dict[str, Any]],
):
    df = _prepare_history(
        price_history
    )

    if df.empty:
        return {
            "volume": {},
            "kdj": {},
            "bollinger": {},
        }

    return {
        "volume": (
            _calculate_volume(
                df
            )
        ),
        "kdj": (
            _calculate_kdj(
                df
            )
        ),
        "bollinger": (
            _calculate_bollinger(
                df
            )
        ),
    }