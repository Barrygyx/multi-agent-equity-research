from typing import Any

import pandas as pd


def _prepare_price_df(
    price_history: list[dict[str, Any]],
):
    if not price_history:
        return pd.DataFrame()

    df = pd.DataFrame(
        price_history
    )

    rename_map = {
        "日期": "date",
        "date": "date",
        "开盘": "open",
        "open": "open",
        "收盘": "close",
        "close": "close",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "成交量": "volume",
        "volume": "volume",
        "成交额": "amount",
        "amount": "amount",
        "换手率": "turnover",
        "turnover": "turnover",
    }

    df = df.rename(
        columns={
            column: rename_map[column]
            for column in df.columns
            if column in rename_map
        }
    )

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:
        if column not in df.columns:
            return pd.DataFrame()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = (
        df.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def _resample_price(
    df: pd.DataFrame,
    period: str,
):
    if df.empty:
        return df

    if period == "日K":
        return df.copy()

    temp = (
        df.set_index("date")
        .sort_index()
    )

    if period == "周K":
        rule = "W-FRI"

    elif period == "月K":
        rule = "ME"

    else:
        return df.copy()

    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }

    if "volume" in temp.columns:
        aggregation[
            "volume"
        ] = "sum"

    if "amount" in temp.columns:
        aggregation[
            "amount"
        ] = "sum"

    if "turnover" in temp.columns:
        aggregation[
            "turnover"
        ] = "sum"

    result = (
        temp.resample(rule)
        .agg(aggregation)
        .dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index()
    )

    return result


def _add_ma(
    df: pd.DataFrame,
):
    df = df.copy()

    for period in [
        5,
        10,
        20,
        30,
    ]:
        df[
            f"ma{period}"
        ] = (
            df["close"]
            .rolling(period)
            .mean()
        )

    if "volume" in df.columns:
        df["volume_ma5"] = (
            df["volume"]
            .rolling(5)
            .mean()
        )

        df["volume_ma10"] = (
            df["volume"]
            .rolling(10)
            .mean()
        )

    return df


def _add_kdj(
    df: pd.DataFrame,
    period: int = 9,
):
    df = df.copy()

    low_n = (
        df["low"]
        .rolling(
            period,
            min_periods=period,
        )
        .min()
    )

    high_n = (
        df["high"]
        .rolling(
            period,
            min_periods=period,
        )
        .max()
    )

    denominator = (
        high_n - low_n
    ).replace(
        0,
        pd.NA,
    )

    rsv = (
        (
            df["close"]
            - low_n
        )
        / denominator
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
            2 / 3
            * previous_k
            + 1 / 3
            * float(value)
        )

        current_d = (
            2 / 3
            * previous_d
            + 1 / 3
            * current_k
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

        previous_k = (
            current_k
        )

        previous_d = (
            current_d
        )

    df["k"] = k_values
    df["d"] = d_values
    df["j"] = j_values

    return df


def _add_bollinger(
    df: pd.DataFrame,
    period: int = 20,
):
    df = df.copy()

    middle = (
        df["close"]
        .rolling(period)
        .mean()
    )

    std = (
        df["close"]
        .rolling(period)
        .std(ddof=0)
    )

    df["boll_mid"] = middle

    df["boll_upper"] = (
        middle
        + 2 * std
    )

    df["boll_lower"] = (
        middle
        - 2 * std
    )

    return df


def get_chart_data(
    price_history: list[dict[str, Any]],
    period: str = "日K",
):
    df = _prepare_price_df(
        price_history
    )

    if df.empty:
        return pd.DataFrame()

    df = _resample_price(
        df,
        period,
    )

    df = _add_ma(
        df
    )

    df = _add_kdj(
        df
    )

    df = _add_bollinger(
        df
    )

    return df