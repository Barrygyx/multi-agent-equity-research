from typing import Any

import akshare as ak
import pandas as pd


def _to_python(value: Any):
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def _sort_by_date(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    possible_date_columns = [
        "REPORT_DATE",
        "REPORT_DATE_NAME",
        "日期",
        "报告期",
    ]

    for column in possible_date_columns:
        if column not in df.columns:
            continue

        try:
            parsed_date = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            if parsed_date.notna().any():
                df["_sort_date"] = parsed_date

                df = (
                    df.sort_values(
                        "_sort_date",
                        ascending=False,
                    )
                    .drop(
                        columns=["_sort_date"]
                    )
                    .reset_index(drop=True)
                )

                return df

        except Exception:
            continue

    return df.reset_index(drop=True)


def _compact_records(
    df: pd.DataFrame,
    fields: list[str],
    limit: int = 6,
) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    df = _sort_by_date(df)

    available_fields = [
        field
        for field in fields
        if field in df.columns
    ]

    if not available_fields:
        available_fields = list(
            df.columns[:15]
        )

    records = []

    for _, row in df.head(limit).iterrows():
        record = {}

        for field in available_fields:
            value = _to_python(
                row[field]
            )

            if value is not None:
                record[field] = value

        records.append(record)

    return records


def get_financial_data(
    stock_code: str,
    market_symbol: str,
) -> dict[str, Any]:
    result = {
        "financial_indicators": [],
        "income_statement": [],
        "balance_sheet": [],
        "cash_flow": [],
        "warnings": [],
    }

    indicator_fields = [
        "REPORT_DATE",
        "REPORT_DATE_NAME",
        "日期",
        "报告期",
        "ROE",
        "ROA",
        "GROSS_PROFIT_MARGIN",
        "NET_PROFIT_MARGIN",
        "DEBT_ASSET_RATIO",
        "CURRENT_RATIO",
        "BASIC_EPS",
        "DILUTED_EPS",
        "每股收益",
        "摊薄每股收益(元)",
        "加权净资产收益率(%)",
        "销售毛利率(%)",
        "销售净利率(%)",
        "资产负债率(%)",
        "每股经营性现金流(元)",
    ]

    income_fields = [
        "REPORT_DATE",
        "REPORT_DATE_NAME",
        "营业总收入",
        "营业收入",
        "营业利润",
        "利润总额",
        "净利润",
        "归属于母公司股东的净利润",
        "TOTAL_OPERATE_INCOME",
        "OPERATE_INCOME",
        "OPERATE_PROFIT",
        "TOTAL_PROFIT",
        "NETPROFIT",
        "PARENT_NETPROFIT",
        "BASIC_EPS",
        "DILUTE_EPS",
        "WEIGHTAVG_ROE",
    ]

    balance_fields = [
        "REPORT_DATE",
        "REPORT_DATE_NAME",
        "资产总计",
        "负债合计",
        "股东权益合计",
        "所有者权益合计",
        "货币资金",
        "流动资产合计",
        "流动负债合计",
        "TOTAL_ASSETS",
        "TOTAL_LIABILITIES",
        "TOTAL_EQUITY",
        "MONETARYFUNDS",
        "TOTAL_CURRENT_ASSETS",
        "TOTAL_CURRENT_LIAB",
    ]

    cash_fields = [
        "REPORT_DATE",
        "REPORT_DATE_NAME",
        "经营活动产生的现金流量净额",
        "投资活动产生的现金流量净额",
        "筹资活动产生的现金流量净额",
        "现金及现金等价物净增加额",
        "NETCASH_OPERATE",
        "NETCASH_INVEST",
        "NETCASH_FINANCE",
        "CCE_ADD",
        "OPERATE_CASH_FLOW",
        "INVEST_CASH_FLOW",
        "FINANCE_CASH_FLOW",
    ]

    try:
        indicator_df = (
            ak.stock_financial_analysis_indicator_em(
                symbol=stock_code
            )
        )

        result["financial_indicators"] = (
            _compact_records(
                indicator_df,
                indicator_fields,
            )
        )

    except Exception as exc:
        try:
            indicator_df = (
                ak.stock_financial_analysis_indicator(
                    symbol=stock_code
                )
            )

            result["financial_indicators"] = (
                _compact_records(
                    indicator_df,
                    indicator_fields,
                )
            )

        except Exception as fallback_exc:
            result["warnings"].append(
                "Financial indicators unavailable: "
                f"{exc}; fallback: "
                f"{fallback_exc}"
            )

    try:
        income_df = (
            ak.stock_profit_sheet_by_report_em(
                symbol=market_symbol
            )
        )

        result["income_statement"] = (
            _compact_records(
                income_df,
                income_fields,
            )
        )

    except Exception as exc:
        result["warnings"].append(
            f"Income statement unavailable: {exc}"
        )

    try:
        balance_df = (
            ak.stock_balance_sheet_by_report_em(
                symbol=market_symbol
            )
        )

        result["balance_sheet"] = (
            _compact_records(
                balance_df,
                balance_fields,
            )
        )

    except Exception as exc:
        result["warnings"].append(
            f"Balance sheet unavailable: {exc}"
        )

    try:
        cash_df = (
            ak.stock_cash_flow_sheet_by_report_em(
                symbol=market_symbol
            )
        )

        result["cash_flow"] = (
            _compact_records(
                cash_df,
                cash_fields,
            )
        )

    except Exception as exc:
        result["warnings"].append(
            f"Cash flow statement unavailable: {exc}"
        )

    return result