import operator
from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    user_query: str

    stock_code: str
    company_name: str
    exchange: str
    market_symbol: str

    search_result: dict[str, Any]

    company_data: dict[str, Any]
    financial_data: dict[str, Any]
    price_data: dict[str, Any]
    news_data: list[dict[str, Any]]
    fund_flow_data: dict[str, Any]
    sector_data: dict[str, Any]

    company_summary: str
    financial_summary: str
    price_summary: str
    news_summary: str

    ai_observation: str
    final_report: str

    errors: Annotated[
        list[str],
        operator.add,
    ]