import json
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from state import ResearchState
from tools.technical_tools import get_technical_data
from tools.valuation_tools import get_valuation_data
from tools.company_tools import get_company_data
from tools.financial_tools import get_financial_data
from tools.fund_flow_tools import get_fund_flow_data
from tools.news_tools import get_stock_news
from tools.price_tools import get_price_data
from tools.sector_tools import get_sector_data
from tools.stock_search import resolve_stock


load_dotenv()


@lru_cache(maxsize=1)
def get_llm():
    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    base_url = os.getenv(
        "OPENAI_BASE_URL"
    )

    model_name = os.getenv(
        "OPENAI_MODEL",
        "zai.glm-4.7-flash",
    )

    if not api_key:
        return None

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )


def _call_llm(
    prompt: str,
):
    llm = get_llm()

    if llm is None:
        return ""

    try:
        response = llm.invoke(
            prompt
        )

        return str(
            response.content
        ).strip()

    except Exception:
        return ""


def _to_number(
    value: Any,
):
    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _format_pct(
    value: Any,
):
    number = _to_number(
        value
    )

    if number is None:
        return "--"

    return f"{number:+.2f}%"


def _format_plain_pct(
    value: Any,
):
    number = _to_number(
        value
    )

    if number is None:
        return "--"

    return f"{number:.2f}%"


def _format_money(
    value: Any,
):
    number = _to_number(
        value
    )

    if number is None:
        return "--"

    sign = ""

    if number > 0:
        sign = "+"

    if abs(number) >= 100000000:
        return (
            f"{sign}"
            f"{number / 100000000:.2f}亿"
        )

    if abs(number) >= 10000:
        return (
            f"{sign}"
            f"{number / 10000:.2f}万"
        )

    return (
        f"{sign}"
        f"{number:.2f}"
    )


def _extract_profile_industry(
    company_data: dict[str, Any],
):
    if not company_data:
        return None

    preferred_keys = [
        "industry",
        "所属行业",
        "行业",
        "行业名称",
        "证监会行业",
    ]

    for key in preferred_keys:
        value = company_data.get(
            key
        )

        if (
            value is not None
            and str(value).strip()
        ):
            return str(
                value
            ).strip()

    for key, value in (
        company_data.items()
    ):
        if (
            "行业" in str(key)
            and value is not None
            and str(value).strip()
        ):
            return str(
                value
            ).strip()

    return None


def _normalize_key(
    value: Any,
):
    return (
        str(value)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def _find_value(
    data: Any,
    candidates: list[str],
):
    normalized_candidates = [
        _normalize_key(
            candidate
        )
        for candidate in candidates
    ]

    if isinstance(
        data,
        dict,
    ):
        for key, value in (
            data.items()
        ):
            normalized_key = (
                _normalize_key(
                    key
                )
            )

            for candidate in (
                normalized_candidates
            ):
                if (
                    normalized_key
                    == candidate
                    or candidate
                    in normalized_key
                ):
                    if not isinstance(
                        value,
                        (
                            dict,
                            list,
                        ),
                    ):
                        return value

        for value in (
            data.values()
        ):
            result = _find_value(
                value,
                candidates,
            )

            if result is not None:
                return result

    elif isinstance(
        data,
        list,
    ):
        for item in data[:3]:
            result = _find_value(
                item,
                candidates,
            )

            if result is not None:
                return result

    return None


def _extract_financial_metrics(
    financial_data: dict[str, Any],
):
    if not financial_data:
        return {}

    revenue = _find_value(
        financial_data,
        [
            "OPERATE_INCOME",
            "营业收入",
        ],
    )

    operating_profit = (
        _find_value(
            financial_data,
            [
                "OPERATE_PROFIT",
                "营业利润",
            ],
        )
    )

    operating_margin = None

    revenue_number = _to_number(
        revenue
    )

    operating_profit_number = (
        _to_number(
            operating_profit
        )
    )

    if (
        revenue_number is not None
        and revenue_number != 0
        and operating_profit_number
        is not None
    ):
        operating_margin = round(
            operating_profit_number
            / revenue_number
            * 100,
            2,
        )

    valuation = (
        financial_data.get(
            "valuation",
            {},
        )
    )

    return {
        "report_date": _find_value(
            financial_data,
            [
                "REPORT_DATE",
                "报告期",
                "报告日期",
                "日期",
            ],
        ),
        "revenue": revenue,
        "operating_profit": (
            operating_profit
        ),
        "operating_margin_pct": (
            operating_margin
        ),
        "net_profit": _find_value(
            financial_data,
            [
                "PARENT_NETPROFIT",
                "归母净利润",
                "归属于母公司股东的净利润",
            ],
        ),
        "eps": _find_value(
            financial_data,
            [
                "BASIC_EPS",
                "基本每股收益",
                "摊薄每股收益",
            ],
        ),
        "roe": _find_value(
            financial_data,
            [
                "ROE_WEIGHT",
                "加权净资产收益率",
                "净资产收益率",
            ],
        ),
        "debt_ratio": _find_value(
            financial_data,
            [
                "资产负债率",
                "DEBT_ASSET_RATIO",
            ],
        ),
        "operating_cash_flow": (
            _find_value(
                financial_data,
                [
                    "经营活动产生的现金流量净额",
                    "经营现金流量净额",
                    "NETCASH_OPERATE",
                ],
            )
        ),
        "pe": valuation.get(
            "pe"
        ),
        "pe_date": valuation.get(
            "pe_date"
        ),
        "pe_source": valuation.get(
            "source"
        ),
    }


def _build_ai_payload(
    state: ResearchState,
):
    price_data = state.get(
        "price_data",
        {},
    )

    price_summary = (
        price_data.get(
            "summary",
            {},
        )
    )

    fund_flow_data = state.get(
        "fund_flow_data",
        {},
    )

    sector_data = state.get(
        "sector_data",
        {},
    )

    news_data = state.get(
        "news_data",
        [],
    )

    business = (
        sector_data.get(
            "business_composition",
            {},
        )
    )

    products = (
        business.get(
            "product",
            [],
        )
    )

    financial_metrics = (
        _extract_financial_metrics(
            state.get(
                "financial_data",
                {},
            )
        )
    )

    return {
        "company": {
            "name": state.get(
                "company_name"
            ),
            "stock_code": state.get(
                "stock_code"
            ),
        },
        "market": {
            "latest_close": (
                price_summary.get(
                    "latest_close"
                )
            ),
            "daily_change_pct": (
                price_summary.get(
                    "daily_change_pct"
                )
            ),
            "return_5d_pct": (
                price_summary.get(
                    "return_5d_pct"
                )
            ),
            "return_20d_pct": (
                price_summary.get(
                    "return_20d_pct"
                )
            ),
            "return_60d_pct": (
                price_summary.get(
                    "return_60d_pct"
                )
            ),
            "technical": (
                price_data.get(
                    "technical",
                    {},
                )
            ),
        },
        "fund_flow": {
             "source": (
                 fund_flow_data
                 .get(
                     "fund_flow",
                     {},
                )
                .get(
                    "source"
                )
            ),
            "flow_type": (
                fund_flow_data
                .get(
                    "fund_flow",
                    {},
                )
                .get(
                    "flow_type"
                )
            ),
            "latest": (
                fund_flow_data
                .get(
                    "fund_flow",
                    {},
                )
                .get(
                    "latest",
                    {},
                )
            ),
            "turnover": (
                fund_flow_data.get(
                    "turnover",
                    {},
                )
            ),
            "signal": (
                fund_flow_data.get(
                    "signal"
                )
            ),
            "source_note": (
                fund_flow_data.get(
                    "source_note"
                )
            ),
        },
        "sector": {
            "snapshot": (
                sector_data.get(
                    "industry_snapshot",
                    {},
                )
            ),
            "relative_performance": (
                sector_data.get(
                    "relative_performance",
                    {},
                )
            ),
            "signal": (
                sector_data.get(
                    "sector_signal"
                )
            ),
        },
        "financial": (
            financial_metrics
        ),
        "business": (
            products[:3]
        ),
        "recent_announcements": [
            {
                "date": item.get(
                    "published_at"
                ),
                "freshness": item.get(
                    "freshness"
                ),
                "title": item.get(
                    "title"
                ),
            }
            for item in news_data[:5]
        ],
    }


def _build_fallback_observation(
    state: ResearchState,
):
    observations = []

    fund_signal = (
        state.get(
            "fund_flow_data",
            {},
        ).get(
            "signal"
        )
    )

    if fund_signal:
        observations.append(
            f"- {fund_signal}"
        )

    sector_data = state.get(
        "sector_data",
        {},
    )

    sector_signal = (
        sector_data.get(
            "sector_signal"
        )
    )

    if sector_signal:
        observations.append(
            f"- {sector_signal}"
        )

    relative = (
        sector_data.get(
            "relative_performance",
            {},
        )
    )

    twenty_day = relative.get(
        "20d",
        {},
    )

    excess_20d = (
        twenty_day.get(
            "excess_return_pct"
        )
    )

    if excess_20d is not None:
        observations.append(
            "- 20日相对板块表现："
            f"{_format_pct(excess_20d)}"
        )

    news = state.get(
        "news_data",
        [],
    )

    if news:
        latest = news[0]

        observations.append(
            "- 最近公告："
            f"{latest.get('published_at')} "
            f"{latest.get('title')}"
        )

    if not observations:
        return (
            "- 当前结构化数据不足，"
            "暂不生成额外判断。"
        )

    return "\n".join(
        observations[:4]
    )


def _build_report(
    state: ResearchState,
    ai_observation: str,
):
    company_name = state.get(
        "company_name",
        state.get(
            "user_query",
            "",
        ),
    )

    stock_code = state.get(
        "stock_code",
        "",
    )

    price_summary = (
        state.get(
            "price_data",
            {},
        ).get(
            "summary",
            {},
        )
    )

    fund_flow_data = state.get(
        "fund_flow_data",
        {},
    )

    sector_data = state.get(
        "sector_data",
        {},
    )

    financial_metrics = (
        _extract_financial_metrics(
            state.get(
                "financial_data",
                {},
            )
        )
    )

    news_data = state.get(
        "news_data",
        [],
    )

    lines = [
        (
            f"# {company_name} "
            f"({stock_code})"
        )
    ]

    lines.append(
        "\n## 市场概况"
    )

    latest_close = (
        price_summary.get(
            "latest_close"
        )
    )

    if latest_close is not None:
        lines.append(
            "- 最新收盘："
            f"{latest_close} 元"
            f"（{_format_pct(price_summary.get('daily_change_pct'))}）"
        )

    lines.append(
        "- 阶段表现："
        f"5日 {_format_pct(price_summary.get('return_5d_pct'))}，"
        f"20日 {_format_pct(price_summary.get('return_20d_pct'))}，"
        f"60日 {_format_pct(price_summary.get('return_60d_pct'))}"
    )

    turnover = (
        fund_flow_data.get(
            "turnover",
            {},
        )
    )

    if turnover:
        lines.append(
            "- 换手率："
            f"{_format_plain_pct(turnover.get('current_turnover_rate_pct'))}"
            "；5日均值 "
            f"{_format_plain_pct(turnover.get('avg_turnover_5d_pct'))}"
            "；20日均值 "
            f"{_format_plain_pct(turnover.get('avg_turnover_20d_pct'))}"
            "；活跃度 "
            f"{turnover.get('turnover_activity_ratio', '--')}×"
        )
    technical = (
        state.get(
            "price_data",
            {},
        ).get(
            "technical",
            {},
        )
    )

    volume = technical.get(
        "volume",
        {},
    )

    if volume:
        lines.append(
            "- 成交量："
            f"{volume.get('latest_volume_wan_shou', '--')} 万手"
            "；5日均量 "
            f"{volume.get('avg_volume_5d_wan_shou', '--')} 万手"
            "；20日均量 "
            f"{volume.get('avg_volume_20d_wan_shou', '--')} 万手"
            "；约为20日均量 "
            f"{volume.get('volume_vs_20d_avg', '--')}×"
        )

    kdj = technical.get(
        "kdj",
        {},
    )

    if kdj:
        lines.append(
            "- KDJ："
            f"K {kdj.get('k', '--')}，"
            f"D {kdj.get('d', '--')}，"
            f"J {kdj.get('j', '--')}；"
            f"{kdj.get('zone', '')}，"
            f"{kdj.get('relation', '')}"
        )

    bollinger = technical.get(
        "bollinger",
        {},
    )

    if bollinger:
        lines.append(
            "- 布林轨道："
            f"上轨 {bollinger.get('upper', '--')}，"
            f"中轨 {bollinger.get('middle', '--')}，"
            f"下轨 {bollinger.get('lower', '--')}；"
            f"当前位于{bollinger.get('position', '')}"
        )
        
    fund_flow = (
        fund_flow_data.get(
            "fund_flow",
            {},
        )
    )

    latest_flow = (
        fund_flow.get(
            "latest",
            {},
        )
    )

    source = fund_flow.get(
        "source"
    )

    if source == "东方财富":
        lines.append(
            "- 主力资金："
            f"{_format_money(latest_flow.get('main_net_inflow'))}"
            "；"
            f"{fund_flow_data.get('signal', '')}"
        )

    elif source == "同花顺":
        lines.append(
            "- 整体资金："
            f"{_format_money(latest_flow.get('net_flow'))}"
            "；"
            f"{fund_flow_data.get('signal', '')}"
        )

    lines.append(
        "\n## 板块与相对表现"
    )

    snapshot = (
        sector_data.get(
            "industry_snapshot",
            {},
        )
    )

    if snapshot:
        sector_name = (
            snapshot.get(
                "industry_name",
                "--",
            )
        )

        net_flow_billion = (
            snapshot.get(
                "net_flow_billion"
            )
        )

        rank = snapshot.get(
            "fund_flow_rank"
        )

        count = snapshot.get(
            "industry_count"
        )

        sector_line = (
            f"- {sector_name}："
            f"{_format_pct(snapshot.get('change_pct'))}"
        )

        if (
            net_flow_billion
            is not None
        ):
            sector_line += (
                f"，资金净额 "
                f"{net_flow_billion:+.2f} 亿"
            )

        if (
            rank is not None
            and count is not None
        ):
            sector_line += (
                f"，资金排名 "
                f"{rank}/{count}"
            )

        sector_line += (
            f"；{sector_data.get('sector_signal', '')}"
        )

        lines.append(
            sector_line
        )

    relative = (
        sector_data.get(
            "relative_performance",
            {},
        )
    )

    for period in [
        "5d",
        "20d",
        "60d",
    ]:
        data = relative.get(
            period
        )

        if not data:
            continue

        label = period.replace(
            "d",
            "日",
        )

        lines.append(
            f"- {label}："
            f"个股 {_format_pct(data.get('stock_return_pct'))}，"
            f"板块 {_format_pct(data.get('sector_return_pct'))}，"
            f"相对表现 {_format_pct(data.get('excess_return_pct'))}"
        )

    lines.append(
        "\n## 基本面"
    )

    report_date = (
        financial_metrics.get(
            "report_date"
        )
    )

    if report_date:
        lines.append(
            f"- 财务报告期：{report_date}"
        )

    revenue = (
        financial_metrics.get(
            "revenue"
        )
    )

    net_profit = (
        financial_metrics.get(
            "net_profit"
        )
    )

    if (
        revenue is not None
        or net_profit is not None
    ):
        lines.append(
            "- 营收 / 归母净利润："
            f"{_format_money(revenue)} / "
            f"{_format_money(net_profit)}"
        )

    roe = (
        financial_metrics.get(
            "roe"
        )
    )

    debt_ratio = (
        financial_metrics.get(
            "debt_ratio"
        )
    )

    net_margin = (
        financial_metrics.get(
            "net_margin"
        )
    )

    financial_parts = []

    pe = financial_metrics.get(
        "pe"
    )

    if pe is not None:
        financial_parts.append(
            f"市盈率 {float(pe):.2f}×"
        )

    operating_margin = (
        financial_metrics.get(
            "operating_margin_pct"
        )
    )

    if operating_margin is not None:
        financial_parts.append(
            "营业利润率 "
            f"{float(operating_margin):.2f}%"
        )

    roe = financial_metrics.get(
        "roe"
    )

    if roe is not None:
        financial_parts.append(
            "ROE "
            f"{_format_plain_pct(roe)}"
        )

    debt_ratio = (
        financial_metrics.get(
            "debt_ratio"
        )
    )

    if debt_ratio is not None:
        financial_parts.append(
            "资产负债率 "
            f"{_format_plain_pct(debt_ratio)}"
        )

    if financial_parts:
        lines.append(
            "- "
            + "；".join(
                financial_parts
            )
        )

    operating_cash_flow = (
        financial_metrics.get(
            "operating_cash_flow"
        )
    )

    if operating_cash_flow is not None:
        lines.append(
            "- 经营现金流："
            f"{_format_money(operating_cash_flow)}"
        )

    business = (
        sector_data.get(
            "business_composition",
            {},
        )
    )

    products = business.get(
        "product",
        [],
    )

    if products:
        lines.append(
            "\n## 主营构成"
        )

        report_date = business.get(
            "report_date"
        )

        if report_date:
            lines.append(
                f"- 数据期：{report_date}"
            )

        for item in products[:4]:
            lines.append(
                "- "
                f"{item.get('name', '--')}："
                f"收入占比 "
                f"{_format_plain_pct(item.get('revenue_share_pct'))}，"
                f"毛利率 "
                f"{_format_plain_pct(item.get('gross_margin_pct'))}"
            )

    lines.append(
        "\n## 近期公告"
    )

    if news_data:
        for item in news_data[:5]:
            lines.append(
                "- "
                f"{item.get('published_at', '--')} "
                f"[{item.get('freshness', '未知')}] "
                f"{item.get('title', '')}"
            )

    else:
        lines.append(
            "- 近期未获取到公告。"
        )

    lines.append(
        "\n## AI观察"
    )

    lines.append(
        ai_observation
    )

    lines.append(
        "\n> 数据用于研究与信息整理，不构成投资建议。"
    )

    return "\n".join(
        lines
    )


def company_agent(
    state: ResearchState,
):
    query = state.get(
        "user_query",
        "",
    )

    try:
        search_result = (
            resolve_stock(
                query
            )
        )

        if not search_result:
            raise RuntimeError(
                "无法识别股票名称或代码。"
            )

        stock_code = (
            search_result.get(
                "stock_code"
            )
        )

        company_name = (
            search_result.get(
                "company_name"
            )
        )

        company_data = (
            get_company_data(
                stock_code
            )
        )

        return {
            "search_result": (
                search_result
            ),
            "stock_code": (
                stock_code
            ),
            "company_name": (
                company_name
            ),
            "exchange": (
                search_result.get(
                    "exchange"
                )
            ),
            "market_symbol": (
                search_result.get(
                    "market_symbol"
                )
            ),
            "company_data": (
                company_data
            ),
        }

    except Exception as exc:
        return {
            "errors": [
                "Company agent: "
                f"{exc}"
            ]
        }


def fund_flow_agent(
    state: ResearchState,
):
    stock_code = state.get(
        "stock_code"
    )

    if not stock_code:
        return {
            "errors": [
                "Fund flow agent: "
                "missing stock code."
            ]
        }

    try:
        data = (
            get_fund_flow_data(
                stock_code
            )
        )

        return {
            "fund_flow_data": (
                data
            )
        }

    except Exception as exc:
        return {
            "errors": [
                "Fund flow agent: "
                f"{exc}"
            ]
        }


def financial_agent(
    state: ResearchState,
):
    stock_code = state.get(
        "stock_code"
    )

    market_symbol = state.get(
        "market_symbol"
    )

    if not stock_code:
        return {
            "errors": [
                "Financial agent: "
                "missing stock code."
            ]
        }

    if not market_symbol:
        return {
            "errors": [
                "Financial agent: "
                "missing market symbol."
            ]
        }

    try:
        data = (
            get_financial_data(
                stock_code,
                market_symbol,
            )
        )

        valuation = (
            get_valuation_data(
                stock_code
            )
        )

        data["valuation"] = (
            valuation
        )

        return {
            "financial_data": data
        }

    except Exception as exc:
        return {
            "errors": [
                "Financial agent: "
                f"{exc}"
            ]
        }


def price_agent(
    state: ResearchState,
):
    stock_code = state.get(
        "stock_code"
    )

    if not stock_code:
        return {
            "errors": [
                "Price agent: "
                "missing stock code."
            ]
        }

    try:
        data = (
            get_price_data(
                stock_code
            )
        )

        technical_data = (
            get_technical_data(
                data.get(
                    "history",
                    [],
                )
            )
        )

        data["technical"] = (
            technical_data
        )

        return {
            "price_data": data
        }

    except Exception as exc:
        return {
            "errors": [
                "Price agent: "
                f"{exc}"
            ]
        }


def news_agent(
    state: ResearchState,
):
    stock_code = state.get(
        "stock_code"
    )

    company_name = state.get(
        "company_name"
    )

    if not stock_code:
        return {
            "errors": [
                "News agent: "
                "missing stock code."
            ]
        }

    try:
        data = (
            get_stock_news(
                stock_code=stock_code,
                company_name=company_name,
                limit=8,
                include_media=False,
            )
        )

        return {
            "news_data": data
        }

    except Exception as exc:
        return {
            "errors": [
                "News agent: "
                f"{exc}"
            ]
        }


def sector_agent(
    state: ResearchState,
):
    stock_code = state.get(
        "stock_code"
    )

    if not stock_code:
        return {
            "errors": [
                "Sector agent: "
                "missing stock code."
            ]
        }

    company_data = state.get(
        "company_data",
        {},
    )

    profile_industry = (
        _extract_profile_industry(
            company_data
        )
    )

    stock_price_summary = (
        state.get(
            "price_data",
            {},
        ).get(
            "summary",
            {},
        )
    )

    try:
        data = (
            get_sector_data(
                stock_code=stock_code,
                profile_industry=(
                    profile_industry
                ),
                stock_price_summary=(
                    stock_price_summary
                ),
            )
        )

        return {
            "sector_data": data
        }

    except Exception as exc:
        return {
            "errors": [
                "Sector agent: "
                f"{exc}"
            ]
        }


def report_agent(
    state: ResearchState,
):
    payload = (
        _build_ai_payload(
            state
        )
    )

    prompt = f"""
你是一个A股研究助手。

下面所有数字都来自外部数据接口或 Python 计算。
你只负责解释这些数据，不能创造、猜测或补充数据中不存在的事实。

请输出3到4条非常简短的中文观察。

严格遵守以下规则：

1. 每条只写一句，总体控制在150字以内。

2. 优先观察价格、资金、换手率、板块表现、
   相对表现、基本面和公告之间的一致或背离。

3. 如果 fund_flow.flow_type == "overall"，
   只能称为“整体资金”“资金净流入/流出”，
   禁止使用“主力资金”“主力流入”“主力流出”。

4. 只有 fund_flow.flow_type == "main_force" 时，
   才允许使用“主力资金”。

5. 不得根据涨跌幅自行判断“涨停”或“跌停”。
   如果数据只提供涨跌幅，就直接描述具体涨跌幅。

6. 不得因为股价跑赢板块，
   就说“基本面优于行业”“公司质量优于行业”
   或类似结论。
   只有存在明确的行业财务指标比较数据时才能这样说。

7. 不得根据单日成交额自行说“成交额放大”。
   如需描述交易活跃度，
   应根据当前换手率与5日或20日平均换手率进行比较。

8. 主营业务数据和公告数据要区分。
   主营构成只能描述收入占比、毛利率等已有数据，
   不得说某份公告“显示”了这些数据，
   除非输入明确说明两者来自同一份公告。

9. 如果提到公告，必须写明具体日期。

10. 不给出买入、卖出、目标价、上涨概率或价格预测。

11. 不得补充 JSON 中不存在的事实。

数据：
{json.dumps(
    payload,
    ensure_ascii=False,
    default=str,
)}
"""
    ai_observation = (
        _call_llm(
            prompt
        )
    )

    if not ai_observation:
        ai_observation = (
            _build_fallback_observation(
                state
            )
        )

    final_report = (
        _build_report(
            state,
            ai_observation,
        )
    )

    return {
        "ai_observation": (
            ai_observation
        ),
        "final_report": (
            final_report
        ),
    }