from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from agents import report_agent
from tools.chart_tools import get_chart_data
from tools.company_tools import get_company_data
from tools.financial_tools import get_financial_data
from tools.fund_flow_tools import get_fund_flow_data
from tools.news_tools import get_stock_news
from tools.price_tools import get_price_data
from tools.sector_tools import get_sector_data
from tools.stock_search import resolve_stock
from tools.technical_tools import get_technical_data
from tools.valuation_tools import get_valuation_data


st.set_page_config(
    page_title="A股研究助手",
    page_icon="📈",
    layout="wide",
)


# 页面样式
st.markdown(
    """
<style>

.block-container {
    padding-top: 5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

.stock-header {
    margin-top: 24px;
    margin-bottom: 22px;
}

.stock-name {
    font-size: 34px;
    font-weight: 700;
    color: #222222;
    line-height: 1.3;
}

.stock-code {
    color: #888888;
    font-size: 16px;
    margin-top: 5px;
}

.price-up {
    font-size: 48px;
    font-weight: 700;
    color: #e53935;
    line-height: 1.1;
}

.price-down {
    font-size: 48px;
    font-weight: 700;
    color: #16863a;
    line-height: 1.1;
}

.change-up {
    color: #e53935;
    font-size: 21px;
    font-weight: 600;
    margin-top: 10px;
}

.change-down {
    color: #16863a;
    font-size: 21px;
    font-weight: 600;
    margin-top: 10px;
}

.info-row {
    margin-bottom: 14px;
}

.info-label {
    color: #777777;
    font-size: 15px;
    margin-right: 8px;
}

.info-value {
    color: #333333;
    font-size: 19px;
    font-weight: 600;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 24px;
    margin-bottom: 12px;
    color: #222222;
}

.small-note {
    color: #888888;
    font-size: 14px;
}

div[data-testid="stMetric"] {
    background-color: #fafafa;
    border: 1px solid #eeeeee;
    padding: 12px 14px;
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# 通用辅助函数
def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value):
    number = _number(value)

    if number is None:
        return "--"

    return f"{number:+.2f}%"


def _plain_pct(value):
    number = _number(value)

    if number is None:
        return "--"

    return f"{number:.2f}%"


def _money(value):
    number = _number(value)

    if number is None:
        return "--"

    if abs(number) >= 100000000:
        return f"{number / 100000000:.2f}亿"

    if abs(number) >= 10000:
        return f"{number / 10000:.2f}万"

    return f"{number:.2f}"


def _find_value(data, keys):
    if isinstance(data, dict):
        for key, value in data.items():
            key_text = str(key)

            for candidate in keys:
                if candidate == key_text or candidate in key_text:
                    if not isinstance(value, (dict, list)):
                        return value

        for value in data.values():
            result = _find_value(value, keys)

            if result is not None:
                return result

    elif isinstance(data, list):
        for item in data[:3]:
            result = _find_value(item, keys)

            if result is not None:
                return result

    return None


def _extract_industry(company_data):
    if not company_data:
        return None

    for key in [
        "industry",
        "所属行业",
        "行业",
        "行业名称",
        "证监会行业",
    ]:
        value = company_data.get(key)

        if value:
            return str(value).strip()

    for key, value in company_data.items():
        if "行业" in str(key) and value:
            return str(value).strip()

    return None


# Streamlit 缓存
@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def cached_resolve_stock(query):
    return resolve_stock(query)


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def cached_company_data(stock_code):
    return get_company_data(stock_code)


@st.cache_data(
    ttl=120,
    show_spinner=False,
)
def cached_price_data(stock_code):
    data = get_price_data(stock_code)

    data["technical"] = get_technical_data(
        data.get("history", [])
    )

    return data


@st.cache_data(
    ttl=120,
    show_spinner=False,
)
def cached_fund_flow(stock_code):
    return get_fund_flow_data(stock_code)


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def cached_financial_data(
    stock_code,
    market_symbol,
):
    data = get_financial_data(
        stock_code,
        market_symbol,
    )

    valuation = get_valuation_data(
        stock_code
    )

    data["valuation"] = valuation

    return data


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def cached_news(
    stock_code,
    company_name,
):
    return get_stock_news(
        stock_code=stock_code,
        company_name=company_name,
        limit=8,
        include_media=False,
    )


@st.cache_data(
    ttl=600,
    show_spinner=False,
)
def cached_sector(
    stock_code,
    profile_industry,
    price_summary,
):
    return get_sector_data(
        stock_code=stock_code,
        profile_industry=profile_industry,
        stock_price_summary=price_summary,
    )


# K线图
def build_stock_chart(
    chart_df: pd.DataFrame,
):
    if chart_df.empty:
        return None

    df = chart_df.tail(
        100
    ).copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[
            0.58,
            0.20,
            0.22,
        ],
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#e53935",
            increasing_fillcolor="#ffffff",
            decreasing_line_color="#16863a",
            decreasing_fillcolor="#16863a",
            name="K线",
        ),
        row=1,
        col=1,
    )

    # 均线
    ma_config = [
        ("ma5", "MA5", "#222222"),
        ("ma10", "MA10", "#d89b24"),
        ("ma20", "MA20", "#df5b7b"),
        ("ma30", "MA30", "#56ad5a"),
    ]

    for column, label, color in ma_config:
        if column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[column],
                    mode="lines",
                    line=dict(
                        width=1.4,
                        color=color,
                    ),
                    name=label,
                ),
                row=1,
                col=1,
            )

    # BOLL 默认显示
    boll_config = [
        (
            "boll_upper",
            "BOLL上轨",
            "#7e57c2",
        ),
        (
            "boll_mid",
            "BOLL中轨",
            "#607d8b",
        ),
        (
            "boll_lower",
            "BOLL下轨",
            "#7e57c2",
        ),
    ]

    for column, label, color in boll_config:
        if column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[column],
                    mode="lines",
                    line=dict(
                        width=1.1,
                        color=color,
                        dash="dot",
                    ),
                    name=label,
                ),
                row=1,
                col=1,
            )

    # 成交量
    if "volume" in df.columns:
        volume_colors = []

        for _, row_data in df.iterrows():
            if (
                row_data["close"]
                >= row_data["open"]
            ):
                volume_colors.append(
                    "#e53935"
                )
            else:
                volume_colors.append(
                    "#16863a"
                )

        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                marker_color=volume_colors,
                opacity=0.75,
                name="成交量",
            ),
            row=2,
            col=1,
        )

        if "volume_ma5" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[
                        "volume_ma5"
                    ],
                    line=dict(
                        color="#222222",
                        width=1.3,
                    ),
                    name="量MA5",
                ),
                row=2,
                col=1,
            )

        if "volume_ma10" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[
                        "volume_ma10"
                    ],
                    line=dict(
                        color="#d97706",
                        width=1.3,
                    ),
                    name="量MA10",
                ),
                row=2,
                col=1,
            )

    # KDJ
    kdj_config = [
        ("k", "K", "#222222"),
        ("d", "D", "#e59c23"),
        ("j", "J", "#d6547b"),
    ]

    for column, label, color in kdj_config:
        if column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[column],
                    mode="lines",
                    line=dict(
                        width=1.5,
                        color=color,
                    ),
                    name=label,
                ),
                row=3,
                col=1,
            )

    fig.add_hline(
        y=80,
        line_width=1,
        line_dash="dot",
        line_color="#aaaaaa",
        row=3,
        col=1,
    )

    fig.add_hline(
        y=20,
        line_width=1,
        line_dash="dot",
        line_color="#aaaaaa",
        row=3,
        col=1,
    )

    fig.update_layout(
        height=900,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=20,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(
                size=12
            ),
        ),
    )

    for row in [
        1,
        2,
        3,
    ]:
        fig.update_xaxes(
            showgrid=False,
            row=row,
            col=1,
        )

        fig.update_yaxes(
            gridcolor="#eeeeee",
            zeroline=False,
            row=row,
            col=1,
        )

    fig.update_yaxes(
        title_text="价格",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="成交量",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="KDJ",
        range=[
            -10,
            110,
        ],
        row=3,
        col=1,
    )

    return fig


# Sidebar
st.sidebar.title(
    "A股研究助手"
)

query = st.sidebar.text_input(
    "股票名称或代码",
    value="002226",
)

run_button = st.sidebar.button(
    "查询",
    use_container_width=True,
)


# 用户点击查询
if run_button:
    st.session_state[
        "active_query"
    ] = query


if (
    "active_query"
    not in st.session_state
):
    st.info(
        "请输入股票名称或代码，然后点击“查询”。"
    )
    st.stop()


active_query = st.session_state[
    "active_query"
]


# 股票识别
with st.spinner(
    "正在识别股票..."
):
    search_result = (
        cached_resolve_stock(
            active_query
        )
    )


if not search_result:
    st.error(
        "无法识别该股票名称或代码。"
    )
    st.stop()


stock_code = search_result.get(
    "stock_code"
)

company_name = search_result.get(
    "company_name"
)

market_symbol = search_result.get(
    "market_symbol"
)


# 公司信息
with st.spinner(
    "正在加载公司资料..."
):
    try:
        company_data = (
            cached_company_data(
                stock_code
            )
        )

    except Exception:
        company_data = {}


profile_industry = (
    _extract_industry(
        company_data
    )
)


# 先资金流
with st.spinner(
    "正在加载资金流..."
):
    try:
        fund_flow_data = (
            cached_fund_flow(
                stock_code
            )
        )

    except Exception as exc:
        fund_flow_data = {
            "warnings": [
                str(exc)
            ]
        }


# 再股价
with st.spinner(
    "正在加载行情与技术指标..."
):
    try:
        price_data = (
            cached_price_data(
                stock_code
            )
        )

    except Exception as exc:
        st.error(
            f"行情数据加载失败：{exc}"
        )
        st.stop()


price_summary = (
    price_data.get(
        "summary",
        {},
    )
)


# 顶部股票名称
st.markdown(
    f"""
<div class="stock-header">
    <div class="stock-name">
        {company_name}
    </div>
    <div class="stock-code">
        {stock_code}
    </div>
</div>
""",
    unsafe_allow_html=True,
)


latest_close = _number(
    price_summary.get(
        "latest_close"
    )
)

change_pct = _number(
    price_summary.get(
        "daily_change_pct"
    )
)


if (
    change_pct is not None
    and change_pct >= 0
):
    price_class = "price-up"
    change_class = "change-up"

else:
    price_class = "price-down"
    change_class = "change-down"


top_left, top_right = (
    st.columns(
        [
            1,
            2,
        ]
    )
)


with top_left:
    st.markdown(
        f"""
<div class="{price_class}">
    {latest_close if latest_close is not None else "--"}
</div>

<div class="{change_class}">
    {_pct(change_pct)}
</div>
""",
        unsafe_allow_html=True,
    )


turnover = (
    fund_flow_data
    .get(
        "turnover",
        {},
    )
    .get(
        "current_turnover_rate_pct"
    )
)

fund_latest = (
    fund_flow_data
    .get(
        "fund_flow",
        {},
    )
    .get(
        "latest",
        {},
    )
)

amount = fund_latest.get(
    "turnover_amount"
)

latest_high = price_summary.get(
    "latest_high"
)

latest_low = price_summary.get(
    "latest_low"
)

latest_open = price_summary.get(
    "latest_open"
)


with top_right:
    c1, c2, c3 = (
        st.columns(3)
    )

    with c1:
        st.markdown(
            f"""
<div class="info-row">
<span class="info-label">最高</span>
<span class="info-value">{latest_high or "--"}</span>
</div>

<div class="info-row">
<span class="info-label">最低</span>
<span class="info-value">{latest_low or "--"}</span>
</div>

<div class="info-row">
<span class="info-label">开盘</span>
<span class="info-value">{latest_open or "--"}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
<div class="info-row">
<span class="info-label">成交额</span>
<span class="info-value">{_money(amount)}</span>
</div>

<div class="info-row">
<span class="info-label">5日涨跌</span>
<span class="info-value">{_pct(price_summary.get("return_5d_pct"))}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
<div class="info-row">
<span class="info-label">换手率</span>
<span class="info-value">{_plain_pct(turnover)}</span>
</div>

<div class="info-row">
<span class="info-label">20日涨跌</span>
<span class="info-value">{_pct(price_summary.get("return_20d_pct"))}</span>
</div>
""",
            unsafe_allow_html=True,
        )


st.divider()


# K线周期
period = st.radio(
    "周期",
    options=[
        "日K",
        "周K",
        "月K",
    ],
    horizontal=True,
)


chart_df = get_chart_data(
    price_data.get(
        "history",
        [],
    ),
    period=period,
)


if not chart_df.empty:
    latest_chart = (
        chart_df.iloc[-1]
    )

    ma_text = []

    for period_value in [
        5,
        10,
        20,
        30,
    ]:
        column = (
            f"ma{period_value}"
        )

        if (
            column in latest_chart
            and pd.notna(
                latest_chart[
                    column
                ]
            )
        ):
            ma_text.append(
                f"MA{period_value}: "
                f"{latest_chart[column]:.2f}"
            )

    st.caption(
        "  |  ".join(
            ma_text
        )
    )

    stock_chart = (
        build_stock_chart(
            chart_df
        )
    )

    if stock_chart:
        st.plotly_chart(
            stock_chart,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
            },
        )

else:
    st.warning(
        "暂时无法生成技术图表。"
    )


# 技术指标摘要
technical = price_data.get(
    "technical",
    {},
)

volume_data = technical.get(
    "volume",
    {},
)

kdj_data = technical.get(
    "kdj",
    {},
)

boll_data = technical.get(
    "bollinger",
    {},
)


st.markdown(
    '<div class="section-title">技术指标摘要</div>',
    unsafe_allow_html=True,
)


t1, t2, t3 = (
    st.columns(3)
)


t1.metric(
    "成交量",
    (
        f"{volume_data.get('latest_volume_wan_shou')} 万手"
        if volume_data.get(
            "latest_volume_wan_shou"
        )
        is not None
        else "--"
    ),
)


t2.metric(
    "KDJ",
    (
        f"K {kdj_data.get('k', '--')} / "
        f"D {kdj_data.get('d', '--')} / "
        f"J {kdj_data.get('j', '--')}"
    ),
)


t3.metric(
    "BOLL 中轨",
    boll_data.get(
        "middle",
        "--",
    ),
)


st.write(
    "KDJ趋势："
    f"K {kdj_data.get('k_trend', '--')}，"
    f"D {kdj_data.get('d_trend', '--')}，"
    f"J {kdj_data.get('j_trend', '--')}；"
    f"{kdj_data.get('cross', '')}"
)


st.write(
    "布林轨道："
    f"上轨 {boll_data.get('upper', '--')}，"
    f"中轨 {boll_data.get('middle', '--')}，"
    f"下轨 {boll_data.get('lower', '--')}；"
    f"中轨趋势 {boll_data.get('middle_trend', '--')}，"
    f"%B {boll_data.get('percent_b', '--')}%"
)


# 深度数据加载提示
st.divider()

st.markdown(
    '<div class="section-title">深度数据</div>',
    unsafe_allow_html=True,
)

st.caption(
    "行情和技术指标已经可以查看。"
    "下面的基本面、板块和公告正在继续加载。"
)


financial_status = st.status(
    "⏳ 基本面数据加载中...",
    expanded=False,
)

sector_status = st.status(
    "⏳ 板块与主营数据加载中...",
    expanded=False,
)

news_status = st.status(
    "⏳ 最新公告加载中...",
    expanded=False,
)


financial_data = {}
sector_data = {}
news_data = []
errors = []


# 并行加载
with ThreadPoolExecutor(
    max_workers=3
) as executor:

    future_map = {
        executor.submit(
            cached_financial_data,
            stock_code,
            market_symbol,
        ): "financial",

        executor.submit(
            cached_sector,
            stock_code,
            profile_industry,
            price_summary,
        ): "sector",

        executor.submit(
            cached_news,
            stock_code,
            company_name,
        ): "news",
    }

    for future in (
        as_completed(
            future_map
        )
    ):
        task_name = (
            future_map[
                future
            ]
        )

        try:
            result = (
                future.result()
            )

            if (
                task_name
                == "financial"
            ):
                financial_data = (
                    result
                )

                financial_status.update(
                    label=(
                        "✅ 基本面数据加载完成"
                    ),
                    state="complete",
                    expanded=False,
                )

            elif (
                task_name
                == "sector"
            ):
                sector_data = (
                    result
                )

                sector_status.update(
                    label=(
                        "✅ 板块与主营数据加载完成"
                    ),
                    state="complete",
                    expanded=False,
                )

            elif (
                task_name
                == "news"
            ):
                news_data = (
                    result
                )

                news_status.update(
                    label=(
                        "✅ 最新公告加载完成"
                    ),
                    state="complete",
                    expanded=False,
                )

        except Exception as exc:
            errors.append(
                f"{task_name}: {exc}"
            )

            if (
                task_name
                == "financial"
            ):
                financial_status.update(
                    label=(
                        "⚠️ 基本面数据暂时不可用"
                    ),
                    state="error",
                    expanded=False,
                )

            elif (
                task_name
                == "sector"
            ):
                sector_status.update(
                    label=(
                        "⚠️ 板块数据暂时不可用"
                    ),
                    state="error",
                    expanded=False,
                )

            elif (
                task_name
                == "news"
            ):
                news_status.update(
                    label=(
                        "⚠️ 公告数据暂时不可用"
                    ),
                    state="error",
                    expanded=False,
                )


# 深度数据 tabs
tab1, tab2, tab3 = st.tabs(
    [
        "基本面",
        "板块与主营",
        "公告",
    ]
)


# 基本面
with tab1:
    valuation = (
        financial_data.get(
            "valuation",
            {},
        )
    )

    pe = valuation.get(
        "pe"
    )

    roe = _find_value(
        financial_data,
        [
            "ROE_WEIGHT",
            "加权净资产收益率",
            "净资产收益率",
        ],
    )

    debt_ratio = (
        _find_value(
            financial_data,
            [
                "资产负债率",
                "DEBT_ASSET_RATIO",
            ],
        )
    )

    revenue = (
        _find_value(
            financial_data,
            [
                "OPERATE_INCOME",
                "营业收入",
            ],
        )
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

    if (
        _number(
            revenue
        )
        is not None
        and _number(
            revenue
        )
        != 0
        and _number(
            operating_profit
        )
        is not None
    ):
        operating_margin = (
            _number(
                operating_profit
            )
            / _number(
                revenue
            )
            * 100
        )

    f1, f2, f3, f4 = (
        st.columns(4)
    )

    f1.metric(
        "市盈率 TTM",
        (
            f"{pe:.2f}×"
            if pe is not None
            else "--"
        ),
    )

    f2.metric(
        "营业利润率",
        _plain_pct(
            operating_margin
        ),
    )

    f3.metric(
        "资产负债率",
        _plain_pct(
            debt_ratio
        ),
    )

    f4.metric(
        "ROE",
        _plain_pct(
            roe
        ),
    )


# 板块与主营
with tab2:
    snapshot = (
        sector_data.get(
            "industry_snapshot",
            {},
        )
    )

    relative = (
        sector_data.get(
            "relative_performance",
            {},
        )
    )

    s1, s2, s3 = (
        st.columns(3)
    )

    s1.metric(
        "所属板块",
        snapshot.get(
            "industry_name",
            "--",
        ),
    )

    s2.metric(
        "板块涨跌",
        _pct(
            snapshot.get(
                "change_pct"
            )
        ),
    )

    rank = snapshot.get(
        "fund_flow_rank"
    )

    total = snapshot.get(
        "industry_count"
    )

    s3.metric(
        "板块资金排名",
        (
            f"{rank}/{total}"
            if (
                rank is not None
                and total is not None
            )
            else "--"
        ),
    )

    if relative:
        st.markdown(
            "#### 相对板块表现"
        )

        relative_df = (
            pd.DataFrame(
                relative
            ).T
        )

        relative_df = (
            relative_df.rename(
                columns={
                    "stock_return_pct": "个股涨跌(%)",
                    "sector_return_pct": "板块涨跌(%)",
                    "excess_return_pct": "相对表现(%)",
                }
            )
        )

        st.dataframe(
            relative_df,
            use_container_width=True,
        )

    products = (
        sector_data
        .get(
            "business_composition",
            {},
        )
        .get(
            "product",
            [],
        )
    )

    if products:
        st.markdown(
            "#### 主营构成"
        )

        product_df = (
            pd.DataFrame(
                products
            )
        )

        product_df = (
            product_df.rename(
                columns={
                    "name": "业务名称",
                    "revenue": "营业收入",
                    "revenue_share_pct": "收入占比(%)",
                    "gross_margin_pct": "毛利率(%)",
                }
            )
        )

        if (
            "营业收入"
            in product_df.columns
        ):
            product_df[
                "营业收入"
            ] = (
                product_df[
                    "营业收入"
                ].apply(
                    _money
                )
            )

        st.dataframe(
            product_df,
            use_container_width=True,
            hide_index=True,
        )


# 公告
with tab3:
    if news_data:
        news_df = (
            pd.DataFrame(
                news_data
            )
        )

        news_df = (
            news_df.rename(
                columns={
                    "published_at": "发布日期",
                    "freshness": "时效性",
                    "title": "公告标题",
                    "source": "来源",
                    "url": "链接",
                }
            )
        )

        columns = [
            column
            for column
            in [
                "发布日期",
                "时效性",
                "公告标题",
                "来源",
                "链接",
            ]
            if column
            in news_df.columns
        ]

        st.dataframe(
            news_df[
                columns
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "暂未获取到公告。"
        )


# AI研究摘要
full_state = {
    "user_query": active_query,
    "stock_code": stock_code,
    "company_name": company_name,
    "market_symbol": market_symbol,
    "search_result": search_result,
    "company_data": company_data,
    "price_data": price_data,
    "fund_flow_data": fund_flow_data,
    "financial_data": financial_data,
    "sector_data": sector_data,
    "news_data": news_data,
    "errors": errors,
}


st.divider()

ai_status = st.status(
    "⏳ AI研究摘要生成中...",
    expanded=False,
)


try:
    ai_result = (
        report_agent(
            full_state
        )
    )

    final_report = (
        ai_result.get(
            "final_report",
            "",
        )
    )

    ai_status.update(
        label=(
            "✅ AI研究摘要生成完成"
        ),
        state="complete",
        expanded=False,
    )

except Exception as exc:
    final_report = ""

    ai_status.update(
        label=(
            "⚠️ AI研究摘要暂时不可用"
        ),
        state="error",
        expanded=False,
    )

    errors.append(
        f"AI: {exc}"
    )


st.markdown(
    '<div class="section-title">AI研究摘要</div>',
    unsafe_allow_html=True,
)


if final_report:
    st.markdown(
        final_report
    )

else:
    st.info(
        "AI摘要暂时不可用，"
        "上方结构化数据仍可正常查看。"
    )


if errors:
    st.markdown(
        '<div class="section-title">数据提示</div>',
        unsafe_allow_html=True,
    )

    for error in errors:
        st.warning(
            error
        )