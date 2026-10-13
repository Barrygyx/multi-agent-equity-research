from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from agents import (
    company_agent,
    financial_agent,
    fund_flow_agent,
    news_agent,
    price_agent,
    report_agent,
    sector_agent,
)

from state import ResearchState


def build_graph():
    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "company_agent",
        company_agent,
    )

    builder.add_node(
        "fund_flow_agent",
        fund_flow_agent,
    )

    builder.add_node(
        "financial_agent",
        financial_agent,
    )

    builder.add_node(
        "price_agent",
        price_agent,
    )

    builder.add_node(
        "news_agent",
        news_agent,
    )

    builder.add_node(
        "sector_agent",
        sector_agent,
    )

    builder.add_node(
        "report_agent",
        report_agent,
    )

    builder.add_edge(
        START,
        "company_agent",
    )

    builder.add_edge(
        "company_agent",
        "fund_flow_agent",
    )

    builder.add_edge(
        "fund_flow_agent",
        "financial_agent",
    )

    builder.add_edge(
        "fund_flow_agent",
        "price_agent",
    )

    builder.add_edge(
        "fund_flow_agent",
        "news_agent",
    )

    builder.add_edge(
        "price_agent",
        "sector_agent",
    )

    builder.add_edge(
        [
            "financial_agent",
            "news_agent",
            "sector_agent",
        ],
        "report_agent",
    )

    builder.add_edge(
        "report_agent",
        END,
    )

    return builder.compile()


research_graph = build_graph()


def run_research(
    query: str,
):
    initial_state = {
        "user_query": query,
        "errors": [],
    }

    return research_graph.invoke(
        initial_state
    )