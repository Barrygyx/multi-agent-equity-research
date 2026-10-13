import time
from datetime import datetime, timedelta
from typing import Any

import akshare as ak
import pandas as pd


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


def _clean_value(value: Any):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return None

    return text


def _parse_date(value: Any):
    if value is None:
        return None

    try:
        parsed = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(parsed):
            return None

        return parsed

    except Exception:
        return None


def _get_freshness(published_at: Any):
    parsed_date = _parse_date(
        published_at
    )

    if parsed_date is None:
        return {
            "age_days": None,
            "freshness": "未知",
        }

    age_days = (
        datetime.now().date()
        - parsed_date.date()
    ).days

    age_days = max(
        age_days,
        0,
    )

    if age_days <= 7:
        freshness = "最新"

    elif age_days <= 30:
        freshness = "近期"

    elif age_days <= 90:
        freshness = "可参考"

    else:
        freshness = "较旧"

    return {
        "age_days": age_days,
        "freshness": freshness,
    }


def _format_date(value: Any):
    parsed = _parse_date(
        value
    )

    if parsed is None:
        return _clean_value(
            value
        )

    return parsed.strftime(
        "%Y-%m-%d"
    )


def _get_cninfo_announcements(
    stock_code: str,
    days: int,
):
    end_date = datetime.now()

    start_date = (
        end_date
        - timedelta(days=days)
    )

    df = (
        ak.stock_zh_a_disclosure_report_cninfo(
            symbol=stock_code,
            market="沪深京",
            keyword="",
            category="",
            start_date=start_date.strftime(
                "%Y%m%d"
            ),
            end_date=end_date.strftime(
                "%Y%m%d"
            ),
        )
    )

    if df is None or df.empty:
        return []

    items = []

    for _, row in df.iterrows():
        title = _clean_value(
            row.get("公告标题")
        )

        if not title:
            continue

        published_at = _format_date(
            row.get("公告时间")
        )

        freshness = _get_freshness(
            published_at
        )

        items.append(
            {
                "type": "announcement",
                "title": title,
                "summary": None,
                "published_at": published_at,
                "age_days": freshness[
                    "age_days"
                ],
                "freshness": freshness[
                    "freshness"
                ],
                "source": "巨潮资讯",
                "url": _clean_value(
                    row.get("公告链接")
                ),
            }
        )

    return items


def _get_eastmoney_news(
    stock_code: str,
):
    df = ak.stock_news_em(
        symbol=stock_code
    )

    if df is None or df.empty:
        return []

    items = []

    for _, row in df.iterrows():
        title = None

        for column in [
            "新闻标题",
            "标题",
        ]:
            if column in df.columns:
                title = _clean_value(
                    row.get(column)
                )

                if title:
                    break

        if not title:
            continue

        published_at = None

        for column in [
            "发布时间",
            "时间",
            "新闻时间",
        ]:
            if column in df.columns:
                published_at = (
                    _format_date(
                        row.get(column)
                    )
                )

                if published_at:
                    break

        summary = None

        for column in [
            "新闻内容",
            "内容",
            "摘要",
        ]:
            if column in df.columns:
                summary = _clean_value(
                    row.get(column)
                )

                if summary:
                    break

        source = None

        for column in [
            "文章来源",
            "媒体名称",
            "来源",
        ]:
            if column in df.columns:
                source = _clean_value(
                    row.get(column)
                )

                if source:
                    break

        url = None

        for column in [
            "新闻链接",
            "链接",
        ]:
            if column in df.columns:
                url = _clean_value(
                    row.get(column)
                )

                if url:
                    break

        freshness = _get_freshness(
            published_at
        )

        items.append(
            {
                "type": "media_news",
                "title": title,
                "summary": summary,
                "published_at": published_at,
                "age_days": freshness[
                    "age_days"
                ],
                "freshness": freshness[
                    "freshness"
                ],
                "source": (
                    source
                    or "东方财富"
                ),
                "url": url,
            }
        )

    return items


def _deduplicate(
    items: list[dict[str, Any]],
):
    result = []
    seen = set()

    for item in items:
        title = (
            item.get("title")
            or ""
        ).strip()

        published_at = (
            item.get(
                "published_at"
            )
            or ""
        )

        key = (
            title,
            published_at,
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def _sort_items(
    items: list[dict[str, Any]],
):
    def sort_key(item):
        parsed = _parse_date(
            item.get(
                "published_at"
            )
        )

        if parsed is None:
            return pd.Timestamp.min

        return parsed

    return sorted(
        items,
        key=sort_key,
        reverse=True,
    )


def _ensure_freshness(
    items: list[dict[str, Any]],
):
    result = []

    for item in items:
        item = dict(item)

        freshness = (
            _get_freshness(
                item.get(
                    "published_at"
                )
            )
        )

        item["age_days"] = (
            freshness[
                "age_days"
            ]
        )

        item["freshness"] = (
            freshness[
                "freshness"
            ]
        )

        result.append(item)

    return result


def get_stock_news(
    stock_code: str,
    company_name: str | None = None,
    limit: int = 10,
    days: int = 120,
    include_media: bool = False,
):
    stock_code = str(
        stock_code
    ).zfill(6)

    cache_key = (
        stock_code,
        limit,
        days,
        include_media,
    )

    cached = _cache_get(
        cache_key
    )

    if cached is not None:
        return cached

    items = []

    try:
        announcements = (
            _get_cninfo_announcements(
                stock_code=stock_code,
                days=days,
            )
        )

        items.extend(
            announcements
        )

    except Exception:
        pass

    if include_media:
        try:
            media_news = (
                _get_eastmoney_news(
                    stock_code
                )
            )

            items.extend(
                media_news
            )

        except Exception:
            pass

    items = _deduplicate(
        items
    )

    items = _sort_items(
        items
    )

    items = _ensure_freshness(
        items
    )

    items = items[:limit]

    _cache_set(
        cache_key,
        items,
    )

    return items