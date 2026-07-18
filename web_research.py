"""
web_research.py
---------------
Live web research helpers for diagnosis augmentation.

The app uses this module to fetch current support articles and tutorial links
from the public web after a diagnosis is produced. The goal is to combine the
trained model + knowledge base with up-to-date vendor guidance.
"""

from __future__ import annotations

import html
import json
import os
import re
from functools import lru_cache
from urllib.error import URLError, HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse, unquote
from urllib.request import Request, urlopen

DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/?q={query}"
REQUEST_TIMEOUT = float(os.environ.get("WEB_RESEARCH_TIMEOUT", "6"))

OFFICIAL_DOMAINS = {
    "support.hp.com",
    "support.microsoft.com",
    "support.google.com",
    "support.lenovo.com",
    "support.dell.com",
    "support.acer.com",
    "support.asus.com",
    "www.nvidia.com",
    "www.amd.com",
    "downloadcenter.intel.com",
    "support.apple.com",
}


def _normalize_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _decode_duckduckgo_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url

    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and "uddg=" in parsed.query:
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _extract_results(html_text: str, limit: int = 5):
    links = re.findall(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html_text,
        flags=re.I | re.S,
    )
    snippets = re.findall(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        html_text,
        flags=re.I | re.S,
    )

    results = []
    for idx, (href, title_html) in enumerate(links[:limit]):
        title = re.sub(r"<.*?>", "", html.unescape(title_html)).strip()
        snippet_html = snippets[idx] if idx < len(snippets) else ""
        snippet = re.sub(r"<.*?>", "", html.unescape(snippet_html)).strip()
        url = _decode_duckduckgo_url(html.unescape(href))
        domain = _normalize_domain(url)
        if not title or not url:
            continue
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "domain": domain,
                "official": domain in {d.replace("www.", "") for d in OFFICIAL_DOMAINS},
            }
        )
    return results


@lru_cache(maxsize=64)
def search_web(query: str, limit: int = 5):
    """
    Perform a live web search using DuckDuckGo's lightweight HTML results page.
    Returns a list of result dictionaries with title, url, snippet, domain.
    """
    query = (query or "").strip()
    if not query:
        return []

    url = DUCKDUCKGO_HTML.format(query=quote_plus(query))
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            )
        },
    )

    try:
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except (URLError, HTTPError, TimeoutError, ValueError):
        return []

    return _extract_results(body, limit=limit)


def _category_queries(category: str, user_message: str, image_description: str | None):
    msg = " ".join(part for part in [image_description or "", user_message or ""] if part).strip()
    queries = []
    youtube_queries = []

    if category == "boot_recovery_issue":
        queries = [
            f'site:support.hp.com "HP Sure Recover" "{msg}"',
            f'site:support.hp.com "operating system not found" "{msg}"',
            f'site:support.microsoft.com "operating system not found" "{msg}"',
            f'"{msg}" boot recovery Windows',
        ]
        youtube_queries = [
            f'"{msg}" HP Sure Recover tutorial',
            f'"{msg}" Windows recovery tutorial',
        ]
    elif category == "bios_firmware_issue":
        queries = [
            f'site:support.hp.com BIOS UEFI "{msg}"',
            f'site:support.microsoft.com UEFI boot order "{msg}"',
            f'"{msg}" BIOS setup support',
        ]
        youtube_queries = [
            f'"{msg}" BIOS setup tutorial',
            f'"{msg}" UEFI boot order tutorial',
        ]
    elif category == "driver_issue":
        queries = [
            f'site:support.microsoft.com update driver "{msg}"',
            f'site:support.hp.com reinstall driver "{msg}"',
            f'"{msg}" device driver fix',
        ]
        youtube_queries = [
            f'"{msg}" driver update tutorial',
        ]
    elif category == "application_issue":
        queries = [
            f'site:support.microsoft.com app not responding "{msg}"',
            f'site:support.google.com application crash "{msg}"',
            f'"{msg}" repair application',
        ]
        youtube_queries = [
            f'"{msg}" app repair tutorial',
        ]
    else:
        queries = [
            f'"{msg}"',
            f'"{msg}" official support',
            f'"{msg}" troubleshooting',
        ]
        youtube_queries = [
            f'"{msg}" tutorial',
        ]

    return [q for q in queries if q.strip()], [q for q in youtube_queries if q.strip()]


def research_issue(category: str | None, user_message: str, image_description: str | None = None):
    """
    Collect live web references relevant to the current issue.
    Returns a dict with official_sources, general_sources, youtube_sources.
    """
    category = category or ""
    queries, youtube_queries = _category_queries(category, user_message, image_description)

    collected = []
    seen_urls = set()

    for query in queries:
        for item in search_web(query, limit=5):
            url = item["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            item["query"] = query
            collected.append(item)

    official_sources = [item for item in collected if item["official"]][:4]
    general_sources = [item for item in collected if not item["official"]][:4]

    youtube_sources = []
    for query in youtube_queries:
        for item in search_web(query, limit=3):
            if "youtube.com" not in item["domain"] and "youtu.be" not in item["domain"]:
                continue
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            item["query"] = query
            youtube_sources.append(item)
            if len(youtube_sources) >= 2:
                break
        if len(youtube_sources) >= 2:
            break

    summary_parts = []
    if official_sources:
        summary_parts.append(
            "Official guidance found from " + ", ".join(sorted({s["domain"] for s in official_sources if s.get("domain")})) + "."
        )
    if youtube_sources:
        summary_parts.append("A couple of video walkthroughs are available for follow-up.")

    return {
        "queries": queries,
        "official_sources": official_sources,
        "general_sources": general_sources,
        "youtube_sources": youtube_sources,
        "summary": " ".join(summary_parts).strip(),
    }
