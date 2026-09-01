from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


class HtmlSource(Protocol):
    location: Path | str
    source_type: str


def is_html_source(source: HtmlSource) -> bool:
    if source.source_type != "url":
        return False
    suffix = Path(urlparse(str(source.location)).path).suffix.lower()
    return suffix not in {
        ".csv",
        ".doc",
        ".docx",
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".tiff",
        ".xls",
        ".xlsx",
    }


def parse_html_sections(
    html: bytes | str,
    *,
    source_url: str,
    fallback_title: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract main-page text into chunks that retain the real HTML section hierarchy."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required for HTML ingestion") from exc

    soup = BeautifulSoup(html, "html.parser")
    newsroom_title = soup.select_one(
        "#wd_printable_content .wd_title, .wd_newsfeed_releases-detail .wd_title"
    )
    newsroom_body = soup.select_one(
        "#wd_printable_content .wd_news_body, .wd_newsfeed_releases-detail .wd_news_body"
    )
    if newsroom_title is not None and newsroom_body is not None:
        container = newsroom_body
        page_title = _clean_html_text(newsroom_title.get_text(" ", strip=True))
        root_heading = None
        content_started = True
    else:
        container = (
            soup.select_one("main article")
            or soup.find("article")
            or soup.find("main")
            or soup.select_one('[role="main"]')
            or soup.body
        )
        if container is None:
            raise RuntimeError("HTML page has no body or main-content container")
        root_heading = container.find("h1")
        page_title = (
            _clean_html_text(root_heading.get_text(" ", strip=True))
            if root_heading is not None
            else _html_page_title(soup, fallback_title)
        )
        content_started = root_heading is None

    if not page_title:
        page_title = fallback_title

    for unwanted in container.select(
        "script, style, noscript, svg, form, nav, header, footer, aside, "
        '[role="navigation"], [role="banner"], [role="contentinfo"], '
        '[aria-hidden="true"], .breadcrumb, .breadcrumbs, .cookie-banner'
    ):
        unwanted.decompose()

    heading_levels: dict[int, str] = {1: page_title}
    current_blocks: list[str] = []
    sections: list[dict[str, Any]] = []

    def current_headings() -> tuple[str, ...]:
        return tuple(heading_levels[level] for level in sorted(heading_levels))

    def flush_section() -> None:
        if not current_blocks:
            return
        text = "\n\n".join(current_blocks).strip()
        if not text:
            return
        headings = current_headings()
        sections.append(
            {
                "text": text,
                "meta": {
                    "headings": list(headings),
                    "section_title": headings[-1] if headings else page_title,
                    "source_url": source_url,
                    "section_index": len(sections),
                },
            }
        )
        current_blocks.clear()

    def start_section(level: int, title: str) -> None:
        flush_section()
        for existing_level in [value for value in heading_levels if value >= level]:
            heading_levels.pop(existing_level)
        heading_levels[level] = title

    elements = container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]
    )
    for element in elements:
        if root_heading is not None and element is root_heading:
            content_started = True
            heading_levels = {1: page_title}
            continue
        if not content_started:
            continue

        name = element.name.lower()
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            title = _clean_html_text(element.get_text(" ", strip=True))
            if title and title != page_title:
                start_section(max(2, int(name[1])), title)
            continue
        if name in {"p", "li"} and element.find_parent("table") is not None:
            continue
        if name == "p" and element.find_parent("li") is not None:
            continue

        text = (
            _html_table_text(element)
            if name == "table"
            else _clean_html_text(element.get_text(" ", strip=True))
        )
        if not text:
            continue
        if name == "p" and _is_bold_only_paragraph(element, text):
            start_section(2, text)
            continue
        if not current_blocks or current_blocks[-1] != text:
            current_blocks.append(text)

    flush_section()
    if not sections:
        raise RuntimeError("No main-page text sections were found in the HTML response")

    document = {
        "schema_name": "SectionAwareHtmlDocument",
        "source_url": source_url,
        "title": page_title,
        "sections": [
            {"headings": section["meta"]["headings"], "text": section["text"]}
            for section in sections
        ],
    }
    return document, sections


def parse_markdown_sections(
    markdown: bytes | str,
    *,
    source_url: str,
    fallback_title: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse reader-fallback Markdown into explicit page sections."""
    text = markdown.decode("utf-8", errors="replace") if isinstance(markdown, bytes) else markdown
    lines = text.splitlines()
    preamble_title = next(
        (line.removeprefix("Title:").strip() for line in lines if line.startswith("Title:")),
        fallback_title,
    )
    content_marker = next(
        (index for index, line in enumerate(lines) if line.strip() == "Markdown Content:"),
        -1,
    )
    first_h1 = next(
        (
            index
            for index, line in enumerate(lines[content_marker + 1 :], start=content_marker + 1)
            if re.match(r"^#\s+\S", line)
        ),
        None,
    )
    headingless = first_h1 is None
    page_title = (
        preamble_title
        if headingless
        else _clean_markdown_text(lines[first_h1][2:]) or preamble_title
    )
    content_start = content_marker + 1 if headingless else first_h1 + 1
    heading_levels: dict[int, str] = {1: page_title}
    blocks: list[str] = []
    sections: list[dict[str, Any]] = []
    footer_headings = {
        "cookie list",
        "for california residents:",
        "looking for bmo u.s.?",
        "your california privacy choices",
    }

    def flush_section() -> None:
        content = "\n".join(blocks).strip()
        if not content:
            return
        headings = [heading_levels[level] for level in sorted(heading_levels)]
        sections.append(
            {
                "text": content,
                "meta": {
                    "headings": headings,
                    "section_title": headings[-1],
                    "source_url": source_url,
                    "section_index": len(sections),
                },
            }
        )
        blocks.clear()

    for line in lines[content_start:]:
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            level = len(heading_match.group(1))
            title = _clean_markdown_text(heading_match.group(2))
            if level == 2 and title.casefold() in footer_headings:
                break
            flush_section()
            level = max(2, level)
            for existing_level in [value for value in heading_levels if value >= level]:
                heading_levels.pop(existing_level)
            heading_levels[level] = title
            continue

        cleaned = _clean_markdown_text(line)
        inferred_level = _plain_markdown_heading_level(cleaned) if headingless else None
        if inferred_level is not None:
            flush_section()
            for existing_level in [
                value for value in heading_levels if value >= inferred_level
            ]:
                heading_levels.pop(existing_level)
            heading_levels[inferred_level] = cleaned
            continue
        if cleaned:
            blocks.append(cleaned)
        elif blocks and blocks[-1] != "":
            blocks.append("")

    flush_section()
    if not sections:
        raise RuntimeError("No main-page text sections were found in the reader response")

    document = {
        "schema_name": "SectionAwareMarkdownDocument",
        "source_url": source_url,
        "title": page_title,
        "sections": [
            {"headings": section["meta"]["headings"], "text": section["text"]}
            for section in sections
        ],
    }
    return document, sections


def _clean_markdown_text(value: str) -> str:
    value = re.sub(r"!\[[^]]*]\([^)]*\)", "", value)
    value = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*_`]+", "", value)
    return _clean_html_text(value)


def _plain_markdown_heading_level(value: str) -> int | None:
    if re.match(r"^[a-z]\.[ \t]+[A-Z]", value):
        return 3
    letters = [character for character in value if character.isalpha()]
    if 3 <= len(value) <= 140 and letters and value == value.upper():
        return 2
    return None


def _html_page_title(soup: Any, fallback_title: str) -> str:
    if soup.title is None:
        return fallback_title
    title = _clean_html_text(soup.title.get_text(" ", strip=True))
    return title or fallback_title


def _clean_html_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_bold_only_paragraph(element: Any, text: str) -> bool:
    bold = element.find(["b", "strong"], recursive=False)
    return bold is not None and _clean_html_text(bold.get_text(" ", strip=True)) == text


def _html_table_text(table: Any) -> str:
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            _clean_html_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)
