"""GitHub investigation tools.

fetch_github_file  — read source code around a specific line
search_github_callers — find callers of a function via GitHub code search
"""

from __future__ import annotations

import os
import re

import httpx

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}

# ── Tool implementations ──────────────────────────────────────────────────────

async def fetch_github_file(url: str, center_line: int, context_lines: int = 30) -> dict:
    """Fetch source code around a specific line from a GitHub file URL.

    Accepts either a GitHub blob URL
    (https://github.com/owner/repo/blob/ref/path/file.py)
    or a raw.githubusercontent.com URL.
    Returns the relevant lines with line numbers and the full file path.
    """
    raw_url = _to_raw_url(url)
    if raw_url is None:
        return {"error": f"cannot parse GitHub URL: {url!r}"}

    async with httpx.AsyncClient(timeout=10, headers=_HEADERS) as client:
        r = await client.get(raw_url)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code} fetching {raw_url}"}

    lines = r.text.splitlines()
    total = len(lines)
    start = max(0, center_line - 1 - context_lines)
    end   = min(total, center_line + context_lines)

    numbered = "\n".join(
        f"{i + 1:>5}  {'→ ' if i + 1 == center_line else '  '}{line}"
        for i, line in enumerate(lines[start:end], start=start)
    )
    return {
        "url":          url,
        "center_line":  center_line,
        "start_line":   start + 1,
        "end_line":     end,
        "total_lines":  total,
        "content":      numbered,
    }


async def search_github_callers(repo: str, function_name: str) -> dict:
    """Search a GitHub repo for callers of a function using the code search API.

    repo should be in 'owner/repo' or full URL form.
    Returns up to 10 matching file paths and line excerpts.
    """
    slug = _repo_slug(repo)
    if slug is None:
        return {"error": f"cannot parse repo: {repo!r}"}

    query = f"{function_name} repo:{slug}"
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
        r = await client.get(
            "https://api.github.com/search/code",
            params={"q": query, "per_page": 10},
        )
        if r.status_code == 403:
            return {"error": "GitHub search rate-limited or token missing"}
        if r.status_code != 200:
            return {"error": f"GitHub search returned HTTP {r.status_code}"}

    items = r.json().get("items", [])
    return {
        "function":  function_name,
        "repo":      slug,
        "matches":   [
            {"path": item["path"], "url": item["html_url"]}
            for item in items
        ],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_raw_url(url: str) -> str | None:
    # Already a raw URL
    if "raw.githubusercontent.com" in url:
        return url
    # GitHub blob URL: https://github.com/owner/repo/blob/ref/path
    m = re.match(r"https?://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.+)", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return None


def _repo_slug(repo: str) -> str | None:
    if re.match(r"^[\w.-]+/[\w.-]+$", repo):
        return repo
    m = re.match(r"https?://github\.com/([\w.-]+/[\w.-]+)", repo)
    return m.group(1) if m else None


# ── Claude tool definitions ───────────────────────────────────────────────────

DEFINITIONS = [
    {
        "name": "fetch_github_file",
        "description": (
            "Fetch source code from a GitHub file URL, centred on a specific line. "
            "Use this to read the code at a helixSource permalink from an error log. "
            "Returns the file content with line numbers, ±context_lines around center_line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "GitHub blob URL or raw.githubusercontent.com URL",
                },
                "center_line": {
                    "type": "integer",
                    "description": "Line number to centre the excerpt on (1-indexed)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context to include above and below center_line (default 30)",
                },
            },
            "required": ["url", "center_line"],
        },
    },
    {
        "name": "search_github_callers",
        "description": (
            "Search a GitHub repo for callers of a function using the GitHub code search API. "
            "Use this to follow the call chain one level up from an error location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in 'owner/repo' form or full GitHub URL",
                },
                "function_name": {
                    "type": "string",
                    "description": "Function name to search for",
                },
            },
            "required": ["repo", "function_name"],
        },
    },
]

HANDLERS = {
    "fetch_github_file":    fetch_github_file,
    "search_github_callers": search_github_callers,
}
