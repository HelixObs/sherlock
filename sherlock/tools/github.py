"""GitHub investigation tools.

fetch_github_file  — read source code around a specific line
search_github_callers — find callers of a function via GitHub code search
"""

from __future__ import annotations

import logging
import os
import re

import httpx

log = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ── Tool implementations ──────────────────────────────────────────────────────

async def fetch_github_file(url: str, center_line: int, context_lines: int = 30,
                            _token: str = "") -> dict:
    """Fetch source code around a specific line from a GitHub file URL.

    Accepts either a GitHub blob URL
    (https://github.com/owner/repo/blob/ref/path/file.py)
    or a raw.githubusercontent.com URL.
    Returns the relevant lines with line numbers and the full file path.
    """
    raw_url = _to_raw_url(url)
    if raw_url is None:
        return {"error": f"cannot parse GitHub URL: {url!r}"}

    # Operator token takes precedence over the service-level env token.
    token = _token or GITHUB_TOKEN
    log.info("fetch_github_file %s token=%s", raw_url, "yes" if token else "NO TOKEN")
    headers = {**_HEADERS, **({"Authorization": f"Bearer {token}"} if token else {})}

    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        r = await client.get(raw_url)
        if r.status_code != 200:
            return {
                "error": f"HTTP {r.status_code} — could not fetch source file",
                "github_url": url,
                "raw_url": raw_url,
                "hint": "Show the operator the github_url as a markdown link so they can verify access.",
            }

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


async def fetch_github_blame(url: str, center_line: int, context_lines: int = 30,
                             _token: str = "") -> dict:
    """Return git blame for lines around center_line using GitHub GraphQL.

    Each blame range shows: line span, author, authored date, short commit SHA,
    and first line of the commit message — so the agent can tell whether a
    recent commit introduced the failure.
    """
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", url)
    if not m:
        return {"error": f"cannot parse GitHub blob URL: {url!r}"}
    owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)

    token = _token or GITHUB_TOKEN
    if not token:
        return {"error": "GitHub token required for blame (private repo or GraphQL)"}

    query = """
query($owner: String!, $name: String!, $ref: String!, $path: String!) {
  repository(owner: $owner, name: $name) {
    ref(qualifiedName: $ref) {
      target {
        ... on Commit {
          blame(path: $path) {
            ranges {
              startingLine
              endingLine
              commit {
                abbreviatedOid
                oid
                message
                authoredDate
                author { name }
              }
            }
          }
        }
      }
    }
  }
}
"""
    headers = {
        **_HEADERS,
        "Authorization": f"Bearer {token}",
    }
    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        r = await client.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": {"owner": owner, "name": repo, "ref": ref, "path": path}},
        )
        if r.status_code != 200:
            return {"error": f"GraphQL HTTP {r.status_code}"}
        body = r.json()

    if "errors" in body:
        return {"error": body["errors"][0].get("message", str(body["errors"]))}

    try:
        ranges = (
            body["data"]["repository"]["ref"]["target"]["blame"]["ranges"]
        )
    except (KeyError, TypeError):
        return {"error": "unexpected GraphQL response shape"}

    lo = max(1, center_line - context_lines)
    hi = center_line + context_lines
    relevant = [
        r for r in ranges
        if r["endingLine"] >= lo and r["startingLine"] <= hi
    ]

    blame_lines = []
    for r in relevant:
        c = r["commit"]
        msg_first_line = c["message"].splitlines()[0][:80]
        blame_lines.append(
            f"lines {r['startingLine']}–{r['endingLine']}: "
            f"{c['abbreviatedOid']} | {c['authoredDate'][:10]} | "
            f"{c['author']['name']} | {msg_first_line}"
        )

    return {
        "url":         url,
        "file_url":    f"https://github.com/{owner}/{repo}/blob/{ref}/{path}",
        "center_line": center_line,
        "blame":       blame_lines,
    }


async def fetch_github_file_history(url: str, max_commits: int = 10,
                                     _token: str = "") -> dict:
    """Return recent commits that touched the file containing the given GitHub URL.

    Shows commit SHA, date, author, and message so the agent can identify
    whether a recent change introduced the failure.
    """
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", url)
    if not m:
        return {"error": f"cannot parse GitHub blob URL: {url!r}"}
    owner, repo, ref, path = m.group(1), m.group(2), m.group(3), m.group(4)

    token = _token or GITHUB_TOKEN
    headers = {**_HEADERS, **({"Authorization": f"Bearer {token}"} if token else {})}

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        r = await client.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            params={"path": path, "sha": ref, "per_page": max_commits},
        )
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code} fetching commit history"}

    commits = []
    for c in r.json():
        commits.append({
            "sha":     c["sha"][:8],
            "date":    (c.get("commit", {}).get("author") or {}).get("date", "")[:10],
            "author":  (c.get("commit", {}).get("author") or {}).get("name", ""),
            "message": c.get("commit", {}).get("message", "").splitlines()[0][:80],
            "url":     c.get("html_url", ""),
        })

    return {
        "file":    path,
        "file_url": f"https://github.com/{owner}/{repo}/blob/{ref}/{path}",
        "commits": commits,
    }


async def search_github_callers(repo: str, function_name: str, _token: str = "") -> dict:
    """Search a GitHub repo for callers of a function using the code search API.

    repo should be in 'owner/repo' or full URL form.
    Returns up to 10 matching file paths and line excerpts.
    """
    slug = _repo_slug(repo)
    if slug is None:
        return {"error": f"cannot parse repo: {repo!r}"}

    token = _token or GITHUB_TOKEN
    headers = {**_HEADERS, **({"Authorization": f"Bearer {token}"} if token else {})}

    query = f"{function_name} repo:{slug}"
    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
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
        "name": "fetch_github_blame",
        "description": (
            "Fetch git blame for lines around a specific line in a GitHub file. "
            "Shows which commit last changed each line — author, date, SHA, and message. "
            "Always call this after fetch_github_file when investigating a possible code_bug, "
            "to check whether a recent commit introduced the failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "GitHub blob URL (https://github.com/owner/repo/blob/ref/path)",
                },
                "center_line": {
                    "type": "integer",
                    "description": "Line number to centre the blame window on (1-indexed)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of blame context above and below center_line (default 30)",
                },
            },
            "required": ["url", "center_line"],
        },
    },
    {
        "name": "fetch_github_file_history",
        "description": (
            "Fetch recent git commit history for a file from a GitHub blob URL. "
            "Returns the last N commits touching that file: SHA, date, author, message. "
            "Always call this after fetch_github_file to see if a recent change caused the issue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "GitHub blob URL (https://github.com/owner/repo/blob/ref/path)",
                },
                "max_commits": {
                    "type": "integer",
                    "description": "Maximum number of recent commits to return (default 10)",
                },
            },
            "required": ["url"],
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
    "fetch_github_file":         fetch_github_file,
    "fetch_github_blame":        fetch_github_blame,
    "fetch_github_file_history": fetch_github_file_history,
    "search_github_callers":     search_github_callers,
}
