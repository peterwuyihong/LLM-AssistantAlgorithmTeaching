"""
Codeforces problem scraper.

Uses the Codeforces API for problem listing and HTML scraping
for problem statements and sample test cases.

API docs: https://codeforces.com/apiHelp
Problem page URL: https://codeforces.com/problemset/problem/{contestId}/{index}
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Optional
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# API helpers (no external deps — uses urllib to avoid adding requests/httpx)
# ---------------------------------------------------------------------------

_CF_API_BASE = "https://codeforces.com/api"
_CACHE_DIR: Optional[str] = None
_CACHE_TTL = 600  # 10 minutes


def set_cache_dir(path: str):
    """Set directory for API response caching."""
    global _CACHE_DIR
    _CACHE_DIR = path
    os.makedirs(path, exist_ok=True)


def _api_get(method: str, params: dict = None) -> dict:
    """Call a Codeforces API method. Returns parsed JSON result."""
    query = ""
    if params:
        query = "&" + "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{_CF_API_BASE}/{method}{query}"

    # Check cache
    if _CACHE_DIR:
        cache_key = method + "_" + "_".join(str(v) for v in (params or {}).values())
        cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.json")
        if os.path.isfile(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < _CACHE_TTL:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LAAT/1.0"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Codeforces API error: {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Codeforces API unreachable: {e.reason}") from e

    if data.get("status") != "OK":
        raise RuntimeError(f"Codeforces API returned: {data.get('status')} — {data.get('comment', 'unknown')}")

    result = data["result"]

    # Write cache
    if _CACHE_DIR:
        cache_key = method + "_" + "_".join(str(v) for v in (params or {}).values())
        cache_path = os.path.join(_CACHE_DIR, f"{cache_key}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

    return result


def fetch_problemset(tags: str = "", rating_min: int = 0, rating_max: int = 4000,
                     count: int = 50, offset: int = 0) -> list[dict]:
    """Fetch problem list from Codeforces API.

    Returns list of dicts with keys: contestId, index, name, rating, tags, url.
    """
    params = {}
    if tags:
        params["tags"] = tags

    data = _api_get("problemset.problems", params)
    problems = data.get("problems", [])
    stats = {f"{s['contestId']}{s['index']}": s.get("solvedCount", 0)
             for s in data.get("problemStatistics", [])}

    result = []
    for p in problems:
        rating = p.get("rating", 0) or 0
        if rating < rating_min or rating > rating_max:
            continue
        cid = p["contestId"]
        idx = p["index"]
        key = f"{cid}{idx}"
        result.append({
            "contestId": cid,
            "index": idx,
            "name": p.get("name", ""),
            "type": p.get("type", ""),
            "rating": rating,
            "tags": p.get("tags", []),
            "points": p.get("points"),
            "solvedCount": stats.get(key, 0),
            "url": f"https://codeforces.com/problemset/problem/{cid}/{idx}",
        })

    # Sort by contestId desc (newest first), then index
    result.sort(key=lambda x: (-x["contestId"], x["index"]))
    return result[offset:offset + count]


# ---------------------------------------------------------------------------
# HTML scraper for problem statements and test cases
# ---------------------------------------------------------------------------

class _ProblemHTMLParser(HTMLParser):
    """Extract problem statement and sample tests from CF problem page HTML."""

    def __init__(self):
        super().__init__()
        self.in_statement = False
        self.in_sample_test = False
        self.in_input_div = False
        self.in_output_div = False
        self.in_pre = False
        self.current_tag_class = ""
        self.statement_parts: list[str] = []
        self.sample_inputs: list[str] = []
        self.sample_outputs: list[str] = []
        self._current_text: list[str] = []
        self._collecting_input = False
        self._collecting_output = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        if "problem-statement" in cls:
            self.in_statement = True
        if "sample-test" in cls:
            self.in_sample_test = True
        if self.in_sample_test:
            if "input" in cls and "div" in tag:
                self.in_input_div = True
                self._collecting_input = True
                self._current_text = []
            if "output" in cls and "div" in tag:
                self.in_output_div = True
                self._collecting_output = True
                self._current_text = []
        if tag == "pre" and (self.in_input_div or self.in_output_div):
            self.in_pre = True
            self._current_text = []

    def handle_endtag(self, tag):
        if tag == "div":
            if self.in_input_div:
                text = "".join(self._current_text).strip()
                if text:
                    self.sample_inputs.append(text)
                self.in_input_div = False
                self._collecting_input = False
                self._current_text = []
            if self.in_output_div:
                text = "".join(self._current_text).strip()
                if text:
                    self.sample_outputs.append(text)
                self.in_output_div = False
                self._collecting_output = False
                self._current_text = []
        if tag == "pre":
            if self.in_pre:
                text = "".join(self._current_text).strip()
                if self._collecting_input and text:
                    # Replace the last entry if pre has the full content
                    if self.sample_inputs and not self.sample_inputs[-1]:
                        self.sample_inputs[-1] = text
                    else:
                        self.sample_inputs.append(text)
                elif self._collecting_output and text:
                    if self.sample_outputs and not self.sample_outputs[-1]:
                        self.sample_outputs[-1] = text
                    else:
                        self.sample_outputs.append(text)
            self.in_pre = False
            self._current_text = []
        if tag == "div" and self.in_statement and not self.in_sample_test:
            pass  # Stay in statement

    def handle_data(self, data):
        if self.in_pre and (self._collecting_input or self._collecting_output):
            self._current_text.append(data)
        elif self.in_statement and not self.in_sample_test:
            self.statement_parts.append(data)


def _extract_with_regex(html: str) -> dict:
    """Fallback: extract sample tests using regex when HTMLParser fails."""
    inputs = []
    outputs = []

    # Match input blocks: <div class="input">...<pre>...</pre>
    input_pattern = re.compile(
        r'<div\s+class="input[^"]*">.*?<pre>(.*?)</pre>',
        re.DOTALL
    )
    output_pattern = re.compile(
        r'<div\s+class="output[^"]*">.*?<pre>(.*?)</pre>',
        re.DOTALL
    )

    for m in input_pattern.finditer(html):
        text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        # Decode HTML entities
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        if text:
            inputs.append(text)

    for m in output_pattern.finditer(html):
        text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        if text:
            outputs.append(text)

    # Also try to extract problem statement
    statement = ""
    stmt_match = re.search(
        r'<div\s+class="problem-statement">(.+?)<div\s+class="sample-tests">',
        html, re.DOTALL
    )
    if stmt_match:
        statement = re.sub(r'<[^>]+>', ' ', stmt_match.group(1))
        statement = re.sub(r'\s+', ' ', statement).strip()
        statement = statement.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    return {
        "statement": statement,
        "sample_inputs": inputs,
        "sample_outputs": outputs,
    }


def scrape_problem_page(contest_id: int, index: str) -> dict:
    """Scrape a Codeforces problem page for statement and sample test cases.

    Returns dict with: statement, input_format, output_format,
                       sample_inputs, sample_outputs, url.
    """
    url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to fetch CF problem {contest_id}{index}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to Codeforces: {e.reason}") from e

    # Check for Cloudflare challenge
    if "cf-chl-opt" in html or "Just a moment" in html:
        # Try regex extraction anyway — sometimes the page content is there
        result = _extract_with_regex(html)
        if result["sample_inputs"] or result["statement"]:
            result["url"] = url
            result["cloudflare"] = True
            return result
        raise RuntimeError(
            f"Cloudflare blocked scraping for CF {contest_id}{index}. "
            "Try again later or use a different IP."
        )

    # Extract using regex (more reliable than HTMLParser for CF's complex HTML)
    result = _extract_with_regex(html)
    result["url"] = url

    # Try to extract input/output format sections
    input_fmt = ""
    output_fmt = ""
    input_match = re.search(
        r'<div\s+class="input-specification">.*?<p>(.*?)</div>',
        html, re.DOTALL
    )
    if input_match:
        input_fmt = re.sub(r'<[^>]+>', ' ', input_match.group(1))
        input_fmt = re.sub(r'\s+', ' ', input_fmt).strip()
        input_fmt = input_fmt.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    output_match = re.search(
        r'<div\s+class="output-specification">.*?<p>(.*?)</div>',
        html, re.DOTALL
    )
    if output_match:
        output_fmt = re.sub(r'<[^>]+>', ' ', output_match.group(1))
        output_fmt = re.sub(r'\s+', ' ', output_fmt).strip()
        output_fmt = output_fmt.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    result["input_format"] = input_fmt
    result["output_format"] = output_fmt

    return result


def import_cf_problem(contest_id: int, index: str, target_dir: str) -> dict:
    """Import a Codeforces problem into our format.

    Creates target_dir/question.json and target_dir/sample/*.in, *.ans files.
    Returns the question.json data dict.
    """
    # First, get basic info from API
    try:
        api_data = _api_get("problemset.problems")
        api_problems = api_data.get("problems", [])
        problem_info = None
        for p in api_problems:
            if p["contestId"] == contest_id and p["index"] == index:
                problem_info = p
                break
    except Exception:
        problem_info = None

    # Scrape the problem page
    page_data = scrape_problem_page(contest_id, index)

    # Build question.json
    name = problem_info.get("name", "") if problem_info else ""
    tags = problem_info.get("tags", []) if problem_info else []
    rating = problem_info.get("rating") if problem_info else None

    question_data = {
        "question": page_data.get("statement", ""),
        "input_format": page_data.get("input_format", ""),
        "output_format": page_data.get("output_format", ""),
        "data_range": "",
        "source": "codeforces",
        "contest_id": contest_id,
        "index": index,
        "name": name,
        "rating": rating,
        "tags": tags,
        "url": page_data.get("url", f"https://codeforces.com/problemset/problem/{contest_id}/{index}"),
    }

    # Write files
    os.makedirs(target_dir, exist_ok=True)
    sample_dir = os.path.join(target_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    with open(os.path.join(target_dir, "question.json"), "w", encoding="utf-8") as f:
        json.dump(question_data, f, indent=2, ensure_ascii=False)

    # Write sample test cases
    sample_inputs = page_data.get("sample_inputs", [])
    sample_outputs = page_data.get("sample_outputs", [])
    for i, (inp, out) in enumerate(zip(sample_inputs, sample_outputs), 1):
        with open(os.path.join(sample_dir, f"{i}.in"), "w", encoding="utf-8") as f:
            f.write(inp)
        with open(os.path.join(sample_dir, f"{i}.ans"), "w", encoding="utf-8") as f:
            f.write(out)

    return question_data
