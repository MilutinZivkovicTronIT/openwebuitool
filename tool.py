"""
title: Company & Similar Email Search (POST Actions)
author: TronIT
version: 1.2.0
license: MIT
description: Detect a company in the user text (preferring 8-digit codes) by POSTing an OData Action to fetch companies. Then call the OData Action `searchSimilar(q, company)` via POST, and return normalized results + LLM reply.
requirements: requests
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import time
import re
import requests

# --- Endpoints ---
DEFAULT_COMPANY_ENDPOINT = (
    "https://tronit-d-o-o--tronit-dev-01-unsp4h59-dev-vectorizatione62b1c508."
    "cfapps.eu10-004.hana.ondemand.com/odata/v4/vectorization/getCompanies"
)
# Build /searchSimilar next to /getCompanies
DEFAULT_SEARCH_ENDPOINT = DEFAULT_COMPANY_ENDPOINT.rsplit("/", 1)[0] + "/searchSimilar"

# --- Cache / constants ---
_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
TTL_SECONDS = 300
CODE_RE = re.compile(r"\b\d{8}\b")  # exactly 8 digits


# --- Helpers: companies ---
def _normalize_company_code(val: Any) -> str:
    """
    Normalize COMPANYCODE to preserve leading zeros.
    - Numbers are zero-padded to 8 digits.
    - Digit-only strings up to length 8 are left-padded to 8.
    - Otherwise, returned as trimmed string.
    """
    if isinstance(val, (int, float)):
        try:
            return f"{int(val):08d}"
        except Exception:
            return str(int(val))
    s = str(val).strip()
    if s.isdigit() and len(s) <= 8:
        return s.zfill(8)
    return s


def _post_companies(endpoint: str) -> List[Dict[str, Any]]:
    now = time.time()
    cached = _CACHE.get(endpoint)
    if cached and (now - cached[0]) < TTL_SECONDS:
        return cached[1]

    # OData Action invocation -> POST with empty body
    resp = requests.post(endpoint, json={}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # CAP usually returns {"value": [...]}, but also support plain lists.
    if isinstance(data, dict) and "value" in data:
        companies = data["value"]
    elif isinstance(data, list):
        companies = data
    else:
        raise ValueError(
            "Unexpected companies response; expected JSON array or {'value':[...]}."
        )

    # Normalize fields to strings, with leading-zero preservation for codes.
    normed: List[Dict[str, Any]] = []
    for it in companies:
        normed.append(
            {
                "COMPANY": str(it.get("COMPANY", "")).strip(),
                "COMPANYCODE": _normalize_company_code(it.get("COMPANYCODE", "")),
                "COMPANYNAME": str(it.get("COMPANYNAME", "")).strip(),
            }
        )

    _CACHE[endpoint] = (now, normed)
    return normed


def _word_boundary_search(hay: str, needle: str) -> bool:
    if not needle:
        return False
    pat = re.compile(rf"\b{re.escape(needle)}\b", re.IGNORECASE)
    return bool(pat.search(hay))


def _find_occurrences_in_text(
    text: str, companies: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    msg = text or ""
    strict_hits: List[Dict[str, Any]] = []
    codes_in_text = set(m.group(0) for m in CODE_RE.finditer(msg))

    for item in companies:
        code = item["COMPANYCODE"]
        company = item["COMPANY"]
        name = item["COMPANYNAME"]

        code_hit = code and (code in codes_in_text or _word_boundary_search(msg, code))
        name_hit = (name and _word_boundary_search(msg, name)) or (
            company and _word_boundary_search(msg, company)
        )

        if code_hit or name_hit:
            hit = dict(item)
            hit["_matched_by"] = "strict"
            strict_hits.append(hit)

    if strict_hits:
        return strict_hits

    # Loose contains-based fallback
    loose_hits: List[Dict[str, Any]] = []
    low_msg = msg.lower()
    for item in companies:
        c = item["COMPANY"].lower()
        n = item["COMPANYNAME"].lower()
        if (c and c in low_msg) or (n and n in low_msg):
            hit = dict(item)
            hit["_matched_by"] = "loose"
            loose_hits.append(hit)

    return loose_hits


def _pick_company(item: Dict[str, Any], prefer: str = "code") -> str:
    if prefer == "code" and item.get("COMPANYCODE"):
        return item["COMPANYCODE"]
    # prefer readable name if code not available
    return item.get("COMPANY") or item.get("COMPANYNAME") or ""


# --- Helpers: searchSimilar ---
def _normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(item.get("id", "")),
        "subject": str(item.get("subject", "")),
        "fromDisplay": str(item.get("fromDisplay", "")),
        "sentOn": item.get("sentOn") or "",  # keep ISO/timestamp as-is
        "score": float(item.get("score", 0.0)),
        "bodyText": str(item.get("bodyText", "")),
    }


def _post_search_similar(
    endpoint: str, q: str, company: str, **extra_payload: Any
) -> Dict[str, Any]:
    """
    Invoke CAP OData v4 Action via POST:
      POST <endpoint>
      Body: {"q": "...", "company": "..."} (+ optional extras supported by backend)
    """
    payload: Dict[str, Any] = {"q": q or "", "company": company or ""}
    # Only include extras if explicitly provided (e.g., future params your action might accept)
    payload.update({k: v for k, v in extra_payload.items() if v is not None})

    resp = requests.post(endpoint, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json() or {}

    # Extract & normalize results
    raw_results = data.get("results") or []
    norm_results = [_normalize_result(it) for it in raw_results]
    # Sort by score desc for convenience
    norm_results.sort(key=lambda r: r.get("score") or 0.0, reverse=True)

    llm = data.get("llm") or {}
    usage = (llm.get("usage") or {}) if isinstance(llm, dict) else {}

    return {
        "query": data.get("query", q),
        "company": data.get("company", company),
        "k": data.get("k"),
        "minScore": data.get("minScore"),
        "count": data.get("count", len(norm_results)),
        "results": norm_results,
        "llm": {
            "answer": llm.get("answer") if isinstance(llm, dict) else None,
            "model": llm.get("model") if isinstance(llm, dict) else None,
            "message": llm.get("message") if isinstance(llm, dict) else None,
            "error": llm.get("error") if isinstance(llm, dict) else None,
            "usage": (
                {
                    "promptTokens": usage.get("promptTokens"),
                    "completionTokens": usage.get("completionTokens"),
                    "totalTokens": usage.get("totalTokens"),
                }
                if isinstance(usage, dict)
                else None
            ),
        },
        "_raw": data,  # optional: helpful for debugging; remove if you don't want to expose
    }


# --- Public API ---
class Tools:
    def __init__(self):
        # Keep parity with your first tool so outputs can be cited if your UI uses it
        self.citation = True

    def check_company(
        self, text: str, endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        POST the OData action to fetch all companies, then detect any occurrences
        of a company code or name in the chat text.
        """
        target = endpoint or DEFAULT_COMPANY_ENDPOINT
        companies = _post_companies(target)
        matches = _find_occurrences_in_text(text or "", companies)
        return {
            "found": len(matches) > 0,
            "count": len(matches),
            "matches": matches,
            "strategy": "fetch-first-post",
        }

    def search_similar(
        self,
        q: str,
        company: str,
        endpoint: Optional[str] = None,
        include_body: bool = True,
        **extra_payload: Any,
    ) -> Dict[str, Any]:
        """
        Call OData Action searchSimilar(q, company) and return normalized results.
        Set include_body=False to strip 'bodyText' from each result (smaller payloads).
        """
        target = endpoint or DEFAULT_SEARCH_ENDPOINT
        out = _post_search_similar(target, q, company, **extra_payload)

        results = out.get("results", [])
        if not include_body:
            for r in results:
                r.pop("bodyText", None)

        return {
            "found": len(results) > 0,
            "count": len(results),
            "results": results,
            "strategy": "search-similar-post",
            "query": out.get("query"),
            "company": out.get("company"),
            "llm": out.get("llm"),
        }

    def detect_then_search(
        self,
        text: str,
        prefer: str = "code",  # "code" or "name"
        companies_endpoint: Optional[str] = None,
        search_endpoint: Optional[str] = None,
        include_body: bool = False,
        **extra_payload: Any,
    ) -> Dict[str, Any]:
        """
        End-to-end: detect company from free text (prefer 8-digit code), then call searchSimilar.
        Fallback: if no catalog match but an 8-digit code exists in text, use that code anyway.
        """
        # Step 1: detect against company catalog
        detected = self.check_company(text, endpoint=companies_endpoint)
        if detected.get("found"):
            picked = _pick_company(detected["matches"][0], prefer=prefer)
            sr = self.search_similar(
                q=text,
                company=picked,
                endpoint=search_endpoint,
                include_body=include_body,
                **extra_payload,
            )
            sr["strategy"] = "detect-then-search"
            sr["detectedMatches"] = detected["matches"]
            sr["pickedCompany"] = picked
            return sr

        # Step 2: fallback — use any 8-digit code directly from text
        codes = list(CODE_RE.findall(text or ""))
        if codes:
            picked = codes[0]
            sr = self.search_similar(
                q=text,
                company=picked,
                endpoint=search_endpoint,
                include_body=include_body,
                **extra_payload,
            )
            sr["strategy"] = "regex-only-detect-then-search"
            sr["detectedMatches"] = []
            sr["pickedCompany"] = picked
            return sr

        # Step 3: no detection at all
        return {
            "found": False,
            "count": 0,
            "results": [],
            "strategy": "detect-then-search",
            "query": text,
            "company": None,
            "llm": None,
            "detectedMatches": [],
            "message": "No company detected in text.",
        }
