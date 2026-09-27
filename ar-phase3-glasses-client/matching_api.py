"""
HTTP client for the Phase 2 matching engine — stdlib only, so it runs on a
paired phone's Python runtime (or ports trivially to Kotlin/Swift) without
needing a dependency manager for a single API call.

This is the piece that changes when Phase 2's storage backend changes
(FAISS/Milvus swap, auth, etc.) — nothing here or in the state machine or
pipeline should need to change, only the URL and any auth headers.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MatchedProfile:
    profile_id: str
    name: str
    role: Optional[str]
    company: Optional[str]
    similarity: float


class MatchingAPIError(Exception):
    pass


class MatchingAPIClient:
    def __init__(self, base_url: str, timeout_s: float = 0.7):
        # timeout defaults to MATCH_TIMEOUT_MS from state_machine.py — a slow
        # backend should look like "no match" to the wearer, not a hang.
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def match(self, event_id: str, embedding: List[float], top_k: int = 1,
              min_similarity: float = 0.85) -> List[MatchedProfile]:
        payload = json.dumps({
            "event_id": event_id,
            "embedding": embedding,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/match",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise MatchingAPIError(str(e)) from e

        return [
            MatchedProfile(
                profile_id=m["profile"]["profile_id"],
                name=m["profile"]["name"],
                role=m["profile"].get("role"),
                company=m["profile"].get("company"),
                similarity=m["similarity"],
            )
            for m in body.get("matches", [])
        ]
