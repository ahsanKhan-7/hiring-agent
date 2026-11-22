"""LLM access for the interview-planning step. The workflow never needs a network to run."""
from __future__ import annotations

import os
import re
from typing import Protocol


class LLM(Protocol):
    async def generate(self, prompt: str) -> str: ...


class TemplateLLM:
    """Deterministic offline fallback. Reads the skills out of the prompt and writes template questions."""

    async def generate(self, prompt: str) -> str:
        matched = _field(prompt, "MATCHED")
        missing = _field(prompt, "MISSING")
        qs = [f"Walk me through a project where you used {s} and what went wrong." for s in matched[:3]]
        qs += [f"The role needs {s}. What is your closest experience with it?" for s in missing[:2]]
        qs += ["Describe a time your first approach failed and how you changed it.",
               "How do you decide when a solution is good enough to ship?"]
        return "\n".join(f"{i}. {q}" for i, q in enumerate(qs[:5], 1))


class GeminiLLM:
    """Optional real LLM through google-genai. Not exercised in CI (needs a key)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    async def generate(self, prompt: str) -> str:
        from google import genai
        client = genai.Client(api_key=self.api_key)
        resp = await client.aio.models.generate_content(model=self.model, contents=prompt)
        return resp.text


def _field(prompt: str, label: str) -> list:
    m = re.search(rf"^{label}: (.*)$", prompt, re.M)
    return [x.strip() for x in m.group(1).split(",") if x.strip() and x.strip() != "none"] if m else []


def parse_numbered(text: str, expected: int = 5) -> list:
    qs = [re.sub(r"^\d+[\.\)]\s*", "", ln.strip()) for ln in (text or "").splitlines()
          if re.match(r"^\s*\d+[\.\)]\s*", ln)]
    return qs[:expected]


def build_prompt(jd_text: str, name: str, matched: list, missing: list) -> str:
    return (f"You are a technical interviewer. Write exactly 5 numbered interview questions for {name}.\n"
            f"3 should probe depth on matched skills, 2 should test the genuinely missing ones.\n"
            f"Return only the numbered list.\n"
            f"JOB: {jd_text[:1500]}\n"
            f"MATCHED: {', '.join(matched) or 'none'}\n"
            f"MISSING: {', '.join(missing) or 'none'}")
