"""Deterministic scoring engine used by the MCP tools.

Same approach as the resume-ranker project: a transparent weighted scorecard (no trained model),
evidence-weighted skills, experience from date ranges, and blind-mode redaction.
Pure Python plus scikit-learn, no network, fully deterministic.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

WEIGHTS = {"tfidf": 0.25, "skills": 0.55, "experience": 0.20}
LISTED_ONLY_CREDIT = 0.5
MAX_YEARS = 10
ALIASES = {"nlp": "natural language processing", "ml": "machine learning", "js": "javascript",
           "sklearn": "scikit-learn", "llms": "llm", "postgres": "postgresql", "k8s": "kubernetes",
           "genai": "generative ai", "gen ai": "generative ai"}

_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_EXP_START = re.compile(r"^\s*(work experience|professional experience|experience|employment)\s*:?\s*$", re.I | re.M)
_END = re.compile(r"^\s*(education|projects?|skills|technical skills|certifications?|awards|summary|languages)\s*:?\s*$",
                  re.I | re.M)
_SKILLS_START = re.compile(r"^\s*(technical skills|skills|core competencies)\s*:?\s*$", re.I | re.M)
_OTHER = re.compile(r"^\s*(education|projects?|experience|work experience|professional experience|employment|"
                    r"certifications?|awards|summary|languages)\s*:?\s*$", re.I | re.M)
_RANGE = re.compile(r"(?:(?P<m1>[A-Za-z]{3,9})\.?\s+)?(?P<y1>(?:19|20)\d{2})\s*(?:-|–|—|to)\s*"
                    r"(?:(?P<now>present|current|now)|(?:(?P<m2>[A-Za-z]{3,9})\.?\s+)?(?P<y2>(?:19|20)\d{2}))", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+|(?:linkedin|github)\.com/\S+", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")


@lru_cache(maxsize=1)
def skills_list() -> tuple:
    path = Path(__file__).with_name("skills_list.json")
    return tuple(s.lower() for s in json.loads(path.read_text(encoding="utf-8"))["skills"])


def _pat(skill: str) -> re.Pattern:
    return re.compile(r"(?<![a-zA-Z0-9+#])" + re.escape(skill) + r"(?![a-zA-Z0-9+#])")


def extract_skills(text: str) -> list:
    t = (text or "").lower()
    for alias, canon in ALIASES.items():
        t = _pat(alias).sub(canon, t)
    return sorted(s for s in skills_list() if _pat(s).search(t))


def redact_pii(text: str) -> str:
    """Drop contact details and the first line (usually the name)."""
    lines = (text or "").splitlines()
    if lines and len(lines[0]) <= 40 and "@" not in lines[0] and not any(c.isdigit() for c in lines[0]):
        lines = lines[1:]
    out = "\n".join(lines)
    for rx in (_EMAIL, _URL, _PHONE):
        out = rx.sub(" ", out)
    return out


def split_skills_section(text: str) -> tuple:
    m = _SKILLS_START.search(text)
    if not m:
        return text, ""
    rest = text[m.end():]
    end = _OTHER.search(rest)
    return text[:m.start()] + (rest[end.start():] if end else ""), rest[:end.start()] if end else rest


def experience_years(text: str, today: Optional[date] = None) -> float:
    """Union of date ranges inside the Experience section (overlaps merged, education ignored)."""
    today = today or date.today()
    now = today.year * 12 + today.month
    start = _EXP_START.search(text or "")
    if not start:
        return 0.0
    rest = text[start.end():]
    end = _END.search(rest)
    section = rest[:end.start()] if end else rest
    spans = []
    for m in _RANGE.finditer(section):
        s = int(m["y1"]) * 12 + _MONTHS.get((m["m1"] or "")[:3].lower(), 1)
        e = now if m["now"] else int(m["y2"]) * 12 + _MONTHS.get((m["m2"] or "")[:3].lower(), 12)
        if e >= s:
            spans.append((s, min(e, now)))
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return round(sum(e - s for s, e in merged) / 12, 2)


def score_resumes(jd_text: str, resumes: list, today: Optional[date] = None) -> list:
    """Rank resumes (list of {"name","text"}) against a JD. Best first, with exact breakdowns."""
    jd_skills = extract_skills(jd_text)
    texts = [redact_pii(r["text"]) for r in resumes]
    if resumes:
        try:
            m = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([jd_text] + texts)
            tfidf = [float(x) for x in cosine_similarity(m[0], m[1:])[0]]
        except ValueError:
            tfidf = [0.0] * len(resumes)
    else:
        tfidf = []

    ranked = []
    for r, text, tf in zip(resumes, texts, tfidf):
        body, listed_section = split_skills_section(text)
        evidence, listed = set(extract_skills(body)), set(extract_skills(listed_section))
        listed_only = sorted((listed - evidence) & set(jd_skills))
        matched = sorted(evidence | listed)
        jd = set(jd_skills)
        coverage = None
        if jd:
            full = (evidence & jd) - set(listed_only)  # skills with evidence in the resume body
            coverage = (len(full) + LISTED_ONLY_CREDIT * len(listed_only)) / len(jd)
        years = experience_years(r["text"], today)
        signals = {"tfidf": tf, "skills": coverage, "experience": min(years / MAX_YEARS, 1.0)}
        used = {k: w for k, w in WEIGHTS.items() if signals[k] is not None}
        total = sum(used.values())
        contrib = {k: round(100 * (w / total) * min(max(signals[k], 0.0), 1.0), 2) for k, w in used.items()}
        ranked.append({"name": r["name"], "score": round(sum(contrib.values()), 2), "contributions": contrib,
                       "matched_skills": matched, "missing_skills": sorted(jd - set(matched)),
                       "listed_only_skills": listed_only, "experience_years": years})
    ranked.sort(key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(ranked, 1):
        c["rank"] = i
    return ranked
