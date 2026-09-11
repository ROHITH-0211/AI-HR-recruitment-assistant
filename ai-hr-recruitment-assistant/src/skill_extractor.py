"""
Skill Extractor
---------------
Utility functions for normalizing skill names and categorizing candidate
skills against job requirements as Strong Match / Partial Match / Missing,
using sentence-embedding cosine similarity (so "JS" and "JavaScript" still
match, not just exact string equality).
"""

from functools import lru_cache
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

STRONG_MATCH_THRESHOLD = 0.75
PARTIAL_MATCH_THRESHOLD = 0.55


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load (and cache) the sentence-transformer embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def normalize_skill(skill: str) -> str:
    return skill.strip().lower()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def categorize_skills(candidate_skills: List[str], required_skills: List[str]) -> Dict[str, List[str]]:
    """
    Compare candidate skills to required job skills using embedding similarity.
    Returns a dict with 'strong_match', 'partial_match', and 'missing' lists,
    reported from the job's perspective (one entry per required skill).
    """
    result = {"strong_match": [], "partial_match": [], "missing": []}

    if not required_skills:
        return result
    if not candidate_skills:
        result["missing"] = list(required_skills)
        return result

    embedder = get_embedder()
    candidate_embeddings = embedder.encode(candidate_skills)
    required_embeddings = embedder.encode(required_skills)

    for req_skill, req_emb in zip(required_skills, required_embeddings):
        best_score = max(cosine_sim(req_emb, cand_emb) for cand_emb in candidate_embeddings)
        if best_score >= STRONG_MATCH_THRESHOLD:
            result["strong_match"].append(req_skill)
        elif best_score >= PARTIAL_MATCH_THRESHOLD:
            result["partial_match"].append(req_skill)
        else:
            result["missing"].append(req_skill)

    return result


def build_skill_comparison_table(candidate_skills: List[str], required_skills: List[str],
                                  preferred_skills: List[str]) -> List[dict]:
    """
    Build a row-per-skill comparison table:
    Skill | Candidate Has It | Requirement Type | Status
    used directly by the Streamlit "Skill Comparison" view.
    """
    rows = []
    req_categories = categorize_skills(candidate_skills, required_skills)
    pref_categories = categorize_skills(candidate_skills, preferred_skills)

    def status_for(skill, categories):
        if skill in categories["strong_match"]:
            return "Strong Match"
        if skill in categories["partial_match"]:
            return "Partial Match"
        return "Missing"

    def has_it_label(status):
        return {"Strong Match": "Yes", "Partial Match": "Basic", "Missing": "No"}[status]

    for skill in required_skills:
        status = status_for(skill, req_categories)
        rows.append({"Skill": skill, "Candidate": has_it_label(status), "Requirement": "Required", "Status": status})

    for skill in preferred_skills:
        status = status_for(skill, pref_categories)
        rows.append({"Skill": skill, "Candidate": has_it_label(status), "Requirement": "Preferred", "Status": status})

    return rows
