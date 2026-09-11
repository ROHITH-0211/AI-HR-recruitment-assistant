"""
Candidate-JD Matcher
--------------------
Computes a 0-100 match score between a parsed resume and a parsed job
description using a blend of:
  1. Semantic similarity between resume text and JD text (overall fit)
  2. Skill overlap (required skills coverage, weighted higher than preferred)
  3. Experience level fit

Then asks the LLM (via OpenRouter) to explain, in plain language, why the
candidate does or does not match, and to produce a final recommendation.
Both prompts explicitly exclude protected/sensitive characteristics.
"""

from src.skill_extractor import get_embedder, categorize_skills, cosine_sim
from src.llm import chat_text, chat_json


def _skill_coverage_score(skill_categories: dict, total_required: int) -> float:
    """Strong match = full credit, partial = half credit."""
    if total_required == 0:
        return 1.0
    strong = len(skill_categories["strong_match"])
    partial = len(skill_categories["partial_match"])
    return min(1.0, (strong + 0.5 * partial) / total_required)


def _experience_fit_score(candidate_years: float, min_required_years: float) -> float:
    if min_required_years <= 0:
        return 1.0
    if candidate_years >= min_required_years:
        return 1.0
    return max(0.0, candidate_years / min_required_years)


def compute_match(resume: dict, jd: dict) -> dict:
    """
    Compute the full match result: overall score, semantic similarity,
    required/preferred skill categorization, and experience fit.
    """
    embedder = get_embedder()

    resume_text = resume.get("_raw_text", "") or resume.get("summary", "")
    jd_text = jd.get("_raw_text", "")

    resume_emb = embedder.encode(resume_text[:5000])
    jd_emb = embedder.encode(jd_text[:5000])
    semantic_similarity = cosine_sim(resume_emb, jd_emb)  # 0-1

    candidate_skills = resume.get("skills", []) or []
    required_skills = jd.get("required_skills", []) or []
    preferred_skills = jd.get("preferred_skills", []) or []

    required_categories = categorize_skills(candidate_skills, required_skills)
    preferred_categories = categorize_skills(candidate_skills, preferred_skills)

    required_coverage = _skill_coverage_score(required_categories, len(required_skills))
    preferred_coverage = _skill_coverage_score(preferred_categories, len(preferred_skills))

    candidate_years = float(resume.get("total_experience_years", 0) or 0)
    min_required_years = float(jd.get("min_experience_years", 0) or 0)
    experience_fit = _experience_fit_score(candidate_years, min_required_years)

    weighted = (
        0.40 * required_coverage
        + 0.25 * semantic_similarity
        + 0.20 * experience_fit
        + 0.15 * preferred_coverage
    )
    match_score = round(weighted * 100, 1)

    return {
        "match_score": match_score,
        "semantic_similarity": round(semantic_similarity * 100, 1),
        "required_skills": required_categories,
        "preferred_skills": preferred_categories,
        "experience_fit_pct": round(experience_fit * 100, 1),
        "candidate_years": candidate_years,
        "min_required_years": min_required_years,
    }


EXPLANATION_SYSTEM_PROMPT = """
You are an HR assistant writing a concise, factual explanation of a candidate-job
match, based only on the structured data provided. Do not consider or mention
gender, race, religion, age, disability, nationality, sexual orientation,
political affiliation, or any other protected/sensitive characteristic — base
the explanation strictly on skills, experience, projects, and qualifications.
If any such sensitive information appears in the input, ignore it entirely.

Write 4-6 sentences covering, in order:
- Overall fit summary
- Strong areas / matching skills
- Partial matches
- Missing requirements
- Relevant experience
- Any potential concerns (skills- or experience-based only)

Keep it professional and specific. Do not repeat raw JSON back to the user.
"""


def generate_match_explanation(resume: dict, jd: dict, match_result: dict) -> str:
    """Ask the LLM to explain the match result in plain language."""
    context = f"""
Candidate: {resume.get('name', 'Unknown')}
Candidate experience: {match_result['candidate_years']} years
Candidate skills: {', '.join(resume.get('skills', []) or [])}
Candidate projects: {'; '.join(p.get('name', '') for p in (resume.get('projects') or []))}

Job title: {jd.get('job_title', 'Unknown')}
Required experience: {match_result['min_required_years']} years
Required skills: {', '.join(jd.get('required_skills', []) or [])}
Preferred skills: {', '.join(jd.get('preferred_skills', []) or [])}

Overall Match Score: {match_result['match_score']}%
Strong matches: {', '.join(match_result['required_skills']['strong_match'])}
Partial matches: {', '.join(match_result['required_skills']['partial_match'])}
Missing required skills: {', '.join(match_result['required_skills']['missing'])}
"""
    return chat_text(EXPLANATION_SYSTEM_PROMPT, context, max_tokens=600)


RECOMMENDATION_SYSTEM_PROMPT = """
You are an HR assistant producing a final recruitment recommendation based only
on skills, experience, projects, certifications, and qualifications data
provided. Never factor in gender, race, religion, age, disability, nationality,
sexual orientation, political affiliation, or any other protected/sensitive
characteristic. If any such information appears in the input, ignore it.

Return STRICT JSON only, no markdown fences, no commentary, with this schema:
{
  "decision": "Strongly Recommend" | "Recommend" | "Consider" | "Not Recommended",
  "explanation": string
}

Guidance:
- Strongly Recommend: meets/exceeds nearly all required skills and experience.
- Recommend: meets most required skills; minor, learnable gaps.
- Consider: meets some required skills but has notable gaps.
- Not Recommended: missing multiple critical required skills or experience level.
- "explanation" should be 2-4 sentences, specific and factual, based only on job-relevant information.
- Return ONLY the JSON object.
"""


def generate_recommendation(resume: dict, jd: dict, match_result: dict) -> dict:
    """Ask the LLM for a final Strongly Recommend / Recommend / Consider / Not Recommended call."""
    req = match_result["required_skills"]
    context = f"""
Match score: {match_result['match_score']}/100
Required skills strong match: {', '.join(req['strong_match'])}
Required skills partial match: {', '.join(req['partial_match'])}
Required skills missing: {', '.join(req['missing'])}
Candidate experience: {match_result['candidate_years']} years
Required experience: {match_result['min_required_years']} years
Candidate certifications: {', '.join(resume.get('certifications', []) or [])}
Candidate projects: {'; '.join(p.get('name', '') for p in (resume.get('projects') or []))}
"""
    return chat_json(RECOMMENDATION_SYSTEM_PROMPT, context, max_tokens=500)
