"""
Interview Question Generator
------------------------------
Generates personalized technical, behavioral, role-specific, and skill-gap
interview questions based on the candidate's skills/projects/experience,
the job requirements, and the skill-gap breakdown — grounded with
retrieved best-practice guidance (RAG) and answered by the OpenRouter LLM.
"""

from src.llm import chat_json
from src.rag import retrieve_knowledge


QUESTION_GEN_SYSTEM_PROMPT = """
You are an expert technical interviewer creating a personalized interview plan.
Base your questions only on the candidate/job information and interview
guidance context provided. Do not reference or infer protected or sensitive
characteristics (age, gender, religion, disability, nationality, etc.).

Return STRICT JSON only, no markdown fences, no commentary, with this schema:
{
  "technical_questions": [
    {"question": string, "category": "Technical", "why_relevant": string,
     "evaluation_points": [string], "expected_strong_answer": string}
  ],
  "behavioral_questions": [
    {"question": string, "category": "Behavioral", "why_relevant": string,
     "evaluation_points": [string], "expected_strong_answer": string}
  ],
  "role_specific_questions": [
    {"question": string, "category": "Role-Specific", "why_relevant": string,
     "evaluation_points": [string], "expected_strong_answer": string}
  ],
  "skill_gap_questions": [
    {"question": string, "category": "Skill-Gap", "why_relevant": string,
     "evaluation_points": [string], "expected_strong_answer": string}
  ]
}

Rules:
- Generate 3 technical, 2 behavioral, 2 role-specific, and 2 skill-gap questions.
- "why_relevant" briefly explains what prompted the question (a project, a job
  responsibility, or a specific missing/partial skill).
- "skill_gap_questions" must specifically target the candidate's partial-match
  or missing required skills, framed as gauging ramp-up ability, not penalizing the gap.
- "evaluation_points" are 2-3 concrete things a strong answer would include.
- "expected_strong_answer" is 1-2 sentences describing what a strong response looks like.
- Return ONLY the JSON object.
"""


def generate_interview_questions(resume: dict, jd: dict, match_result: dict) -> dict:
    """Generate a personalized interview question set, grounded with RAG context."""
    guidance_chunks = retrieve_knowledge(
        f"interview questions and evaluation guidelines for {jd.get('job_title', 'this role')} "
        f"covering {', '.join((jd.get('required_skills') or [])[:5])}",
        n_results=3,
    )
    guidance_text = "\n\n".join(f"[{c['source']}] {c['text']}" for c in guidance_chunks)

    projects = resume.get("projects", []) or []
    project_summaries = "; ".join(
        f"{p.get('name', '')}: {p.get('description', '')}" for p in projects
    )

    context = f"""
Candidate name: {resume.get('name', 'Unknown')}
Candidate skills: {', '.join(resume.get('skills', []) or [])}
Candidate projects: {project_summaries}
Candidate experience: {match_result.get('candidate_years', 0)} years

Job title: {jd.get('job_title', 'Unknown')}
Job responsibilities: {'; '.join(jd.get('responsibilities', []) or [])}
Required skills: {', '.join(jd.get('required_skills', []) or [])}

Strong-match skills: {', '.join(match_result['required_skills']['strong_match'])}
Partial-match skills: {', '.join(match_result['required_skills']['partial_match'])}
Missing required skills: {', '.join(match_result['required_skills']['missing'])}

Relevant interview guidance (from knowledge base):
{guidance_text}
"""
    return chat_json(QUESTION_GEN_SYSTEM_PROMPT, context, max_tokens=2500)
