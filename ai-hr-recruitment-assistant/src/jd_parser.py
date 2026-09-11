"""
Job Description Parser
-----------------------
Takes a raw job description (typed or uploaded) and extracts structured
requirements: required skills, preferred skills, qualifications, experience
level, responsibilities, technical requirements, and soft skills.
"""

from src.llm import chat_json
from src.resume_parser import extract_text_from_pdf, extract_text_from_docx, ResumeParsingError


JD_SCHEMA_INSTRUCTIONS = """
You are an expert job description analyst for an HR recruitment system.

Read the job description text and extract it into STRICT JSON only
(no markdown fences, no commentary, no extra text).

Use exactly this schema:
{
  "job_title": string,
  "company": string,
  "required_skills": [string],
  "preferred_skills": [string],
  "technical_requirements": [string],
  "soft_skills": [string],
  "qualifications": [string],
  "min_experience_years": number,
  "responsibilities": [string],
  "employment_type": string,
  "location": string
}

Rules:
- "required_skills" are must-have/mandatory skills explicitly stated as required.
- "preferred_skills" are nice-to-have or bonus skills.
- "technical_requirements" covers tools/platforms/methodologies beyond core skills (e.g. CI/CD, cloud platform).
- "soft_skills" covers communication, teamwork, leadership, etc. if mentioned.
- If a field is missing, use an empty string, empty list, or 0.
- Return ONLY the JSON object, nothing else.
"""


class JDParsingError(Exception):
    """Raised when a job description is empty or its file can't be read."""


def extract_jd_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text from an uploaded JD file (PDF or DOCX)."""
    if not file_bytes:
        raise JDParsingError("The uploaded job description file is empty.")

    lower_name = filename.lower()
    try:
        if lower_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif lower_name.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        else:
            raise JDParsingError("Unsupported file type. Please upload a PDF or DOCX, or paste text instead.")
    except ResumeParsingError as exc:
        raise JDParsingError(str(exc))

    if not text.strip():
        raise JDParsingError("No readable text was found in this job description file.")
    return text


def parse_job_description(jd_text: str) -> dict:
    """Send JD text to the LLM and get back structured JSON requirements."""
    if not jd_text or not jd_text.strip():
        raise JDParsingError("Job description text is empty. Please paste or upload a job description.")

    structured = chat_json(JD_SCHEMA_INSTRUCTIONS, f"Job description text:\n\n{jd_text[:12000]}")
    structured["_raw_text"] = jd_text
    return structured
