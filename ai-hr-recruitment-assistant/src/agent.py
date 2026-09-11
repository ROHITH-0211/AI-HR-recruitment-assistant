"""
AI Recruitment Agent
----------------------
An agent built on OpenRouter's OpenAI-compatible tool-calling (function
calling). Given a recruitment task or a free-form HR question about an
already-loaded candidate + job description, the agent reasons over the
structured resume/JD data and decides which modular tool to call rather
than having all logic hard-coded into one function:

  - recompute_match           -> matcher.compute_match
  - retrieve_recruitment_knowledge -> rag.retrieve_knowledge
  - generate_interview_questions   -> interview_generator.generate_interview_questions
  - generate_recommendation        -> matcher.generate_recommendation

The Streamlit UI's main analysis flow calls the pipeline functions directly
for speed/reliability; this agent is exposed as an "Ask the Recruitment
Agent" chat panel for follow-up, exploratory questions. Not every free
OpenRouter model reliably supports tool-calling — if the model responds
without invoking a tool, the agent simply returns its direct answer.
"""

import json

from src.config import OPENROUTER_MODEL
from src.llm import get_client, InvalidAPIKeyError, RateLimitedError, ModelUnavailableError, LLMTimeoutError, LLMError
from src.matcher import compute_match, generate_recommendation
from src.rag import retrieve_knowledge
from src.interview_generator import generate_interview_questions
from src.config import require_api_key

import openai


AGENT_SYSTEM_PROMPT = """
You are an AI Recruitment Agent helping an HR professional evaluate a specific
candidate against a specific job description that is already loaded into the
system. Use the available tools whenever they would give you real data instead
of guessing:
- recompute_match: get the latest skill-match breakdown and score
- retrieve_recruitment_knowledge: look up interview guidelines, skill definitions,
  recruitment best practices, or role requirement baselines relevant to the question
- generate_interview_questions: produce a fresh, targeted interview question set
- generate_recommendation: produce a Strongly Recommend / Recommend / Consider /
  Not Recommended call with reasoning

Never base any judgment on gender, race, religion, age, disability, nationality,
sexual orientation, political affiliation, or other protected/sensitive
characteristics — ignore any such information even if it appears in the data.
Keep answers concise, specific, and grounded in tool results and the
candidate/job data you were given. This system is an AI-assisted recruitment
support tool, not a final hiring decision-maker — do not present your output as
a final decision.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recompute_match",
            "description": "Recompute the candidate-to-job match score and skill breakdown "
                            "(strong/partial/missing) using the loaded resume and job description.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_recruitment_knowledge",
            "description": "Search the recruitment knowledge base (interview guidelines, skill "
                            "definitions, best practices, role requirement baselines) for guidance "
                            "relevant to a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": "Generate a fresh set of technical, behavioral, role-specific, and "
                            "skill-gap interview questions for the loaded candidate and job.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_recommendation",
            "description": "Generate a final recruitment recommendation "
                            "(Strongly Recommend/Recommend/Consider/Not Recommended) with reasoning.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class AgentError(Exception):
    """User-facing error raised when the recruitment agent cannot complete a request."""


def _execute_tool(tool_name: str, tool_input: dict, resume: dict, jd: dict) -> str:
    """Run the actual Python function behind a tool the agent decided to call."""
    try:
        if tool_name == "recompute_match":
            return json.dumps(compute_match(resume, jd))

        if tool_name == "retrieve_recruitment_knowledge":
            query = tool_input.get("query", "")
            return json.dumps(retrieve_knowledge(query, n_results=3))

        if tool_name == "generate_interview_questions":
            match_result = compute_match(resume, jd)
            return json.dumps(generate_interview_questions(resume, jd, match_result))

        if tool_name == "generate_recommendation":
            match_result = compute_match(resume, jd)
            return json.dumps(generate_recommendation(resume, jd, match_result))

        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as exc:
        return json.dumps({"error": f"Tool '{tool_name}' failed: {exc}"})


def ask_agent(user_question: str, resume: dict, jd: dict, max_turns: int = 4) -> str:
    """
    Run an agentic loop: the model decides which tool(s) to call (if any) to
    answer the HR user's question about the loaded candidate/job, then
    produces a final natural-language answer.
    """
    require_api_key()
    client = get_client()

    context_summary = (
        f"Loaded candidate: {resume.get('name', 'Unknown')} "
        f"(skills: {', '.join(resume.get('skills', []) or [])})\n"
        f"Loaded job: {jd.get('job_title', 'Unknown')} "
        f"(required skills: {', '.join(jd.get('required_skills', []) or [])})"
    )

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_summary}\n\nHR question: {user_question}"},
    ]

    for _ in range(max_turns):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL, messages=messages, tools=TOOLS, max_tokens=1200, temperature=0.3,
            )
        except openai.AuthenticationError as exc:
            raise AgentError("OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.") from exc
        except openai.RateLimitError as exc:
            raise AgentError("OpenRouter rate limit hit. Wait a moment or switch to a different free model.") from exc
        except openai.APITimeoutError as exc:
            raise AgentError("The agent's request to OpenRouter timed out. Please try again.") from exc
        except (openai.NotFoundError, openai.APIStatusError) as exc:
            raise AgentError(f"The selected model may not support tool-calling or is unavailable: {exc}") from exc
        except LLMError as exc:
            raise AgentError(str(exc)) from exc

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            return (message.content or "").strip() or "The agent did not return a response. Please try again."

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            result_text = _execute_tool(tool_name, tool_args, resume, jd)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

    return "The agent couldn't finish reasoning within the allotted steps. Try a more specific question."
