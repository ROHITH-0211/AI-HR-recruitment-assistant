# 🧑‍💼 AI HR Recruitment Assistant

Screen a resume against a job description and get back a match score, a
skill-gap breakdown, ready-to-ask interview questions, and a recruiter
recommendation — in the time it takes to make coffee.

It runs entirely on a free [OpenRouter](https://openrouter.ai) model, so
there's no GPU, no local LLM install, and no paid API required to try it.

> **This is a support tool, not a decision-maker.** Every score and
> recommendation is AI-generated and meant to save a recruiter time — a
> human should always make the final call.

---

## What you get

| | |
|---|---|
| 📄 **Resume parsing** | Upload a PDF/DOCX → structured JSON (name, skills, experience, education, projects, certifications) |
| 📋 **JD parsing** | Paste or upload a job description → required/preferred skills, responsibilities, qualifications |
| 🎯 **Match score** | 0–100%, blending semantic similarity, skill overlap, and experience fit |
| 🧩 **Skill gaps** | Every skill sorted into Strong Match / Partial Match / Missing |
| ❓ **Interview questions** | Technical, behavioral, role-specific, and skill-gap questions with grading notes |
| ✅ **Recommendation** | Strongly Recommend → Not Recommended, with plain-language reasoning |
| 🤖 **Chat agent** | Ask follow-up questions; it decides which tool to call for you |
| 📥 **PDF report** | One-click export of the whole analysis |

---

## Quickstart

```bash
git clone https://github.com/ROHITH-0211/AI-HR-recruitment-assistant
cd ai-hr-recruitment-assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add a free key to `.env`:

```
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nex-agi/nex-n2.5-mini:free
```

Grab one at https://openrouter.ai/keys — takes under a minute, no card
needed.

Then launch it:

```bash
streamlit run app.py
```

Streamlit opens at `http://localhost:8501`. First launch also downloads a
small local embedding model (~90MB) for the knowledge base — one-time cost.

Want to try it without your own files? `data/sample_data/` has a sample
resume and job description you can paste straight in.

---

## If your free model stops working

Free-tier model availability on OpenRouter shifts over time. If you hit a
`model not found` or `could not parse JSON` error, do one of these:

1. Swap in any other `:free` model from https://openrouter.ai/models —
   just make sure its "supported parameters" list includes **tools** (for
   the chat agent) and **structured outputs** (for reliable parsing), or
2. Set `OPENROUTER_MODEL=openrouter/free` — an auto-router that always
   picks a currently-live free, tool-capable model for you.

---

## Using the app

1. Drop in a resume (PDF/DOCX).
2. Paste or upload the job description.
3. Hit **Analyze Candidate**.
4. Work through the tabs:
   - **Overview** — the headline match score + AI explanation
   - **Skill Comparison** — the full gap table
   - **Interview Questions** — ready to use in the actual interview
   - **Recommendation** — the AI's final call, with reasoning
   - **Ask the Agent** — free-form follow-up chat
   - **Report** — download everything as a PDF

---

## How it fits together

The Streamlit UI hands the resume and JD off to an **agent** built on
OpenRouter's tool-calling. The agent doesn't just answer directly — it
decides, per question, whether it needs to re-run the match, pull from the
knowledge base, regenerate interview questions, or produce a fresh
recommendation, and calls the matching module itself.

```
resume/jd files
      │
      ▼
resume_parser.py / jd_parser.py  ──▶  structured JSON
      │
      ▼
matcher.py  ──▶  match score + skill gap categorization
      │
      ▼
agent.py  ──┬──▶  interview_generator.py  (questions, RAG-grounded)
            ├──▶  rag.py                  (knowledge base lookups)
            └──▶  report_generator.py     (final PDF)
```

The knowledge base (`data/knowledge_base/*.txt` — interview guidelines,
skill definitions, best practices, role baselines) is chunked and embedded
locally with Sentence Transformers, stored in ChromaDB, and retrieved only
when something actually needs it — not on every request.

---

## What's in the box

```
ai-hr-recruitment-assistant/
├── app.py                      Streamlit UI
├── src/
│   ├── config.py                env/config loading
│   ├── llm.py                   OpenRouter client + error handling
│   ├── agent.py                 tool-calling agent
│   ├── resume_parser.py         PDF/DOCX → resume JSON
│   ├── jd_parser.py             JD text/file → JD JSON
│   ├── matcher.py                match score, explanation, recommendation
│   ├── skill_extractor.py        embedding-based skill categorization
│   ├── interview_generator.py    question generation
│   ├── report_generator.py       PDF export
│   └── rag.py                    knowledge base ingestion + retrieval
├── data/
│   ├── knowledge_base/           RAG source docs
│   └── sample_data/              sample resume + JD
├── uploads/                     uploaded files land here at runtime
└── reports/                     local report copies (optional)
```

Module cheat-sheet:

| File | Job |
|---|---|
| `config.py` | single source of truth for env vars |
| `llm.py` | one shared OpenRouter client + all LLM error handling |
| `resume_parser.py` | resume → structured JSON |
| `jd_parser.py` | job description → structured JSON |
| `skill_extractor.py` | skill categorization + comparison table |
| `matcher.py` | scoring, explanation, recommendation |
| `rag.py` | chunk / embed / retrieve knowledge base docs |
| `interview_generator.py` | technical / behavioral / role / gap questions |
| `agent.py` | orchestrates everything above via tool-calling |
| `report_generator.py` | builds the PDF |

---

## Built with

Python · Streamlit · OpenRouter (OpenAI-compatible API) · ChromaDB ·
Sentence Transformers (`all-MiniLM-L6-v2`) · PyMuPDF · python-docx ·
ReportLab · python-dotenv · Pandas

---

## Good to know

- **No raw tracebacks.** Bad API keys, rate limits, timeouts, unreadable
  files, empty inputs, and malformed model JSON all surface as plain
  error messages in the UI.
- **Bias guardrails.** Every prompt explicitly tells the model to ignore
  gender, race, religion, age, disability, nationality, and other
  protected characteristics, even if they appear in the resume text.
- **Secrets stay local.** `.env` is git-ignored; only `.env.example`
  (placeholders only) is committed. Switching models is a one-line `.env`
  edit — no code changes.
- **Scanned resumes won't parse.** If a PDF has no real text layer (i.e.
  it's just an image), there's nothing for the parser to extract.
- **This doesn't replace a recruiter.** It's built to speed up screening,
  not to make the hiring call.
