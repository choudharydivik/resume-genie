# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

# ───────────────────────────────────────────────
#   CONFIG
# ───────────────────────────────────────────────
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in secrets or environment.")
    st.stop()

# The template we refined earlier
PROMPT_TEMPLATE = """You are an expert resume scorer and ATS optimization specialist with deep knowledge of recruitment practices across industries.

Task: Carefully analyze how well the candidate's resume matches the job description below. Base EVERY statement, score, and suggestion **strictly and exclusively** on the content actually present in the provided resume and job description. Do NOT invent, assume, or add any experience, skills, tools, achievements, or facts that are not explicitly written in the resume.

Job Description:
{job_description}

Candidate's Resume:
{context}

Produce the analysis using **exactly** the following structure and headings (do not add/remove sections, do not change headings):

Score: [integer]/100  
Overall Match: [integer]%  

Keywords matched:  
• [bullet list of important keywords/phrases from JD that DO appear in the resume]  

Missing keywords:  
• [bullet list of important/hard-required keywords/phrases from JD that are completely absent or extremely weakly represented in the resume]  

Readability Score: [integer]/100  
ATS Compatibility Score: [integer]/100  

2-liner summary:  
[One strong, concise sentence summarizing the overall fit]  
[One strong, concise sentence naming the single biggest current weakness]

Skill gap analysis:  
• [Bullet points – clear skill/tool/experience gaps, phrased as "Missing / weak: X → needed for Y part of the role"]  

Overall improvement suggestions:  
• [Prioritized, actionable bullet points]

Industry specific feedback:  
• [2–5 bullets tailored to this role’s industry]

Scoring rubrics to follow (use your judgment applying these):
• Score (0–100)
• Overall Match %
• Readability
• ATS Compatibility

Be honest, direct, and constructive.
"""

# ───────────────────────────────────────────────
#   STREAMLIT APP
# ───────────────────────────────────────────────

st.set_page_config(page_title="Resume Scorer", layout="wide")

st.title("📄 Resume Matcher & Scorer")
st.markdown("Upload your resume (PDF) and paste the job description to get a detailed match analysis powered by Groq.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the full job description here",
        height=320,
        placeholder="Responsibilities...\nRequirements...\nSkills...\n",
        key="jd_input"
    )

with col2:
    st.subheader("Your Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="resume_uploader")

    if uploaded_file is not None:
        st.success("Resume uploaded ✓")

# ── Analyze button ────────────────────────────────────────

if st.button("Analyze Resume Match", type="primary", disabled=not (uploaded_file and job_description.strip())):

    with st.spinner("Extracting resume text..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            context = "\n\n".join(doc.page_content for doc in documents)

            os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

    if not context.strip():
        st.error("No readable text found in the resume PDF.")
        st.stop()

    prompt = PROMPT_TEMPLATE.format(
        job_description=job_description.strip(),
        context=context.strip()
    )

    with st.spinner("Analyzing resume..."):

        try:
            chat = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=GROQ_API_KEY,
                temperature=0.2,
                max_tokens=2200
            )

            response = chat.invoke(prompt)
            analysis_text = response.content

            st.subheader("📊 Resume Analysis Result")
            st.markdown(analysis_text)

        except Exception as e:
            st.error(f"API error: {str(e)}")
            if "rate limit" in str(e).lower():
                st.warning("Rate limit reached — please wait a few minutes and try again.")
            elif "authentication" in str(e).lower():
                st.error("Invalid or missing GROQ_API_KEY.")