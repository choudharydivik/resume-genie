# main.py - Resume AI Toolkit (Groq Powered)
import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Genie",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

from PIL import Image

logo = Image.open("ResumeGenieLogo.png")
st.sidebar.image(logo, width=80)
st.sidebar.markdown("**Resume Genie**")

GROQ_API_KEY = "API_KEY"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("❌ **GROQ_API_KEY missing**. Add to `.streamlit/secrets.toml` or env vars.")
    st.stop()

@st.cache_resource(show_spinner="🔄 Initializing Groq Model...")
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=2000
    )

llm = get_llm()

# ───────────────────────────────────────────────
# PDF LOADER
# ───────────────────────────────────────────────
@st.cache_data
def extract_resume_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs)
        return text
    finally:
        os.unlink(tmp_path)

# ───────────────────────────────────────────────
# PROMPTS
# ───────────────────────────────────────────────
COVER_LETTER_PROMPT = PromptTemplate.from_template("""
Write a professional cover letter (300–450 words) for this job. Match resume to JD exactly. Standard format.
Job Description: {job_description}
Resume: {resume_text}
Do not invent facts.
""")

RESUME_SCORER_PROMPT = """
You are an expert resume scorer. Analyze match to JD. EXACT structure:
**Score**: X/100
**Overall Match**: X%
Keywords matched: • ...
Missing keywords: • ...
Readability Score: X/100
ATS Compatibility Score: X/100
2-liner summary: ...
Skill gap analysis: • ...
Overall improvement suggestions: • ...
Industry specific feedback: • ...
Job: {job_description}
Resume: {context}
Be honest, use rubrics.
"""

RESUME_CHECKER_PROMPT = PromptTemplate.from_template("""
Score resume standalone (clarity, format, ATS, skills): EXACT structure:
1. **Score**: X/100
2. **Strengths**: • ...
3. **Weaknesses**: • ...
4. **Skills Mentioned**: • ...
5. **Recommended Skills**: • ...
6. **Next Career Steps**: • ...
Resume: {context}
""")

# ───────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────
st.title("🚀 Resume Genie")
st.markdown("""
**Powered by Groq AI** • Your all-in-one solution for job applications  
AI Tools to craft winning resumes, cover letters & career strategies 💼✨
""")

st.sidebar.title("🛠️ Select Tool")
tool = st.sidebar.radio(
    "Choose a service:",
    [
        "✉️ Cover Letter Generator",
        "📊 Resume-JD Matcher",
        "🔍 Resume Checker",
        "💬 Career Coach Chat"
    ],
    index=0
)

# Shared inputs
if tool in ["✉️ Cover Letter Generator", "📊 Resume-JD Matcher"]:
    st.sidebar.subheader("📤 Inputs")
    job_desc = st.sidebar.text_area("Job Description", height=200)
    resume_file = st.sidebar.file_uploader("Resume PDF", type="pdf")

# ───────────────────────────────────────────────
# COVER LETTER TOOL
# ───────────────────────────────────────────────
if tool == "✉️ Cover Letter Generator":

    st.header("✉️ AI Cover Letter Generator")

    col1, col2 = st.columns(2)

    with col1:
        job_description = st.text_area("Paste Job Description", height=350)

    with col2:
        uploaded_file = st.file_uploader("Upload Resume PDF", type="pdf")

        if uploaded_file and st.button("Generate Cover Letter"):

            with st.spinner("Generating..."):

                resume_text = extract_resume_text(uploaded_file)

                chain = COVER_LETTER_PROMPT | llm

                response = chain.invoke({
                    "job_description": job_description,
                    "resume_text": resume_text
                })

                st.markdown(response.content)

# ───────────────────────────────────────────────
# RESUME MATCHER
# ───────────────────────────────────────────────
elif tool == "📊 Resume-JD Matcher":

    st.header("📊 Resume vs Job Description Matcher")

    uploaded_file = st.file_uploader("Upload Resume", type="pdf")
    job_description = st.text_area("Paste Job Description")

    if uploaded_file and job_description and st.button("Analyze"):

        with st.spinner("Analyzing..."):

            context = extract_resume_text(uploaded_file)

            prompt = RESUME_SCORER_PROMPT.format(
                job_description=job_description,
                context=context
            )

            response = llm.invoke(prompt)

            st.markdown(response.content)

# ───────────────────────────────────────────────
# RESUME CHECKER
# ───────────────────────────────────────────────
elif tool == "🔍 Resume Checker":

    st.header("🔍 Resume Evaluation")

    uploaded_file = st.file_uploader("Upload Resume", type="pdf")

    if uploaded_file and st.button("Evaluate Resume"):

        context = extract_resume_text(uploaded_file)

        chain = RESUME_CHECKER_PROMPT | llm

        response = chain.invoke({"context": context})

        st.markdown(response.content)

# ───────────────────────────────────────────────
# CAREER CHAT
# ───────────────────────────────────────────────
elif tool == "💬 Career Coach Chat":

    st.header("💬 Career Coach")

    if "resume_context" not in st.session_state:
        st.session_state.resume_context = None
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload Resume", type="pdf")

    if uploaded_file and st.session_state.resume_context is None:
        context = extract_resume_text(uploaded_file)
        st.session_state.resume_context = context
        st.rerun()

    if not st.session_state.resume_context:
        st.warning("Upload your resume first.")
        st.stop()

    system_msg = SystemMessage(
        content=f"You are a career coach. Resume: {st.session_state.resume_context}"
    )

    for msg in st.session_state.chat_history:

        role = "user" if isinstance(msg, HumanMessage) else "assistant"

        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask about your career..."):

        st.session_state.chat_history.append(HumanMessage(content=prompt))

        messages = [system_msg] + st.session_state.chat_history

        with st.chat_message("assistant"):

            response = llm.invoke(messages)

            st.markdown(response.content)

        st.session_state.chat_history.append(
            AIMessage(content=response.content)
        )

        st.rerun()

# ───────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit + LangChain + Groq")