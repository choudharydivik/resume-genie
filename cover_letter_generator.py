import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import tempfile

# =============================================================================
#   CONFIG
# =============================================================================

GROQ_API_KEY = "GROQ_API_KEY"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

if not GROQ_API_KEY:
    _api_key_missing = True
else:
    _api_key_missing = False


# ────────────────────────────────────────────────
#  LLM (lazy init)
# ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=1500,
    )


# ────────────────────────────────────────────────
#  Prompt
# ────────────────────────────────────────────────

COVER_LETTER_PROMPT = PromptTemplate.from_template(
"""Write a professional, compelling cover letter (300–450 words) tailored specifically to the job description below.
Emphasize the candidate's most relevant experience, skills, achievements and qualifications that directly match or exceed the job requirements.
Use concrete examples from the resume where possible.
Show enthusiasm for the role and company without fabricating information.

Structure the letter in standard business format:
- Header (date, employer's contact if known, or just salutation)
- Opening paragraph: state the position and how you found it + brief why you're a strong fit
- 1–2 body paragraphs: highlight strongest matching qualifications with evidence
- Closing paragraph: reiterate interest, call to action, thanks

Job Description:
{job_description}

Candidate's Resume:
{resume_text}

Do not invent any experience, skills or facts not present in the resume.
"""
)

# =============================================================================
#   FIRST Streamlit command
# =============================================================================

st.set_page_config(
    page_title="AI Cover Letter Generator",
    page_icon="✉️",
    layout="wide"
)

# =============================================================================
#   UI
# =============================================================================

st.title("✉️ AI-Powered Cover Letter Generator")
st.markdown("Upload your resume + paste job description → generate a tailored cover letter.")

if _api_key_missing:
    st.error("GROQ_API_KEY not found.")
    st.stop()

llm = get_llm()

# ─── Layout ─────────────────────────────────────

col1, col2 = st.columns([5,5])

with col1:
    st.subheader("Job Description")
    job_desc = st.text_area(
        "Paste Job Description",
        height=380
    )

with col2:
    st.subheader("Upload Resume (PDF)")
    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"]
    )

    generate_clicked = st.button(
        "Generate Cover Letter",
        type="primary",
        disabled=not (uploaded_file and job_desc.strip())
    )


# ────────────────────────────────────────────────
#   GENERATE
# ────────────────────────────────────────────────

if generate_clicked:

    with st.spinner("Reading resume..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        resume_text = "\n\n".join(doc.page_content for doc in documents)

        os.unlink(tmp_path)

    with st.spinner("Generating cover letter..."):

        chain = COVER_LETTER_PROMPT | llm

        response_container = st.empty()
        full_response = ""

        for chunk in chain.stream({
            "job_description": job_desc,
            "resume_text": resume_text
        }):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_response += content
            response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

        st.success("Cover letter generated!")

        st.download_button(
            label="Download Cover Letter",
            data=full_response,
            file_name="cover_letter.txt",
            mime="text/plain"
        )

st.markdown("---")
st.caption("Powered by Groq Llama model")