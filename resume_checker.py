# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import os
import tempfile

# ───────────────────────────────────────────────
#  Config
# ───────────────────────────────────────────────
st.set_page_config(page_title="Resume Checker (Groq)", layout="wide")

GROQ_API_KEY = "YOUR_GROQ_API_KEY"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables or st.secrets.")
    st.stop()

# ───────────────────────────────────────────────
#  LLM
# ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Initializing Groq model...")
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=2000,
    )

llm = get_llm()

# ───────────────────────────────────────────────
#  Prompt (same as yours)
# ───────────────────────────────────────────────
EVAL_PROMPT = """
You are an advanced resume evaluation assistant. Analyze the provided resume text and 
score it out of 100 based on the following criteria: clarity, relevance, format, comprehensiveness, and keywords/ATS-friendliness.

Your response MUST follow this exact structure:

1. **Score**: X/100  
2. **Strengths**:  
   • point one  
   • point two  
   • point three (minimum 3)  
3. **Weaknesses / Areas for Improvement**:  
   • point one  
   • point two  
   • point three (minimum 3)  
4. **Skills Explicitly Mentioned**:  
   • skill 1  
   • skill 2  
   ...  
5. **Recommended Additional Skills**: (to make the resume stronger / more ATS-friendly / future-proof)  
   • suggestion 1  
   • suggestion 2  
   ...  
6. **Suggested Next Career Steps / Roles**:  
   • realistic next role 1  
   • realistic next role 2  
   • longer-term direction (optional)

Be specific, honest, constructive and professional.  
Resume text:  
{context}
"""

prompt_template = PromptTemplate(
    input_variables=["context"],
    template=EVAL_PROMPT
)

# ───────────────────────────────────────────────
#  UI
# ───────────────────────────────────────────────
st.title("📄 Resume Checker powered by Groq")
st.markdown("Upload your resume (PDF) → get a detailed score & improvement suggestions")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        accept_multiple_files=False
    )

    evaluate_button = st.button(
        "Evaluate Resume",
        type="primary",
        disabled=not uploaded_file
    )

if evaluate_button and uploaded_file:
    with st.spinner("Reading PDF... → Extracting text... → Evaluating resume..."):
        try:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            context = "\n\n".join(doc.page_content for doc in documents)

            os.unlink(tmp_path)

            if not context.strip():
                st.error("No readable text was extracted from the PDF.")
                st.stop()

            chain = prompt_template | llm
            response = chain.invoke({"context": context})

            st.subheader("Evaluation Result")
            st.markdown(response.content)

        except Exception as e:
            st.error("An error occurred during processing.")
            st.exception(e)

st.markdown("---")
st.caption("Built with Streamlit + LangChain + Groq")