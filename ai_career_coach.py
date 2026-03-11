import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
import tempfile

# Set up the Streamlit app
st.title("Resume-Based Career Coach Chatbot")

GROQ_API_KEY = "GROQ_API_KEY"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in your environment.")
    st.stop()

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0.2
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "resume_context" not in st.session_state:
    st.session_state.resume_context = None


# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    context = "\n\n".join(doc.page_content for doc in documents)
    st.session_state.resume_context = context

    os.unlink(temp_path)

    st.success("Resume uploaded and processed!")

# Stop if resume not uploaded
if not st.session_state.resume_context:
    st.info("Please upload a resume to start the chatbot.")
    st.stop()

# System prompt
system_message = SystemMessage(
    content=f"""
You are a professional career coach and resume mentor.

You help with:
- Career Guidance
- Resume Improvements
- Interview Preparation
- Job Search Strategy
- Skill Gap Analysis

Candidate Resume:
{st.session_state.resume_context}
"""
)

# Layout
left_col, right_col = st.columns(2)

# Resume viewer
with left_col:
    st.subheader("Your Resume")
    with st.expander("View Resume Content", expanded=True):
        st.text_area(
            "Resume Text",
            st.session_state.resume_context,
            height=600,
            disabled=True
        )

# Chatbot
with right_col:

    st.subheader("Chat with Career Coach")

    for message in st.session_state.chat_history:

        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    user_input = st.chat_input("Ask a question about your career or resume...")

    if user_input:

        st.session_state.chat_history.append(HumanMessage(content=user_input))

        messages = [system_message] + st.session_state.chat_history

        with st.chat_message("assistant"):

            response_placeholder = st.empty()
            response_text = ""

            for chunk in chat.stream(messages):

                content = chunk.content if hasattr(chunk, "content") else str(chunk)

                response_text += content
                response_placeholder.markdown(response_text + "▌")

            response_placeholder.markdown(response_text)

        st.session_state.chat_history.append(AIMessage(content=response_text))

        st.rerun()