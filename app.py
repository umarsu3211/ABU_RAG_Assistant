import streamlit as st
import os
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# Configure NEW Gemini Client
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Secrets")
    st.stop()

def get_gemini_summary(context, question):
    prompt = f"""
    You are ABU Zaria Admin Assistant for COEN 872.
    Summarize for student - DO NOT copy raw text.

    Question: {question}
    Context: {context[:8000]}

    Give:
    - SUMMARY (3 bullets)
    - REFERENCE (Statute number)
    - ACTION
    """
    try:
        # NEW API - 2026 working models
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e2:
            return f"GEMINI ERROR: {e} | {e2}"

st.title("ABU RAG Assistant - COEN 872")

DATA_FOLDER = "data"
all_docs = []

if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, f)
            try:
                reader = PdfReader(path)
                text = ""
                for page in reader.pages[:10]:
                    text += (page.extract_text() or "") + "\n"
                if len(text) > 100:
                    all_docs.append({"source": f, "content": text})
            except:
                pass

user_query = st.text_input("Ask question:", value="What are functions of Institute of Administration?")

if user_query and all_docs:
    # Simple keyword search
    best_doc = all_docs[0]
    best_score = 0
    for doc in all_docs:
        score = sum(1 for w in user_query.lower().split() if w in doc["content"].lower())
        if score > best_score:
            best_score = score
            best_doc = doc
    
    st.session_state.full_content = best_doc["content"][:4000]

if "full_content" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Full Content (Raw from PDF)")
        st.text_area("Raw", st.session_state.full_content, height=400)
    with col2:
        st.subheader("RAG Answer (Gemini Summary)")
        with st.spinner("Gemini 2.0 is summarizing..."):
            summary = get_gemini_summary(st.session_state.full_content, user_query)
        if "ERROR" in summary:
            st.error(summary)
        else:
            st.success(summary)
