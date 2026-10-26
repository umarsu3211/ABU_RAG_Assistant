import streamlit as st
import os, base64
from PyPDF2 import PdfReader
import google.generativeai as genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# Configure Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Secrets! Must start with AIza...")
    st.stop()

def get_gemini_summary(context, question):
    prompt = f"""
    You are ABU Zaria Administrative Assistant for COEN 872 Project.
    Summarize this document for a student.

    Question: {question}
    Context: {context[:8000]}

    Instructions:
    - Give 3-4 bullet points summary (NOT copy raw text)
    - Mention Statute number if present
    - Use simple English
    - Format: SUMMARY: ... REFERENCE: ... ACTION: ...
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        try:
            model2 = genai.GenerativeModel("gemini-pro")
            response = model2.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"GEMINI ERROR: {str(e)[:500]} | {str(e2)[:300]}"

# UI
st.title("ABU RAG Assistant - COEN 872")
st.info("Workflow: Query -> Keyword Search -> Gemini Summary (No Torch)")

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
            except Exception as e:
                st.error(f"Error reading {f}: {e}")

if not all_docs:
    st.warning("No PDFs found in data folder!")
    st.stop()

# Search
user_query = st.text_input("Ask question:", value="What are functions of Institute of Administration?")

if user_query:
    # Simple keyword search - No FAISS, No Torch!
    best_doc = None
    best_score = 0
    query_words = user_query.lower().split()
    
    for doc in all_docs:
        score = sum(1 for w in query_words if w in doc["content"].lower())
        if score > best_score:
            best_score = score
            best_doc = doc
    
    if best_doc is None:
        best_doc = all_docs[0]

    st.session_state.full_content = best_doc["content"][:3000]
    st.session_state.source = best_doc["source"]

if "full_content" in st.session_state and st.session_state.full_content:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Full Content (Raw from PDF)")
        st.write(f"Source: {st.session_state.get('source','')}")
        st.text_area("Raw", st.session_state.full_content, height=400)
    
    with col2:
        st.subheader("RAG Answer (Gemini Summary)")
        with st.spinner("Gemini is summarizing..."):
            summary = get_gemini_summary(st.session_state.full_content, user_query)
        st.success(summary)
        
        # Debug info
        if "GEMINI ERROR" in summary or summary == st.session_state.full_content:
            st.error("Gemini failed! Check API key must be AIza... not AQ...")
        else:
            st.caption("✅ Gemini summary working! Different from raw text")
