import streamlit as st
import os
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Secrets")
    st.stop()

def get_gemini_summary(context, question):
    prompt = f"""You are ABU Zaria Admin Assistant.
Question: {question}
Context: {context[:6000]}
Task: Summarize in 3 bullet points, mention Statute number, simple English. DO NOT copy raw text."""
    
    models_to_try = [
        "gemini-3.8-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-flash-latest"
    ]
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return f"[{model_name}]\n" + response.text
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str:
                continue  # Try next model if busy
            else:
                continue
    
    return "GEMINI ERROR: All models busy (503). Please click Rerun button to try again! High demand now."

st.title("ABU RAG Assistant - COEN 872")
st.info("Using gemini-3.8-flash - 2026 latest model")

DATA_FOLDER = "data"
all_docs = []

if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, f)
            try:
                reader = PdfReader(path)
                text = ""
                for p in reader.pages[:15]:
                    t = p.extract_text()
                    if t:
                        text += t + "\n"
                if len(text) > 100:
                    all_docs.append({"source": f, "content": text})
            except Exception as e:
                st.write(f"Error {f}: {e}")

if not all_docs:
    st.warning("No PDFs loaded - Check data folder has PDFs")
    st.stop()
else:
    st.success(f"Loaded {len(all_docs)} PDFs")

user_query = st.text_input("Ask question:", value="What are functions of Institute of Administration?")

if user_query:
    best_doc = all_docs[0]
    best_score = -1
    q_words = user_query.lower().split()
    for doc in all_docs:
        score = sum(1 for w in q_words if w in doc["content"].lower())
        if score > best_score:
            best_score = score
            best_doc = doc
    st.session_state.full_content = best_doc["content"][:5000]
    st.session_state.source_name = best_doc["source"]

if "full_content" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Full Content (Raw from PDF)")
        st.caption(f"Source: {st.session_state.get('source_name','')}")
        st.text_area("Raw", st.session_state.full_content, height=500)
    with col2:
        st.subheader("RAG Answer (Gemini Summary)")
        with st.spinner("Gemini 3.8 summarizing..."):
            summary = get_gemini_summary(st.session_state.full_content, user_query)
        if "ERROR" in summary:
            st.error(summary)
        else:
            st.success(summary)
