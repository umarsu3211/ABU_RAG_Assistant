import streamlit as st
import os, time
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# --- LOGOS SECTION ---
col_logo1, col_title, col_logo2 = st.columns([1,3,1])
with col_logo1:
    if os.path.exists("abu_logo.png"):
        st.image("abu_logo.png", width=100)
    elif os.path.exists("assets/abu_logo.png"):
        st.image("assets/abu_logo.png", width=100)
    else:
        st.write("ABU")

with col_title:
    st.title("ABU RAG Assistant - COEN 872")

with col_logo2:
    if os.path.exists("coen_logo.png"):
        st.image("coen_logo.png", width=100)
    elif os.path.exists("assets/coen_logo.png"):
        st.image("assets/coen_logo.png", width=100)
    else:
        st.write("COEN")

# --- GEMINI CLIENT ---
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Secrets")
    st.stop()

def get_gemini_summary(context, question):
    prompt = f"""ABU Zaria Admin Assistant. 
Q: {question}
Context: {context[:6000]}
Summarize in 3 bullets, mention Statute, simple English. DO NOT copy raw text."""
    
    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-flash-latest"]
    
    for model_name in models_to_try:
        try:
            time.sleep(1)  # Avoid rate limit
            response = client.models.generate_content(model=model_name, contents=prompt)
            return f"✅ {model_name}\n\n" + response.text
        except Exception as e:
            if "503" in str(e):
                time.sleep(2)
                continue
            continue
    return "⚠️ Google Gemini is overloaded (503 high demand). Click Rerun button at top right or wait 1 min and refresh. Raw PDF is on left - you can still read it!"

st.info("Using gemini-3.8-flash with auto fallback - 2026 model")

# --- LOAD PDFs ---
DATA_FOLDER = "data"
all_docs = []
if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.lower().endswith(".pdf"):
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
            except:
                pass

if not all_docs:
    st.warning("No PDFs in data/")
    st.stop()
else:
    st.success(f"Loaded {len(all_docs)} PDFs - Raw and RAG will show below")

# --- SEARCH ---
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
    st.session_state.full_content = best_doc["content"]
    st.session_state.source_name = best_doc["source"]

# --- DISPLAY RAW + RAG ---
if "full_content" in st.session_state:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Full Content (Raw from PDF)")
        st.caption(f"Source: {st.session_state.get('source_name','')}")
        st.text_area("Raw Text", st.session_state.full_content[:5000], height=500, key="raw_box")
    with c2:
        st.subheader("RAG Answer (Gemini Summary)")
        with st.spinner("Summarizing..."):
            summary = get_gemini_summary(st.session_state.full_content, user_query)
        if "503" in summary or "overloaded" in summary:
            st.warning(summary)
            st.button("🔄 Rerun / Try Again")
        else:
            st.success(summary)
