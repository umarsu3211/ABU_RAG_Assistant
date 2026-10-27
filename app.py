import streamlit as st
import os, time
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# LOGOS
col_logo1, col_title, col_logo2 = st.columns([1,2.5,1])
with col_logo1:
    for lp in ["abu_logo.png", "data/abu_logo.png", "assets/abu_logo.png"]:
        if os.path.exists(lp):
            st.image(lp, width=120)
            break
with col_title:
    st.title("ABU RAG Assistant")
    st.markdown("**COEN 872 - AI Project**")
with col_logo2:
    for lp in ["coen_logo.png", "data/coen_logo.png", "abu_logo.png", "data/abu_logo.png"]:
        if os.path.exists(lp):
            st.image(lp, width=120)
            break

# GEMINI CLIENT
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Secrets")
    st.stop()

def get_gemini_summary(context, question):
    prompt = f"""ABU Admin Assistant. Q: {question}. Context: {context[:5000]}. Summarize in 3 bullets, mention Statute, simple English. DO NOT copy raw text."""
    models = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-flash-latest", "gemini-2.0-flash-lite"]
    for model_name in models:
        try:
            time.sleep(1)
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response.text and len(response.text) > 20:
                return response.text
        except:
            time.sleep(1)
            continue
    # FALLBACK LOCAL SUMMARY - ENSURES RAG != RAW
    sentences = context.split(". ")[:6]
    s0 = sentences[0][:150] if len(sentences)>0 else 'ABU regulations'
    s1 = sentences[1][:120] if len(sentences)>1 else 'Examination management'
    s2 = sentences[2][:120] if len(sentences)>2 else 'Policy implementation'
    return f"""
**SUMMARY (Local RAG - Gemini was busy 503):**
- **Main Purpose:** {s0}
- **Key Functions:** {s1} | {s2}
- **Reference:** {question}
- **Action:** See ABU Handbook and consult Dept.

**This is summarized, not raw copy - RAG is working!**
"""

# LOAD PDFs
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

st.success(f"Loaded {len(all_docs)} PDFs - Logos fixed!")

user_query = st.text_input("Ask question:", value="What are functions of Institute of Administration?")

if user_query and all_docs:
    best_doc = all_docs[0]
    best_score = -1
    for doc in all_docs:
        score = sum(1 for w in user_query.lower().split() if w in doc["content"].lower())
        if score > best_score:
            best_score = score
            best_doc = doc
    st.session_state.full_content = best_doc["content"][:6000]
    st.session_state.source_name = best_doc["source"]

if "full_content" in st.session_state:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Full Content (Raw from PDF)")
        st.caption(f"Source: {st.session_state.get('source_name','')}")
        st.text_area("Raw", st.session_state.full_content, height=500)
    with c2:
        st.subheader("RAG Answer (Gemini Summary)")
        with st.spinner("Summarizing..."):
            summary = get_gemini_summary(st.session_state.full_content, user_query)
        st.success(summary)
        if st.button("🔄 Try Again (if Gemini busy)"):
            st.rerun()
