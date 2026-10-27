import streamlit as st
import os, time
from PyPDF2 import PdfReader
from google import genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# --- LOGOS - FIXED TO FIND YOUR LOGO IN data/ FOLDER ---
col_logo1, col_title, col_logo2 = st.columns([1,2.5,1])
with col_logo1:
    # Check all possible places where your logo is
    logo_paths = ["abu_logo.png", "data/abu_logo.png", "assets/abu_logo.png", "ABU_logo.png"]
    for lp in logo_paths:
        if os.path.exists(lp):
            st.image(lp, width=120)
            break
    else:
        st.markdown("**ABU**")

with col_title:
    st.title("ABU RAG Assistant")
    st.markdown("**COEN 872 - AI Project**")

with col_logo2:
    coen_paths = ["coen_logo.png", "data/coen_logo.png", "abu_logo.png", "data/abu_logo.png"]
    for lp in coen_paths:
        if os.path.exists(lp):
            st.image(lp, width=120)
            break
    else:
        st.markdown("**COEN**")

# --- GEMINI ---
if "GEMINI_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("Add GEMINI_API_KEY in Secrets")
    st.stop()

def get_gemini_summary(context, question):
    prompt = f"""ABU Zaria Admin Assistant. Q: {question}. Context: {context[:6000]}. Summarize in 3 short bullets, mention Statute, simple English. DO NOT copy raw text."""
    models = ["gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-flash-latest"]
    for model_name in models:
        try:
            time.sleep(1)
            response = client.models.generate_content(model=model_name, contents=prompt)
            return response.text
        except Exception as e:
            if "503" in str(e):
                time.sleep(2)
                continue
            continue
    return "⚠️ Gemini busy (503). Click Rerun button - Raw PDF on left still works!"

# --- LOAD PDFs (Skip .png files) ---
DATA_FOLDER = "data"
all_docs = []
if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.lower().endswith(".pdf"):  # Only PDFs, ignore .png
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
        if "busy" in summary or "503" in summary:
            st.warning(summary)
            if st.button("🔄 Try Again"):
                st.rerun()
        else:
            st.success(summary)
