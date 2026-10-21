import streamlit as st
import os, base64, time
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# --- LOGO PATH - YOU UPLOADED INSIDE data FOLDER ---
logo_path = "data/abu_logo.png"

# --- WATERMARK ---
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: url('data:image/png;base64,{b64}');
        background-repeat: no-repeat;
        background-position: center;
        background-size: 550px 550px;
        opacity: 0.07;
        z-index: -1;
        pointer-events: none;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- HEADER WITH LOGOS ---
c1, c2, c3 = st.columns([1,4,1])
with c1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
with c2:
    st.markdown("<div style='text-align:center'><b style='color:#c62828;'>STUDENT PROJECT - NOT OFFICIAL ABU WEBSITE</b><br><span style='color:#006400; font-weight:bold; font-size:22px'>ABU Administrative Knowledge Assistant</span><br><span>COEN 872</span></div>", unsafe_allow_html=True)
with c3:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
st.divider()

# --- SESSION ---
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""
if 'selected_doc_index' not in st.session_state:
    st.session_state.selected_doc_index = None
if 'full_content' not in st.session_state:
    st.session_state.full_content = ""
if 'meta' not in st.session_state:
    st.session_state.meta = {}

st.info("Workflow: Query → Embedding (all-MiniLM) → Vector Search (FAISS) → Retrieval → LLM Synthesis (Gemini)")

# --- LOAD DOCS ---
DATA_FOLDER = "data"
docs_text = []
metadatas = []
files_in_data = []

if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.endswith((".pdf",".docx")) and f != ".keep" and f != "abu_logo.png":
            files_in_data.append(f)

files_in_data = files_in_data[:5]

if files_in_data:
    for file_name in files_in_data:
        file_path = os.path.join(DATA_FOLDER, file_name)
        try:
            if file_name.endswith(".pdf"):
                reader = PdfReader(file_path)
                for idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        docs_text.append(text)
                        metadatas.append({"source": file_name, "page": idx+1})
            else:
                doc_file = docx.Document(file_path)
                full_text = "\n".join([p.text for p in doc_file.paragraphs if p.text.strip()])
                if full_text:
                    docs_text.append(full_text)
                    metadatas.append({"source": file_name, "page": 1})
        except Exception as e:
            st.error(f"Error {file_name}: {e}")

    if docs_text:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.create_documents(docs_text, metadatas=metadatas)

        @st.cache_resource
        def get_vectorstore(_chunks):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return FAISS.from_documents(_chunks, embeddings)

        vectorstore = get_vectorstore(chunks)

        st.markdown("### Ask Your Question")
        examples = ["How do I register courses?", "Graduation requirements?", "Late registration?", "Examination regulations?", "How to pay school fees?"]
        cols = st.columns(3)
        for idx, q in enumerate(examples):
            if cols[idx % 3].button(q, key=f"q{idx}"):
                st.session_state.selected_query = q

        user_query = st.text_input("Enter question:", value=st.session_state.selected_query)

        if user_query:
            start = time.time()
            results = vectorstore.similarity_search_with_score(user_query, k=3)
            elapsed = time.time() - start
            st.write(f"Query: {user_query} | Time: {elapsed:.2f}s")

            for idx, (doc, score) in enumerate(results):
                relevance = max(0, 1-score)*100
                with st.container(border=True):
                    st.write(f"Result {idx+1} | {doc.metadata.get('source')} | Page {doc.metadata.get('page')} | {relevance:.1f}%")
                    st.write(doc.page_content[:400] + "...")
                    if st.button("View Full", key=f"vf_{idx}"):
                        st.session_state.selected_doc_index = idx
                        st.session_state.full_content = doc.page_content
                        st.session_state.meta = doc.metadata

        if st.session_state.selected_doc_index is not None:
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Full Content (Raw from PDF)")
                st.success(st.session_state.full_content)
            with col2:
                st.subheader("RAG Answer (Gemini Summary)")
                try:
                    from google import genai
                    gemini_key = st.secrets["GEMINI_API_KEY"]
                    client = genai.Client(api_key=gemini_key)
                    context = st.session_state.full_content
                    query = user_query
                    prompt = f"You are ABU Assistant. Summarize in your own words, student-friendly. Question: {query} Context: {context} Cite [Source: {st.session_state.meta.get('source')} Page {st.session_state.meta.get('page')}]"
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Gemini error: {e}")
                    st.info(st.session_state.full_content[:800])
            if st.button("Clear"):
                st.session_state.selected_doc_index = None
                st.rerun()
