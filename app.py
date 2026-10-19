import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="ABU RAG Assistant - COEN 872", page_icon="🎓", layout="wide")

# FIX WHITE BACKGROUND + BLACK TEXT - Visible!
st.markdown("""
<style>
.stApp {background-color: #FFFFFF !important;}
.main .block-container {background-color: #FFFFFF !important;}
h1, h2, h3, p, div, span, label, b, i {color: #000000 !important;}
[data-testid="stMetricValue"] {color: #000000 !important;}
button {background-color: #e8f5e9 !important; color: #000000 !important; border: 1px solid #2e7d32 !important;}
</style>
""", unsafe_allow_html=True)

if 'selected_doc_index' not in st.session_state:
    st.session_state.selected_doc_index = None
if 'selected_doc_content' not in st.session_state:
    st.session_state.selected_doc_content = ""
if 'selected_doc_meta' not in st.session_state:
    st.session_state.selected_doc_meta = {}
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""

st.markdown("""
<div style='background-color:#ffebee;padding:12px;border-left:5px solid #c62828;border-radius:5px;text-align:center'>
<b style='color:#c62828 !important;font-size:16px'>⚠️ STUDENT PROJECT - NOT OFFICIAL ABU WEBSITE</b><br>
<span style='color:#000000 !important;font-size:13px'>COEN 872 SYSTEM DEVELOPMENT | Academic Purpose</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#006400 !important;'>🎓 ABU Administrative Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#000000 !important;'>COEN 872 - Ahmadu Bello University</h3>", unsafe_allow_html=True)
st.info("System Workflow: User Query → Embedding → Vector Search → Retrieval → LLM Processing → Response")

# PERMANENT STORAGE LOGIC
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

st.markdown("### 📁 Permanent Knowledge Base")
docs_text = []
metadatas = []

files_in_data = []
if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f == ".keep":
            continue
        if f.endswith((".pdf",".docx")):
            files_in_data.append(f)

if files_in_data:
st.success(f"✅ {len(files_in_data)} permanent files - PERMANENT STORAGE (Will remain after logout)")
# Show files in clean broken lines, 2 columns
with st.expander(f"📚 View {len(files_in_data)} ABU Documents Loaded", expanded=False):
    cols = st.columns(2)
    for idx, fname in enumerate(files_in_data):
        # Clean name - remove .pdf and shorten
        clean_name = fname.replace('.pdf','').replace('_',' ').replace('-',' ')[:45]
        cols[idx % 2].markdown(f"**{idx+1}.** {clean_name}")
    for file_name in files_in_data:
        file_path = os.path.join(DATA_FOLDER, file_name)
        try:
            if file_name.endswith(".pdf"):
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        docs_text.append(text)
                        metadatas.append({"source": file_name, "page": i+1, "category": "ABU Document"})
            else:
                doc_file = docx.Document(file_path)
                full_text = "\n".join([para.text for para in doc_file.paragraphs if para.text.strip()])
                if full_text:
                    docs_text.append(full_text)
                    metadatas.append({"source": file_name, "page": 1, "category": "ABU Document"})
        except Exception as e:
            st.error(f"Error {file_name}: {e}")
else:
    st.warning("⚠️ No PDFs in data/ folder. Upload PDFs to data/ folder on GitHub to make permanent.")

if docs_text:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents(docs_text, metadatas=metadatas)
    st.success(f"✅ Processed {len(chunks)} chunks from {len(files_in_data)} documents - PERMANENT")

    @st.cache_resource
    def create_vectorstore(_chunks):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(_chunks, embeddings)
        return vectorstore
    
    vectorstore = create_vectorstore(chunks)
    st.divider()

    st.markdown("### Ask Your Administrative Question")
    example_questions = ["How do I register my courses?", "What are the requirements for graduation?", "What is the procedure for late registration?", "What are the university examination regulations?", "How to pay school fees?"]
    cols = st.columns(3)
    for idx, q in enumerate(example_questions):
        if cols[idx % 3].button(q, key=f"ex_{idx}", use_container_width=True):
            st.session_state.selected_query = q
            st.session_state.selected_doc_index = None

    user_query = st.text_input("🔍 Enter your question:", value=st.session_state.selected_query, key="main_q")

    if user_query:
        start_time = time.time()
        results = vectorstore.similarity_search_with_score(user_query, k=4)
        elapsed_time = time.time() - start_time

        st.divider()
        st.markdown(f"**Query:** {user_query}")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Response Time", f"{elapsed_time:.2f}s")
        with c2: st.metric("Docs Retrieved", f"{len(results)}")
        with c3: st.metric("Storage", "Permanent")

        st.markdown("### 📚 Retrieved Documents")
        for idx, (doc, score) in enumerate(results):
            relevance = max(0, 1 - score) * 100
            with st.container(border=True):
                col_a, col_b = st.columns([4,1])
                with col_a:
                    st.markdown(f"**Result {idx+1} | Source:** {doc.metadata.get('source')} | **Page:** {doc.metadata.get('page')} | **Relevance:** {relevance:.1f}%")
                    st.write(doc.page_content[:350] + "...")
                with col_b:
                    if st.button("View Full", key=f"view_full_{idx}_{doc.metadata.get('page')}_{idx}", use_container_width=True):
                        st.session_state.selected_doc_index = idx
                        st.session_state.selected_doc_content = doc.page_content
                        st.session_state.selected_doc_meta = doc.metadata

        if st.session_state.selected_doc_index is not None:
            st.divider()
            st.subheader(f"✅ Full Content - Result {st.session_state.selected_doc_index+1}")
            st.markdown(f"**Source:** {st.session_state.selected_doc_meta.get('source')} | **Page:** {st.session_state.selected_doc_meta.get('page')}")
            st.success(st.session_state.selected_doc_content)

            st.markdown("### 🤖 RAG Generated Answer")
            st.markdown(f"""
            <div style='background-color:#e3f2fd;padding:15px;border-left:5px solid #1976d2;border-radius:5px'>
            <span style='color:#000000 !important;'>{st.session_state.selected_doc_content[:1500]}</span><br><br>
            <i style='color:#000000 !important;font-size:12px'>✅ Grounded from: {st.session_state.selected_doc_meta.get('source')} Page {st.session_state.selected_doc_meta.get('page')} | Query: {user_query}</i>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Clear", key="clear_btn"):
                st.session_state.selected_doc_index = None
                st.rerun()
else:
    st.info("👋 No documents loaded yet. Make sure PDFs are in data/ folder on GitHub.")
