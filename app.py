import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

# WHITE BACKGROUND FIX
st.markdown("""
<style>
.stApp {background-color: #FFFFFF !important;}
h1,h2,h3,p,div,span,b,label {color:#000000 !important;}
</style>
""", unsafe_allow_html=True)

if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""
if 'selected_doc_index' not in st.session_state:
    st.session_state.selected_doc_index = None

# HEADER WITH ABU LOGOS BOTH SIDES
st.markdown("""
<div style='display:flex; justify-content:space-between; align-items:center; background:#f9f9f9; padding:10px; border-radius:10px; border:2px solid #006400'>
<div><img src='https://upload.wikimedia.org/wikipedia/en/2/29/Ahmadu_Bello_University_logo.png' width='65'></div>
<div style='text-align:center; flex:1'>
<b style='color:#c62828 !important;'>STUDENT PROJECT - NOT OFFICIAL ABU WEBSITE</b><br>
<span style='color:#006400 !important; font-weight:bold; font-size:22px'>ABU Administrative Knowledge Assistant</span><br>
<span style='color:#000 !important;'>COEN 872 - Ahmadu Bello University</span>
</div>
<div><img src='https://upload.wikimedia.org/wikipedia/en/2/29/Ahmadu_Bello_University_logo.png' width='65'></div>
</div>
""", unsafe_allow_html=True)

st.info("Workflow: User Query -> Embedding -> Vector Search -> Retrieval -> Response")

# PERMANENT STORAGE
DATA_FOLDER = "data"
docs_text = []
metadatas = []
files_in_data = []

if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.endswith((".pdf",".docx")) and f != ".keep":
            files_in_data.append(f)

# LIMIT TO 5 FILES TO AVOID THROTTLING
files_in_data = files_in_data[:5]

if files_in_data:
    st.success(f"Found {len(files_in_data)} permanent ABU documents - PERMANENT STORAGE")
    with st.expander("Click to view loaded documents (broken lines)", expanded=False):
        for i, fname in enumerate(files_in_data):
            st.write(f"{i+1}. {fname}")
    
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
    st.success(f"Processed {len(chunks)} chunks - Ready!")
    
    @st.cache_resource
    def get_vectorstore(_chunks):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(_chunks, embeddings)
    
    vectorstore = get_vectorstore(chunks)
    
    st.markdown("### Ask Your Question")
    examples = ["How do I register courses?", "Graduation requirements?", "Late registration procedure?", "Examination regulations?", "How to pay school fees?"]
    cols = st.columns(3)
    for idx, q in enumerate(examples):
        if cols[idx % 3].button(q, key=f"q{idx}"):
            st.session_state.selected_query = q
    
    user_query = st.text_input("Enter question:", value=st.session_state.selected_query)
    
    if user_query:
        start = time.time()
        results = vectorstore.similarity_search_with_score(user_query, k=3)
        elapsed = time.time() - start
        
        st.write(f"Query: {user_query} | Time: {elapsed:.2f}s | Docs: {len(results)}")
        
        for idx, (doc, score) in enumerate(results):
            relevance = max(0, 1-score)*100
            with st.container(border=True):
                st.write(f"Result {idx+1} | {doc.metadata.get('source')} | Page {doc.metadata.get('page')} | Relevance {relevance:.1f}%")
                st.write(doc.page_content[:400] + "...")
                if st.button("View Full", key=f"vf_{idx}"):
                    st.session_state.selected_doc_index = idx
                    st.session_state.full_content = doc.page_content
                    st.session_state.meta = doc.metadata
        
        if st.session_state.selected_doc_index is not None:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Full Content")
        st.write(f"Source: {st.session_state.meta.get('source')} | Page {st.session_state.meta.get('page')}")
        st.success(st.session_state.full_content)

    with col2:
        st.subheader("RAG Answer (Gemini Synthesis)")
        # --- GEMINI RAG CODE STARTS HERE Line 102 ---
        try:
            import google.generativeai as genai
            
            gemini_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # This is the RAG prompt - f"Context:" is here
            context = st.session_state.full_content
            query = user_query
            
            prompt = f"""You are ABU Assistant.
            Summarize the answer from context in your own words, don't copy verbatim.
            Be student-friendly.
            Question: {query}
            Context: {context}
            Answer concisely with citation [Source: {st.session_state.meta.get('source')} Page {st.session_state.meta.get('page')}]
            """
            
            response = model.generate_content(prompt)
            st.info(response.text)
            
        except Exception as e:
            st.error(f"Gemini not connected: {e}")
            st.info(st.session_state.full_content[:800] + " (Fallback)")
        # --- GEMINI RAG CODE ENDS ---

    if st.button("Clear"):
        st.session_state.selected_doc_index = None
        st.rerun()
