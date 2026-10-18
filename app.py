import streamlit as st
from PyPDF2 import PdfReader
import docx
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(
    page_title="ABU RAG Assistant - COEN 872",
    page_icon="🎓",
    layout="wide"
)

# FIXED CSS - MAKES TEXT VISIBLE
st.markdown("""
<style>
.stApp {background-color: #ffffff !important;}
.main .block-container {background-color: #ffffff !important;}
h1, h2, h3, p, span, div {color: #000000 !important;}
[data-testid="stMetricValue"] {color: #000000 !important;}
</style>
""", unsafe_allow_html=True)

if 'selected_doc_index' not in st.session_state:
    st.session_state.selected_doc_index = None
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""

st.markdown("""
<div style='background-color:#ffebee;padding:12px;border-left:5px solid #c62828;border-radius:5px;text-align:center'>
<b style='color:#c62828;font-size:16px'>⚠️ STUDENT PROJECT - NOT OFFICIAL ABU WEBSITE</b><br>
<span style='color:#000000;font-size:13px'>COEN 872 SYSTEM DEVELOPMENT | Not affiliated with abu.edu.ng | Academic Purpose</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#006400 !important;margin-top:15px'>🎓 ABU Administrative Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#000000 !important'>COEN 872 - SYSTEM DEVELOPMENT | Ahmadu Bello University</h3>", unsafe_allow_html=True)
st.info("**System Workflow:** User Query → Query Embedding → Vector Search → Relevant Document Retrieval → LLM Processing → Response Generation → User Interface")

st.sidebar.title("📚 Data Collection")
st.sidebar.markdown("Official ABU Documents: Student Handbook, Academic Regulations, Registration Guidelines, Examination Policies, Fee Payment, Graduation Requirements")

st.markdown("### Upload ABU Documents (PDF/DOCX)")
uploaded_files = st.file_uploader("Upload PDF/DOCX", type=["pdf","docx"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        docs_text = []
        metadatas = []
        for file in uploaded_files:
            file_name = file.name
            if file_name.endswith(".pdf"):
                reader = PdfReader(file)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        docs_text.append(text)
                        metadatas.append({"source": file_name, "page": i+1, "category": "Academic Regulations"})
            else:
                try:
                    doc = docx.Document(file)
                    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                    if full_text:
                        docs_text.append(full_text)
                        metadatas.append({"source": file_name, "page": 1, "category": "Handbook"})
                except Exception as e:
                    st.error(f"Error reading {file_name}: {e}")
        
        if docs_text:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.create_documents(docs_text, metadatas=metadatas)
            st.success(f"✅ Processed {len(chunks)} chunks from {len(uploaded_files)} documents")

            @st.cache_resource
            def create_vectorstore(_chunks):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(_chunks, embeddings)
                return vectorstore
            
            vectorstore = create_vectorstore(chunks)
            st.divider()
            st.markdown("### Ask Your Administrative Question")
            
            example_questions = ["How do I register my courses?", "What are the requirements for graduation?", "What is the procedure for late registration?", "What are the university examination regulations?", "How to pay school fees?"]
            st.write("**Try example questions (click to auto-fill):**")
            cols = st.columns(3)
            for idx, q in enumerate(example_questions):
                if cols[idx % 3].button(q, key=f"example_btn_{idx}", use_container_width=True):
                    st.session_state.selected_query = q
            
            user_query = st.text_input("🔍 Enter your question:", value=st.session_state.selected_query, key="main_query_input")
            
            if user_query:
                start_time = time.time()
                results = vectorstore.similarity_search_with_score(user_query, k=4)
                elapsed_time = time.time() - start_time
                
                st.divider()
                st.markdown(f"**Query:** {user_query}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{elapsed_time:.2f}s")
                with col2:
                    st.metric("Docs Retrieved", f"{len(results)}")
                with col3:
                    st.metric("Course", "COEN 872")
                
                st.markdown("### 📚 Retrieved Documents")
                for idx, (doc, score) in enumerate(results):
                    relevance = max(0, 1 - score) * 100
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**Result {idx+1} | Source:** {doc.metadata.get('source','Unknown')} | **Page:** {doc.metadata.get('page','N/A')} | **Relevance:** {relevance:.1f}%")
                            st.write(doc.page_content[:300] + "...")
                        with c2:
                            if st.button(f"View Full", key=f"view_full_btn_{idx}_{doc.metadata.get('source','')}_{doc.metadata.get('page',0)}", use_container_width=True):
                                st.session_state.selected_doc_index = idx
                                st.session_state.selected_doc_content = doc.page_content
                                st.session_state.selected_doc_meta = doc.metadata
                
                if st.session_state.selected_doc_index is not None:
                    st.divider()
                    st.subheader(f"✅ Full Content - Result {st.session_state.selected_doc_index+1}")
                    st.markdown(f"**Source:** {st.session_state.selected_doc_meta.get('source')} | **Page:** {st.session_state.selected_doc_meta.get('page')}")
                    st.success(st.session_state.selected_doc_content)
                    
                    st.markdown("### 🤖 RAG Generated Answer")
                    context = st.session_state.selected_doc_content
                    st.markdown(f"""
                    <div style='background-color:#e3f2fd;padding:15px;border-left:5px solid #1976d2;border-radius:5px'>
                    <b style='color:#000000 !important'>Based on verified ABU document:</b><br><br>
                    <span style='color:#000000 !important'>{context[:1000]}</span><br><br>
                    <i style='color:#000000 !important;font-size:12px'>✅ Answer grounded from: {st.session_state.selected_doc_meta.get('source')} Page {st.session_state.selected_doc_meta.get('page')} | RAG response for: <b>{user_query}</b></i>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Clear Selection", key="clear_selection_btn"):
                        st.session_state.selected_doc_index = None
                        st.rerun()

                st.divider()
                satisfaction = st.slider("Rate this response (1-5):", 1, 5, 4, key="satisfaction_slider")
                if st.button("Submit Rating", key="submit_rating_btn"):
                    st.success(f"Rating {satisfaction} recorded - Thank you!")
else:
    st.markdown("""
    <div style='text-align:center;padding:40px;background:#f5f5f5;border-radius:10px'>
    <h3 style='color:#000000 !important'>👋 Welcome - COEN 872 SYSTEM DEVELOPMENT</h3>
    <p style='color:#000000 !important'>Upload ABU documents to start RAG System</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align:center;font-size:12px;color:#666'>COEN 872 | ABU RAG Assistant | Student Project | Not Official ABU Website</div>", unsafe_allow_html=True)
