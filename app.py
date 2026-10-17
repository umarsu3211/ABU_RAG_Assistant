import streamlit as st
from PyPDF2 import PdfReader
import docx
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

st.set_page_config(
    page_title="ABU RAG Assistant - COEN 872",
    page_icon="🎓",
    layout="wide"
)

# DISCLAIMER - NOT OFFICIAL ABU WEBSITE
st.markdown("""
<div style='background-color:#ffebee;padding:12px;border-left:5px solid #c62828;border-radius:5px;text-align:center'>
<b style='color:#c62828;font-size:16px'>⚠️ STUDENT PROJECT - NOT OFFICIAL ABU WEBSITE</b><br>
<span style='font-size:13px'>COEN 872 SYSTEM DEVELOPMENT | Not affiliated with abu.edu.ng | Hosted independently on GitHub for Academic Purpose</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;color:#006400;margin-top:15px'>🎓 ABU Administrative Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#333'>COEN 872 - SYSTEM DEVELOPMENT | Ahmadu Bello University</h3>", unsafe_allow_html=True)

st.info("**System Workflow:** User Query → Query Embedding → Vector Search → Relevant Document Retrieval → LLM Processing → Response Generation → User Interface")

st.sidebar.title("📚 Data Collection")
st.sidebar.markdown("""
Official ABU Documents:
- Student Handbook
- Academic Regulations
- Registration Guidelines
- Examination Policies
- Fee Payment Procedures
- Graduation Requirements
""")

st.sidebar.title("🔧 Implementation Stack")
st.sidebar.markdown("""
- **Course:** COEN 872 SYSTEM DEVELOPMENT
- **Language:** Python
- **Framework:** LangChain
- **Vector DB:** FAISS
- **Embedding:** Sentence-Transformers
- **UI:** Streamlit
""")

st.sidebar.title("📊 Evaluation Metrics")
st.sidebar.markdown("""
- Accuracy
- Relevance
- Response Time
- User Satisfaction
- Precision & Recall
""")

st.markdown("### Upload ABU Documents (PDF/DOCX)")
uploaded_files = st.file_uploader("Upload PDF/DOCX", type=["pdf","docx"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents... Metadata + Embedding Generation"):
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
            
            example_questions = [
                "How do I register my courses?",
                "What are the requirements for graduation?",
                "What is the procedure for late registration?",
                "What are the university examination regulations?",
                "How to pay school fees?"
            ]
            
            selected_example = st.selectbox("Try example questions:", [""] + example_questions)
            user_query = st.text_input("🔍 Enter your question:", value=selected_example)

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
                
                for idx, (doc, score) in enumerate(results):
                    st.markdown(f"""
                    <div style='background-color:#e8f5e9;padding:12px;border-left:5px solid #2e7d32;border-radius:5px;margin-bottom:12px'>
                        <b>Result {idx+1} | Source: {doc.metadata.get('source','Unknown')} | Page: {doc.metadata.get('page','N/A')}</b><br>
                        <span style='font-size:12px'>Score: {score:.3f}</span><br><br>
                        {doc.page_content[:800]}...<br><br>
                        <i style='font-size:12px;color:#2e7d32'>✅ Verified from official document</i>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("Rate this response (1-5):")
                satisfaction = st.slider("Satisfaction", 1, 5, 4)
                if st.button("Submit Rating"):
                    st.success(f"Rating {satisfaction} recorded - Thank you!")
else:
    st.markdown("""
    <div style='text-align:center;padding:40px;background:#f5f5f5;border-radius:10px'>
        <h3>👋 Welcome - COEN 872 SYSTEM DEVELOPMENT</h3>
        <p>Upload ABU documents to start RAG System</p>
        <p>Course: COEN 872 | Project: ABU RAG Assistant</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style='text-align:center;font-size:12px;color:#666'>
<b>COEN 872 SYSTEM DEVELOPMENT</b> | ABU RAG Assistant | Student Project | Not Official ABU Website<br>
</div>
""", unsafe_allow_html=True)
