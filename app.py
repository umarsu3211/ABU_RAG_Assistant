import streamlit as st
import os, base64, time
from PyPDF2 import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

st.set_page_config(page_title="ABU RAG - COEN 872", layout="wide")

logo_path = "data/abu_logo.png"

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
        opacity: 0.08;
        z-index: -1;
        pointer-events: none;
    }}
    </style>
    """, unsafe_allow_html=True)

c1, c2, c3 = st.columns([1,4,1])
with c1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
with c2:
    st.markdown("<div style='text-align:center'><b style='color:#006400; font-size:24px'>ABU RAG Assistant - COEN 872 Project</b></div>", unsafe_allow_html=True)
with c3:
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)

st.divider()

if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""
if 'selected_doc_index' not in st.session_state:
    st.session_state.selected_doc_index = None
if 'full_content' not in st.session_state:
    st.session_state.full_content = ""
if 'meta' not in st.session_state:
    st.session_state.meta = {}

st.info("Workflow: Query -> Embedding -> Vector Search -> Retrieval -> Gemini Summary")

DATA_FOLDER = "data"
docs_text = []
metadatas = []
files_in_data = []

if os.path.exists(DATA_FOLDER):
    for f in os.listdir(DATA_FOLDER):
        if f.endswith((".pdf",".docx")) and f != ".keep":
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
                full_text = "\n".join([p.text for p in doc_file.paragraphs])
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
    examples = ["How do I register courses?", "Graduation requirements?", "Exam rules?"]
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
                st.write(f"Result {idx+1} | {doc.metadata.get('source')} | Relevance: {relevance:.1f}%")
                st.write(doc.page_content[:400] + "....")
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
            context = st.session_state.full_content
            
            # FIXED GEMINI FUNCTION
            def get_gemini_summary(context, question):
                if "GEMINI_API_KEY" not in st.secrets:
                    return "GEMINI_API_KEY not found in Streamlit Secrets!"
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                
                prompt = f"You are ABU Administrative Assistant. Summarize based on context. Question: {question} Context: {context} Give concise summary with statute reference."
                
                models_to_try = [
                    "gemini-2.5-flash",
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-flash-002",
                    "gemini-pro"
                ]
                for model_name in models_to_try:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        return response.text
                    except Exception as e:
                        continue
                return context[:1500]

            gemini_answer = get_gemini_summary(context, user_query)
            st.success(gemini_answer)
