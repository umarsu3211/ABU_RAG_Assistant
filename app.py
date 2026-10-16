import streamlit as st
import PyPDF2

st.set_page_config(page_title="ABU Assistant", layout="wide")
st.markdown("<style>.big-text{font-size:24px !important}</style>", unsafe_allow_html=True)

st.title("ABU Registry Assistant")
st.caption("Student Project - Not Official | ABU Zaria")

uploaded = st.file_uploader("Upload ABU PDF Handbook", type=["pdf"])
pdf_text = ""

if uploaded:
    reader = PyPDF2.PdfReader(uploaded)
    pdf_text = " ".join([p.extract_text() or "" for p in reader.pages])
    st.success(f"PDF Loaded: {len(reader.pages)} pages")

q = st.text_input("Ask question", placeholder="e.g. Brief history of ABU?")
if q and pdf_text:
    q_low = q.lower()
    parts = pdf_text.split(".")
    found = [p.strip() for p in parts if q_low in p.lower()]
    if found:
        for ff in found[:8]:
            st.markdown(f'<p class="big-text">{ff}.</p>', unsafe_allow_html=True)
    else:
        st.write(pdf_text[:3000])
elif pdf_text:
    st.write(pdf_text[:3000])
