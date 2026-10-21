import streamlit as st, os, base64
st.set_page_config(page_title="ABU RAG", layout="wide")

logo_path = "data/abu_logo.png"

# Header - SAFE - no crash if logo missing
c1,c2,c3 = st.columns([1,4,1])
with c1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with c2:
    st.markdown("<div style='text-align:center'><b style='color:red'>STUDENT PROJECT</b><br><b style='color:green; font-size:20px'>ABU Administrative Knowledge Assistant</b><br>COEN 872</div>", unsafe_allow_html=True)
with c3:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)

# Watermark only if logo exists
if os.path.exists(logo_path):
    with open(logo_path,"rb") as f:
        b64=base64.b64encode(f.read()).decode()
    st.markdown("<style>.stApp::before{content:'';position:fixed;top:0;left:0;width:100%;height:100%;background-image:url('data:image/png;base64,"+b64+"');background-repeat:no-repeat;background-position:center;background-size:500px;opacity:0.07;z-index:-1;pointer-events:none;}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Logo not uploaded yet. Upload abu_logo.png to GitHub to see logos + watermark. App works without it.")

# Rest of your RAG code continues here...
# Keep all your old code from DATA_FOLDER = "data" downwards
