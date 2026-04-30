Perfect — clear direction. I'll craft a polished dark, professional UI with a distinctive identity. I'm calling it **"PRISM"** — *a multimodal RAG assistant* (a prism splits one beam into many — fits text/web/vision beautifully).

Design direction:
- **Fonts**: `Fraunces` (variable serif display) + `Geist` (body) + `JetBrains Mono` (accents) — avoiding the typical Inter/Space Grotesk
- **Palette**: deep near-black `#0A0A0F`, citron-lime accent `#C9F26B`, prismatic source-badge chips (amber for PDF, cyan for URL, magenta for Image)
- **Layout**: asymmetric sidebar with section dividers, glass-morphism cards, generous spacing
- **Chat**: persistent styled bubbles using your existing `st.session_state.chat`, avatars, smooth entrance animation
- **Controls**: Clear chat + Reset knowledge buttons, live source counter badges, hidden debug lines (silent fail-safe still in place)
- **Logic**: 100% untouched — same imports, same RAG flow, same LLM call

Let me create the file.
Action: file_editor create /app/streamlit_app.py --file-text "import streamlit as st

from Loaders.loaders import load_pdf
from Loaders.webbase import load_url
from Loaders.img import load_image
from Processing.chunking import chunk_documents
from vectorstore.vectordb import create_vectorstore, retrieve_docs
from rag_pipeline import generate_answer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / \".env\"
load_dotenv(ENV_PATH)

api_key = os.getenv(\"GEMINI_API_KEY\")

# =========================================================
# PAGE CONFIG  (must be the first Streamlit command)
# =========================================================
st.set_page_config(
    page_title=\"PRISM — Multimodal RAG\",
    page_icon=\"◣\",
    layout=\"wide\",
    initial_sidebar_state=\"expanded\",
)

# =========================================================
# GLOBAL STYLES  —  dark professional, prismatic accents
# =========================================================
st.markdown(
    \"\"\"
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
    <link href=\"https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,800&family=Geist:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap\" rel=\"stylesheet\">

    <style>
      :root{
        --bg:        #0A0A0F;
        --bg-soft:   #11111A;
        --surface:   #15151F;
        --surface-2: #1B1B27;
        --border:    #262633;
        --border-2:  #34344A;
        --text:      #EDEDF2;
        --text-mute: #8A8A9A;
        --text-dim:  #5C5C6E;
        --accent:    #C9F26B;
        --accent-2:  #9FE03A;
        --pdf:       #F7B955;
        --url:       #5BD0E8;
        --img:       #E879C7;
        --danger:    #FF6B6B;
      }

      /* ---------- Reset Streamlit defaults ---------- */
      #MainMenu, footer, header [data-testid=\"stToolbar\"]{ visibility:hidden; }
      .stDeployButton{ display:none; }
      .stApp{
        background:
          radial-gradient(1200px 600px at 85% -10%, rgba(201,242,107,0.06), transparent 60%),
          radial-gradient(900px 500px at -10% 110%, rgba(91,208,232,0.05), transparent 60%),
          var(--bg);
        color: var(--text);
        font-family: 'Geist', system-ui, sans-serif;
      }

      /* subtle grain */
      .stApp::before{
        content:\"\";
        position:fixed; inset:0;
        background-image: url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 0.04 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\");
        opacity:.5; pointer-events:none; z-index:0;
      }
      .block-container{ position:relative; z-index:1; padding-top: 2.2rem; padding-bottom: 8rem; max-width: 1080px; }

      /* ---------- Typography ---------- */
      h1,h2,h3,h4{ font-family:'Fraunces', serif; letter-spacing:-0.02em; color:var(--text); font-weight:600; }
      .mono{ font-family:'JetBrains Mono', monospace; }

      /* ---------- Brand header ---------- */
      .brand{
        display:flex; align-items:center; gap:18px;
        padding: 6px 0 28px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
      }
      .brand-mark{
        width:46px; height:46px; border-radius:12px;
        background: conic-gradient(from 200deg at 50% 50%, #C9F26B, #5BD0E8, #E879C7, #F7B955, #C9F26B);
        position:relative; box-shadow: 0 0 0 1px var(--border-2), 0 8px 30px rgba(201,242,107,0.18);
      }
      .brand-mark::after{
        content:\"\"; position:absolute; inset:6px; border-radius:7px;
        background: var(--bg);
      }
      .brand-mark::before{
        content:\"◣\"; position:absolute; inset:0;
        display:flex; align-items:center; justify-content:center;
        font-size:18px; color:var(--accent); z-index:2; font-weight:700;
      }
      .brand-text h1{
        font-size: 2.6rem; line-height:1; margin:0;
        font-style: italic; font-weight:600;
      }
      .brand-text h1 .accent{ color: var(--accent); font-style: normal; }
      .brand-text p{
        margin: 6px 0 0 0; color: var(--text-mute);
        font-size: 0.92rem; letter-spacing: 0.02em;
      }
      .brand-text p .dot{ color: var(--accent); margin: 0 8px; }

      /* ---------- Sidebar ---------- */
      section[data-testid=\"stSidebar\"]{
        background: linear-gradient(180deg, #0D0D14 0%, #0A0A0F 100%);
        border-right: 1px solid var(--border);
      }
      section[data-testid=\"stSidebar\"] .block-container{ padding-top: 2rem; }
      section[data-testid=\"stSidebar\"] h2,
      section[data-testid=\"stSidebar\"] h3{
        font-family: 'Geist', sans-serif; font-weight:600;
        font-size: 0.78rem; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--text-mute); margin: 0 0 14px 0;
      }
      section[data-testid=\"stSidebar\"] hr{
        border:none; border-top:1px solid var(--border); margin: 22px 0;
      }

      /* radio (source picker) */
      div[role=\"radiogroup\"]{ gap: 6px !important; }
      div[role=\"radiogroup\"] label{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px !important;
        transition: border-color .2s ease, background .2s ease, transform .15s ease;
        cursor: pointer;
      }
      div[role=\"radiogroup\"] label:hover{
        border-color: var(--border-2);
        background: var(--surface-2);
      }
      div[role=\"radiogroup\"] label[data-checked=\"true\"]{
        border-color: var(--accent);
        background: rgba(201,242,107,0.06);
      }

      /* inputs */
      .stTextInput input, .stTextArea textarea{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-family: 'Geist', sans-serif !important;
      }
      .stTextInput input:focus, .stTextArea textarea:focus{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(201,242,107,0.15) !important;
      }

      /* file uploader */
      [data-testid=\"stFileUploaderDropzone\"]{
        background: var(--surface);
        border: 1.5px dashed var(--border-2);
        border-radius: 12px;
        transition: border-color .2s ease, background .2s ease;
      }
      [data-testid=\"stFileUploaderDropzone\"]:hover{
        border-color: var(--accent);
        background: rgba(201,242,107,0.03);
      }
      [data-testid=\"stFileUploaderDropzone\"] *{ color: var(--text-mute) !important; }
      [data-testid=\"stFileUploaderDropzone\"] small{ color: var(--text-dim) !important; }

      /* buttons */
      .stButton > button{
        width: 100%;
        background: var(--accent);
        color: #0A0A0F;
        border: 1px solid var(--accent);
        border-radius: 10px;
        font-family: 'Geist', sans-serif;
        font-weight: 600;
        letter-spacing: 0.01em;
        padding: 10px 16px;
        transition: transform .15s ease, background .2s ease, box-shadow .2s ease;
      }
      .stButton > button:hover{
        background: var(--accent-2);
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(201,242,107,0.25);
      }
      .stButton > button:active{ transform: translateY(0); }

      /* secondary buttons (we tag with kind) */
      .ghost-btn .stButton > button{
        background: transparent;
        color: var(--text-mute);
        border: 1px solid var(--border);
        font-weight: 500;
      }
      .ghost-btn .stButton > button:hover{
        color: var(--text);
        border-color: var(--border-2);
        background: var(--surface);
        box-shadow: none;
      }
      .danger-btn .stButton > button{
        background: transparent;
        color: var(--danger);
        border: 1px solid rgba(255,107,107,0.3);
      }
      .danger-btn .stButton > button:hover{
        background: rgba(255,107,107,0.08);
        border-color: var(--danger);
        color: var(--danger);
        box-shadow: none;
      }

      /* alerts */
      .stAlert{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
      }
      div[data-baseweb=\"notification\"]{ background: var(--surface) !important; }

      /* ---------- Source badges ---------- */
      .badges{ display:flex; flex-wrap:wrap; gap:8px; margin-top: 6px; }
      .badge{
        display:inline-flex; align-items:center; gap:8px;
        padding: 6px 11px; border-radius: 999px;
        font-family:'JetBrains Mono', monospace; font-size: 0.72rem;
        background: var(--surface); border: 1px solid var(--border);
        color: var(--text-mute);
      }
      .badge .dot{
        width:7px; height:7px; border-radius:50%;
      }
      .badge.pdf .dot{ background: var(--pdf); box-shadow: 0 0 8px var(--pdf); }
      .badge.url .dot{ background: var(--url); box-shadow: 0 0 8px var(--url); }
      .badge.img .dot{ background: var(--img); box-shadow: 0 0 8px var(--img); }
      .badge .num{ color: var(--text); font-weight: 600; }

      /* ---------- KB status card ---------- */
      .kb-card{
        background: linear-gradient(180deg, var(--surface) 0%, var(--bg-soft) 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 8px;
      }
      .kb-card .label{
        font-family:'JetBrains Mono', monospace;
        font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--text-dim); margin-bottom: 6px;
      }
      .kb-card .value{
        font-family:'Fraunces', serif; font-style: italic;
        font-size: 1.05rem; color: var(--text);
      }
      .kb-card .value.ready{ color: var(--accent); }

      /* ---------- Chat ---------- */
      [data-testid=\"stChatMessage\"]{
        background: transparent !important;
        border: none !important;
        padding: 4px 0 !important;
        animation: fadein .35s ease;
      }
      @keyframes fadein{
        from{ opacity:0; transform: translateY(6px); }
        to{ opacity:1; transform: translateY(0); }
      }
      [data-testid=\"stChatMessage\"] [data-testid=\"stChatMessageContent\"]{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 18px !important;
        font-size: 0.97rem; line-height: 1.6;
        color: var(--text);
      }
      /* user bubble — accent tint */
      [data-testid=\"stChatMessage\"]:has(div[data-testid=\"chatAvatarIcon-user\"]) [data-testid=\"stChatMessageContent\"]{
        background: rgba(201,242,107,0.06);
        border-color: rgba(201,242,107,0.25);
      }
      /* avatars */
      [data-testid=\"chatAvatarIcon-user\"], [data-testid=\"chatAvatarIcon-assistant\"]{
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
      }

      /* chat input pinned */
      [data-testid=\"stChatInput\"]{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
      }
      [data-testid=\"stChatInput\"] textarea{
        background: transparent !important; color: var(--text) !important;
        font-family:'Geist', sans-serif !important;
      }
      [data-testid=\"stChatInput\"]:focus-within{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(201,242,107,0.12) !important;
      }

      /* ---------- Empty state ---------- */
      .empty{
        text-align:left; padding: 56px 8px 20px 8px;
      }
      .empty .eyebrow{
        font-family:'JetBrains Mono', monospace;
        font-size: 0.72rem; letter-spacing: 0.22em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 18px;
      }
      .empty h2{
        font-size: 2.2rem; line-height: 1.15; margin: 0 0 14px 0;
        max-width: 640px; font-weight: 400;
      }
      .empty h2 em{ color: var(--accent); font-style: italic; font-weight: 600; }
      .empty p{
        color: var(--text-mute); max-width: 560px;
        font-size: 1rem; line-height: 1.6;
      }
      .hint-grid{
        display:grid; grid-template-columns: repeat(3, minmax(0,1fr));
        gap:12px; margin-top: 28px; max-width: 720px;
      }
      .hint{
        background: var(--surface); border:1px solid var(--border);
        border-radius: 12px; padding: 14px;
        transition: border-color .2s ease, transform .15s ease;
      }
      .hint:hover{ border-color: var(--border-2); transform: translateY(-2px); }
      .hint .k{
        font-family:'JetBrains Mono', monospace;
        font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--text-dim); margin-bottom: 8px;
      }
      .hint .v{ color: var(--text); font-size: 0.92rem; line-height: 1.5; }
      .hint .v b{ color: var(--accent); font-weight: 600; }

      /* scrollbar */
      ::-webkit-scrollbar{ width: 10px; height: 10px; }
      ::-webkit-scrollbar-track{ background: var(--bg); }
      ::-webkit-scrollbar-thumb{ background: var(--border); border-radius: 10px; }
      ::-webkit-scrollbar-thumb:hover{ background: var(--border-2); }
    </style>
    \"\"\",
    unsafe_allow_html=True,
)

# =========================================================
# SILENT API-KEY GUARD  (debug lines hidden)
# =========================================================
if not api_key:
    st.error(\"Configuration error — `GEMINI_API_KEY` is missing. Please add it to your `.env` and reload.\")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model=\"gemini-2.5-flash-lite\",
    google_api_key=api_key
)

# =========================================================
# SESSION STATE  (memory)
# =========================================================
if \"db\" not in st.session_state:
    st.session_state.db = None

if \"chat\" not in st.session_state:
    st.session_state.chat = []

if \"sources\" not in st.session_state:
    st.session_state.sources = {\"pdf\": 0, \"url\": 0, \"image\": 0}

# =========================================================
# BRAND HEADER
# =========================================================
st.markdown(
    \"\"\"
    <div class=\"brand\">
      <div class=\"brand-mark\"></div>
      <div class=\"brand-text\">
        <h1>Prism<span class=\"accent\">.</span></h1>
        <p>Multimodal RAG <span class=\"dot\">●</span> documents <span class=\"dot\">●</span> the web <span class=\"dot\">●</span> images</p>
      </div>
    </div>
    \"\"\",
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR  —  Knowledge ingestion
# =========================================================
with st.sidebar:
    st.markdown(\"### Knowledge\")

    source_type = st.radio(
        \"Source\",
        [\"PDF\", \"URL\", \"Image\"],
        label_visibility=\"collapsed\",
        horizontal=True,
    )

    docs = None

    if source_type == \"PDF\":
        uploaded_pdf = st.file_uploader(\"Upload a PDF document\", type=[\"pdf\"], key=\"pdf_uploader\")

        if uploaded_pdf is not None and st.button(\"Ingest PDF\", key=\"btn_pdf\"):
            temp_path = BASE_DIR / uploaded_pdf.name
            with open(temp_path, \"wb\") as f:
                f.write(uploaded_pdf.getbuffer())
            docs = load_pdf(str(temp_path))
            if docs:
                st.session_state.sources[\"pdf\"] += 1

    elif source_type == \"URL\":
        url = st.text_input(\"Webpage URL\", key=\"url_input\", placeholder=\"https://example.com/article\")

        if url and st.button(\"Ingest URL\", key=\"btn_url\"):
            docs = load_url(url)
            if docs:
                st.session_state.sources[\"url\"] += 1

    elif source_type == \"Image\":
        uploaded_image = st.file_uploader(\"Upload an image\", type=[\"png\", \"jpg\", \"jpeg\"], key=\"image_uploader\")

        if uploaded_image is not None and st.button(\"Ingest Image\", key=\"btn_img\"):
            temp_path = BASE_DIR / uploaded_image.name
            with open(temp_path, \"wb\") as f:
                f.write(uploaded_image.getbuffer())
            docs = load_image(str(temp_path))
            if docs:
                st.session_state.sources[\"image\"] += 1

    if docs:
        chunks = chunk_documents(docs)
        if st.session_state.db is None:
            st.session_state.db = create_vectorstore(chunks)
        else:
            st.session_state.db.add_documents(chunks)
        st.success(\"Indexed — ready to query.\")

    st.markdown(\"<hr/>\", unsafe_allow_html=True)

    # KB status
    st.markdown(\"### Status\")
    s = st.session_state.sources
    total = s[\"pdf\"] + s[\"url\"] + s[\"image\"]
    ready_html = (
        f'<div class=\"value ready\">● Active &nbsp;<span class=\"mono\" style=\"color:var(--text-mute);font-size:.85rem\">{total} sources</span></div>'
        if st.session_state.db is not None
        else '<div class=\"value\">○ Empty</div>'
    )
    st.markdown(
        f\"\"\"
        <div class=\"kb-card\">
          <div class=\"label\">Knowledge base</div>
          {ready_html}
          <div class=\"badges\">
            <span class=\"badge pdf\"><span class=\"dot\"></span>PDF <span class=\"num\">{s[\"pdf\"]}</span></span>
            <span class=\"badge url\"><span class=\"dot\"></span>URL <span class=\"num\">{s[\"url\"]}</span></span>
            <span class=\"badge img\"><span class=\"dot\"></span>IMG <span class=\"num\">{s[\"image\"]}</span></span>
          </div>
        </div>
        \"\"\",
        unsafe_allow_html=True,
    )

    st.markdown(\"<hr/>\", unsafe_allow_html=True)

    # Controls
    st.markdown(\"### Controls\")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class=\"ghost-btn\">', unsafe_allow_html=True)
        if st.button(\"Clear chat\", key=\"btn_clear_chat\"):
            st.session_state.chat = []
            st.rerun()
        st.markdown(\"</div>\", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class=\"danger-btn\">', unsafe_allow_html=True)
        if st.button(\"Reset KB\", key=\"btn_reset_kb\"):
            st.session_state.db = None
            st.session_state.sources = {\"pdf\": 0, \"url\": 0, \"image\": 0}
            st.session_state.chat = []
            st.rerun()
        st.markdown(\"</div>\", unsafe_allow_html=True)

    st.markdown(
        '<p style=\"margin-top:24px;color:var(--text-dim);font-size:.72rem;font-family:JetBrains Mono,monospace;letter-spacing:.12em;\">PRISM · v1.0 · gemini-2.5-flash-lite</p>',
        unsafe_allow_html=True,
    )

# =========================================================
# MAIN  —  Chat surface
# =========================================================
if not st.session_state.chat:
    st.markdown(
        \"\"\"
        <div class=\"empty\">
          <div class=\"eyebrow\">— ready when you are</div>
          <h2>Ask anything across <em>documents</em>, <em>webpages</em>, and <em>images</em>.</h2>
          <p>Add knowledge from the sidebar, or just start chatting — Prism will answer from your indexed sources when available, and from the model otherwise.</p>
          <div class=\"hint-grid\">
            <div class=\"hint\">
              <div class=\"k\">01 · Ingest</div>
              <div class=\"v\">Drop a <b>PDF</b>, paste a <b>URL</b>, or upload an <b>image</b>.</div>
            </div>
            <div class=\"hint\">
              <div class=\"k\">02 · Retrieve</div>
              <div class=\"v\">Prism chunks &amp; embeds it into a <b>vector store</b>.</div>
            </div>
            <div class=\"hint\">
              <div class=\"k\">03 · Ask</div>
              <div class=\"v\">Get grounded answers with <b>multimodal context</b>.</div>
            </div>
          </div>
        </div>
        \"\"\",
        unsafe_allow_html=True,
    )

# replay chat history
for role, message in st.session_state.chat:
    st.chat_message(role).write(message)

user_input = st.chat_input(\"Ask Prism anything…\", key=\"main_chat_input\")

if user_input:
    st.chat_message(\"user\").write(user_input)
    st.session_state.chat.append((\"user\", user_input))

    with st.chat_message(\"assistant\"):
        with st.spinner(\"Thinking…\"):
            if st.session_state.db is None:
                response = llm.invoke(user_input).content
            else:
                docs = retrieve_docs(st.session_state.db, user_input)
                response = generate_answer(docs, user_input, llm)
        st.write(response)

    st.session_state.chat.append((\"assistant\", response))
"
Observation: Create successful: /app/streamlit_app.py