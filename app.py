import streamlit as st
import requests
import json
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import List

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MilIntelX Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── AGGRESSIVE DARK THEME — forces every Streamlit element ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ══ NUKE ALL DEFAULT STREAMLIT COLORS ══ */
*, *::before, *::after { box-sizing: border-box; }

html { background: #000 !important; }

body,
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
.main,
.main > div,
.block-container,
section[data-testid="stMain"],
section[data-testid="stMain"] > div {
  background: #000000 !important;
  background-color: #000000 !important;
  color: #f5f5f7 !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ══ HIDE STREAMLIT CHROME ══ */
#MainMenu,
footer,
header,
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stToolbar"],
.stDeployButton,
.viewerBadge_container__r5tak,
.stDecoration {
  display: none !important;
  visibility: hidden !important;
}

/* ══ BLOCK CONTAINER ══ */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ══ COLUMNS ══ */
[data-testid="stHorizontalBlock"] {
  gap: 0 !important;
  background: #000 !important;
  align-items: flex-start !important;
}

/* Left column panel */
[data-testid="column"]:first-child {
  background: #0a0a0a !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  min-height: 100vh !important;
  padding: 0 !important;
}
[data-testid="column"]:first-child > div,
[data-testid="column"]:first-child [data-testid="stVerticalBlock"] {
  background: #0a0a0a !important;
  padding: 1.5rem 1.2rem !important;
  height: 100% !important;
}

/* Right column */
[data-testid="column"]:last-child {
  background: #000000 !important;
}
[data-testid="column"]:last-child > div,
[data-testid="column"]:last-child [data-testid="stVerticalBlock"] {
  background: #000000 !important;
  padding: 0 1.5rem 5rem !important;
}

/* ══ ALL TEXT ══ */
p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
  color: #f5f5f7 !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
}
.stMarkdown p { color: #f5f5f7 !important; font-size: 15px !important; line-height: 1.72 !important; }

/* ══ SELECTBOX ══ */
[data-baseweb="select"] {
  background: transparent !important;
}
[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  color: #f5f5f7 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}
[data-baseweb="select"] > div:hover {
  border-color: rgba(255,255,255,0.25) !important;
}
[data-baseweb="select"] svg { fill: #a1a1a6 !important; }
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"] {
  background: #1c1c1e !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 12px !important;
}
[role="option"] {
  background: transparent !important;
  color: #f5f5f7 !important;
  font-size: 13px !important;
}
[role="option"]:hover,
[aria-selected="true"] {
  background: rgba(0,113,227,0.2) !important;
  color: #fff !important;
}
.stSelectbox label { color: #6e6e73 !important; font-size: 11px !important; }

/* ══ FILE UPLOADER ══ */
[data-testid="stFileUploader"] {
  background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1.5px dashed rgba(255,255,255,0.15) !important;
  border-radius: 12px !important;
  transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #0071e3 !important;
  background: rgba(0,113,227,0.05) !important;
}
[data-testid="stFileUploaderDropzone"] * {
  color: #a1a1a6 !important;
  font-size: 12.5px !important;
}
[data-testid="stFileUploaderDropzone"] small {
  color: #6e6e73 !important;
}
/* Browse files button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  color: #f5f5f7 !important;
  border-radius: 8px !important;
  font-size: 12px !important;
  padding: 0.3rem 0.9rem !important;
}
/* Uploaded file name row */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileName"] {
  background: rgba(255,255,255,0.04) !important;
  border-radius: 8px !important;
  color: #a1a1a6 !important;
}

/* ══ BUTTONS — kill ALL blue defaults ══ */
.stButton > button,
button[kind="primary"],
button[kind="secondary"],
button[data-testid] {
  background: #0071e3 !important;
  background-color: #0071e3 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.55rem 1rem !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  font-family: 'Inter', sans-serif !important;
  width: 100% !important;
  cursor: pointer !important;
  box-shadow: 0 2px 12px rgba(0,113,227,0.3) !important;
  transition: all 0.2s ease !important;
  letter-spacing: -0.01em !important;
}
.stButton > button:hover {
  background: #0077ed !important;
  background-color: #0077ed !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(0,113,227,0.4) !important;
}
.stButton > button:active,
.stButton > button:focus {
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(0,113,227,0.3) !important;
}

/* Reset button — dark variant */
.stButton > button[title*="Reset"],
.stButton:has(button:contains("Reset")) > button {
  background: rgba(255,255,255,0.07) !important;
  background-color: rgba(255,255,255,0.07) !important;
  color: #a1a1a6 !important;
  box-shadow: none !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
}

/* ══ CHAT INPUT ══ */
[data-testid="stChatInput"] {
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 24px !important;
  padding: 0 !important;
  /* kill Streamlit's default red/blue focus ring */
  box-shadow: none !important;
  outline: none !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: #0071e3 !important;
  box-shadow: 0 0 0 3px rgba(0,113,227,0.2) !important;
  outline: none !important;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: #f5f5f7 !important;
  font-size: 15px !important;
  font-family: 'Inter', sans-serif !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  caret-color: #0071e3 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #6e6e73 !important;
}
/* The send button inside chat input */
[data-testid="stChatInput"] button {
  background: #0071e3 !important;
  border-radius: 50% !important;
  border: none !important;
  color: #fff !important;
  width: 32px !important; height: 32px !important;
  min-width: 32px !important;
  padding: 0 !important;
  box-shadow: none !important;
}
/* Also target the stChatInputContainer variant */
.stChatInputContainer {
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 24px !important;
  box-shadow: none !important;
  outline: none !important;
}
.stChatInputContainer:focus-within {
  border-color: #0071e3 !important;
  box-shadow: 0 0 0 3px rgba(0,113,227,0.2) !important;
}
.stChatInputContainer textarea {
  background: transparent !important;
  color: #f5f5f7 !important;
  caret-color: #0071e3 !important;
}
.stChatInputContainer textarea::placeholder { color: #6e6e73 !important; }

/* ══ PROGRESS BAR ══ */
[data-testid="stProgressBar"] > div,
.stProgress > div > div {
  background: rgba(255,255,255,0.08) !important;
  border-radius: 99px !important;
}
[data-testid="stProgressBar"] > div > div,
.stProgress > div > div > div {
  background: #0071e3 !important;
  border-radius: 99px !important;
}
[data-testid="stProgressBar"] p { color: #a1a1a6 !important; font-size: 12px !important; }

/* ══ SPINNER ══ */
.stSpinner > div { border-top-color: #0071e3 !important; }
[data-testid="stSpinner"] { color: #a1a1a6 !important; }

/* ══ ALERTS ══ */
[data-testid="stAlert"],
.stAlert,
[data-testid="stNotification"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-left: 3px solid #0071e3 !important;
  border-radius: 12px !important;
  color: #f5f5f7 !important;
}
[data-testid="stAlert"] p { color: #f5f5f7 !important; font-size: 13px !important; }

/* ══ EXPANDER ══ */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
  background: transparent !important;
  color: #a1a1a6 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover { color: #f5f5f7 !important; }
[data-testid="stExpanderDetails"] {
  background: transparent !important;
}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 99px; }

/* ══ HR ══ */
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.07) !important; margin: 0.6rem 0 !important; }

/* ══ CUSTOM COMPONENTS ══ */

/* Logo */
.logo-row {
  display: flex; align-items: center; gap: 11px;
  padding-bottom: 1.4rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 0.3rem;
}
.logo-icon {
  width: 40px; height: 40px; flex-shrink: 0;
  background: #0071e3; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 20px;
}
.logo-name {
  font-size: 15px; font-weight: 650;
  letter-spacing: -0.025em; color: #f5f5f7 !important;
}
.logo-sub {
  font-size: 11px; color: #6e6e73 !important; margin-top: 2px;
}

/* Section labels */
.sec {
  font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #6e6e73 !important; margin: 1.3rem 0 0.5rem;
  display: block;
}

/* Indexed file tags */
.ftag {
  display: flex; align-items: center; gap: 7px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  font-size: 12px; color: #a1a1a6 !important;
  margin-bottom: 5px; word-break: break-all;
}

/* Panel footer */
.pfooter {
  margin-top: 2rem; padding-top: 1.2rem;
  font-size: 10.5px; color: #6e6e73 !important;
  text-align: center; line-height: 1.6;
  border-top: 1px solid rgba(255,255,255,0.07);
}

/* Hero */
.hero { text-align: center; padding: 3rem 0.5rem 1.5rem; }
.hero-eye {
  font-size: 11.5px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: #0071e3 !important; margin-bottom: 0.85rem;
}
.hero-title {
  font-size: clamp(1.9rem, 4vw, 3.2rem);
  font-weight: 700; letter-spacing: -0.035em; line-height: 1.04;
  background: linear-gradient(150deg,#f5f5f7 0%,#6e6e73 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; margin: 0 0 0.75rem;
  display: block;
}
.hero-sub {
  font-size: 15px; color: #a1a1a6 !important;
  letter-spacing: -0.01em; margin: 0; line-height: 1.5;
}

/* Pills */
.pills {
  display: flex; flex-wrap: wrap;
  align-items: center; justify-content: center;
  gap: 6px; margin: 1.1rem 0 2rem;
}
.pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 12px; border-radius: 99px;
  font-size: 11.5px; font-weight: 500;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.04);
  color: #a1a1a6 !important;
}
.dot {
  width: 6px; height: 6px; border-radius: 50%;
  display: inline-block;
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

/* Chat message labels */
.you-label {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #6e6e73 !important; text-align: right;
  margin-bottom: 3px; margin-top: 1rem;
}
.you-text {
  text-align: right; color: #a1a1a6 !important;
  font-size: 14.5px; font-style: italic; line-height: 1.5;
  padding: 0.25rem 0 0.8rem;
  border-bottom: 1px solid rgba(255,255,255,0.07);
}
.ai-label {
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: #0071e3 !important; margin-bottom: 4px; margin-top: 0.5rem;
}

/* Source tag */
.srctag {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 10px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  font-size: 12px; color: #a1a1a6 !important;
  margin: 3px 2px; word-break: break-all;
}

/* Empty state */
.empty {
  text-align: center; padding: 4rem 1.5rem 3rem;
}
.empty-icon { font-size: 2.5rem; opacity: .3; margin-bottom: 1rem; display: block; }
.empty-title {
  font-size: 17px; font-weight: 600;
  color: #a1a1a6 !important; margin-bottom: 0.4rem; display: block;
}
.empty-body { font-size: 14px; color: #6e6e73 !important; line-height: 1.65; }

/* Fade in animation */
@keyframes fadeUp {
  from { opacity:0; transform: translateY(8px); }
  to   { opacity:1; transform: translateY(0); }
}
.fade { animation: fadeUp 0.35s ease both; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ───────────────────────────────────────────────────────────────────
NV_KEY = st.secrets["NVIDIA_API_KEY"]

EMBED_MODELS = {
    "nv-embedqa-e5-v5 · Recommended": "nvidia/nv-embedqa-e5-v5",
    "llama-nemotron-embed-1b-v2":      "nvidia/llama-nemotron-embed-1b-v2",
    "llama-3.2-nv-embedqa-1b-v2":      "nvidia/llama-3.2-nv-embedqa-1b-v2",
}
CHAT_MODELS = {
    "llama-3.1-70b · Balanced":   "meta/llama-3.1-70b-instruct",
    "nemotron-70b · Precision":   "nvidia/llama-3.1-nemotron-70b-instruct",
    "llama-3.1-405b · Max Power": "meta/llama-3.1-405b-instruct",
    "nemotron-4b · Fast":         "nvidia/nemotron-4b-instruct-v1",
}
SYSTEM_PROMPT = (
    "You are a precise military intelligence analyst. "
    "Answer exclusively using the provided document context. "
    "Structure responses clearly with key findings first. "
    "If the answer is absent from the context, state: "
    "'This information is not available in the indexed documents.' "
    "Never fabricate or speculate beyond what is documented."
)


# ─── EMBEDDINGS ──────────────────────────────────────────────────────────────────
class NemotronEmbeddings(Embeddings):
    BATCH_SIZE = 32

    def __init__(self, model_name: str):
        self.url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.model = model_name
        self.headers = {"Authorization": f"Bearer {NV_KEY}", "Content-Type": "application/json"}

    def _call(self, texts: List[str], input_type: str) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            payload = {"input": batch, "model": self.model,
                       "input_type": input_type, "encoding_format": "float"}
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            resp.raise_for_status()
            items = sorted(resp.json()["data"], key=lambda x: x["index"])
            results.extend([it["embedding"] for it in items])
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts, "passage")

    def embed_query(self, text: str) -> List[float]:
        return self._call([text], "query")[0]


# ─── SESSION STATE ────────────────────────────────────────────────────────────────
for k, v in {"vector_db": None, "filenames": [], "chat_history": [], "panel_open": True}.items():
    if k not in st.session_state:
        st.session_state[k] = v

kb_ready = st.session_state.vector_db is not None


# ════════════════════════════════════════════════════════════════════════════════
#  LAYOUT — Tab-based on mobile, two-column feel via tabs
# ════════════════════════════════════════════════════════════════════════════════

# We use Streamlit tabs so it works perfectly on mobile AND desktop
# Tab 1 = Settings/Panel, Tab 2 = Chat
# On desktop the left panel still shows nicely in a wide tab

# Actually: use columns but inject a mobile-toggle via tabs at top
# Best approach for mobile: use st.tabs for mobile, hide on desktop

# ── TOP NAV TOGGLE ──
nav_col1, nav_col2 = st.columns([1, 1], gap="small")
with nav_col1:
    panel_tab = st.button("⚙️  Panel", use_container_width=True, key="panel_btn")
with nav_col2:
    chat_tab = st.button("💬  Chat", use_container_width=True, key="chat_btn")

if panel_tab:
    st.session_state.panel_open = True
if chat_tab:
    st.session_state.panel_open = False

st.markdown("<hr>", unsafe_allow_html=True)

# ── RENDER BASED ON STATE ──
if st.session_state.panel_open:
    # ═══ PANEL VIEW ═══
    st.markdown("""
    <div class="logo-row fade">
      <div class="logo-icon">🛡️</div>
      <div>
        <div class="logo-name">MilIntelX Pro</div>
        <div class="logo-sub">Intelligence Analysis System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sec">Search Engine</span>', unsafe_allow_html=True)
    embed_label = st.selectbox("embed", list(EMBED_MODELS.keys()), label_visibility="collapsed")
    embed_model = EMBED_MODELS[embed_label]

    st.markdown('<span class="sec">Language Model</span>', unsafe_allow_html=True)
    chat_label = st.selectbox("chat", list(CHAT_MODELS.keys()), label_visibility="collapsed")
    chat_model = CHAT_MODELS[chat_label]
    # Save to session so chat view can access
    st.session_state["embed_model"] = embed_model
    st.session_state["chat_model"] = chat_model
    st.session_state["chat_label"] = chat_label
    st.session_state["embed_label"] = embed_label

    st.markdown('<span class="sec">Knowledge Base</span>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "pdfs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )

    if st.session_state.filenames:
        st.markdown('<span class="sec">Indexed Files</span>', unsafe_allow_html=True)
        for fn in st.session_state.filenames:
            st.markdown(f'<div class="ftag">📄 {fn}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        build_btn = st.button("⚡  Index", use_container_width=True, key="build")
    with c2:
        reset_btn = st.button("🗑  Reset", use_container_width=True, key="reset")

    # ── Index logic ──
    if build_btn and uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.filenames]
        if not new_files:
            st.info("All files already indexed.")
        else:
            prog = st.progress(0, text="Reading documents…")
            all_chunks, all_meta = [], []
            for idx, file in enumerate(new_files):
                try:
                    reader = PdfReader(file)
                    text = "".join(p.extract_text() or "" for p in reader.pages)
                    if not text.strip():
                        st.warning(f"⚠️ {file.name}: no extractable text (scanned PDF?).")
                        continue
                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=900, chunk_overlap=150
                    ).split_text(text)
                    all_chunks.extend(chunks)
                    all_meta.extend([{"source": file.name}] * len(chunks))
                    st.session_state.filenames.append(file.name)
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                prog.progress(int((idx + 1) / len(new_files) * 60),
                              text=f"Parsed {idx + 1}/{len(new_files)}…")

            if all_chunks:
                prog.progress(70, text="Generating embeddings…")
                try:
                    embedder = NemotronEmbeddings(embed_model)
                    new_db = FAISS.from_texts(all_chunks, embedder, metadatas=all_meta)
                    if st.session_state.vector_db:
                        st.session_state.vector_db.merge_from(new_db)
                    else:
                        st.session_state.vector_db = new_db
                    prog.progress(100, text="Done.")
                    time.sleep(0.4)
                    prog.empty()
                    st.success(f"✅ {len(all_chunks)} chunks indexed from {len(new_files)} file(s).")
                    st.session_state.panel_open = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
    elif build_btn:
        st.warning("Upload at least one PDF first.")

    if reset_btn:
        st.session_state.vector_db = None
        st.session_state.filenames = []
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
    <div class="pfooter">
      Powered by<br>NVIDIA NIM · LangChain · FAISS
    </div>
    """, unsafe_allow_html=True)

else:
    # ═══ CHAT VIEW ═══
    kb_ready = st.session_state.vector_db is not None
    chat_model  = st.session_state.get("chat_model",  "meta/llama-3.1-70b-instruct")
    embed_model = st.session_state.get("embed_model", "nvidia/nv-embedqa-e5-v5")
    chat_label  = st.session_state.get("chat_label",  "llama-3.1-70b · Balanced")
    embed_label = st.session_state.get("embed_label", "nv-embedqa-e5-v5 · Recommended")

    status_color = "#30d158" if kb_ready else "#6e6e73"
    status_label = "Knowledge Base Active" if kb_ready else "No Documents Indexed"

    st.markdown(f"""
    <div class="hero fade">
      <div class="hero-eye">Intelligence Analysis Platform</div>
      <span class="hero-title">MilIntelX Pro</span>
      <p class="hero-sub">Upload classified documents.<br>Ask anything. Get grounded answers.</p>
    </div>
    <div class="pills">
      <div class="pill">
        <span class="dot" style="background:{status_color};box-shadow:0 0 5px {status_color}"></span>
        {status_label}
      </div>
      <div class="pill">🧠 {chat_label.split("·")[0].strip()}</div>
      <div class="pill">🔍 {embed_label.split("·")[0].strip()}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render chat history ──
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="you-label">You</div>'
                    f'<div class="you-text">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="ai-label">MilIntelX</div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Sources", expanded=False):
                        for src in sorted(set(msg["sources"])):
                            st.markdown(f'<div class="srctag">📄 {src}</div>', unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
    else:
        if not kb_ready:
            st.markdown("""
            <div class="empty fade">
              <span class="empty-icon">🛡️</span>
              <span class="empty-title">No documents indexed yet</span>
              <p class="empty-body">Tap ⚙️ Panel above, upload your PDFs<br>and hit ⚡ Index to get started.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty fade">
              <span class="empty-icon">💬</span>
              <span class="empty-title">Knowledge base ready</span>
              <p class="empty-body">Ask anything about your indexed documents.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Chat input ──
    if prompt := st.chat_input("Ask about your documents…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.panel_open = False

        if not kb_ready:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Please go to ⚙️ Panel, upload and index your documents first.",
                "sources": [],
            })
            st.rerun()
        else:
            with st.spinner("Analyzing…"):
                try:
                    docs = st.session_state.vector_db.similarity_search(prompt, k=6)
                    context = "\n\n".join(
                        f"[SOURCE: {d.metadata['source']}]\n{d.page_content}" for d in docs
                    )
                    sources = [d.metadata["source"] for d in docs]

                    response = requests.post(
                        "https://integrate.api.nvidia.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {NV_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": chat_model,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",
                                 "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"},
                            ],
                            "temperature": 0.1,
                            "max_tokens": 1024,
                            "stream": True,
                        },
                        stream=True,
                        timeout=90,
                    )

                    if response.status_code != 200:
                        err = f"Model error {response.status_code}: {response.text[:200]}"
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": err, "sources": []})
                    else:
                        full_answer = ""
                        for line in response.iter_lines():
                            if not line:
                                continue
                            decoded = line.decode("utf-8", errors="replace")
                            if decoded.startswith("data: "):
                                decoded = decoded[6:]
                            if decoded == "[DONE]":
                                break
                            try:
                                chunk = json.loads(decoded)
                                full_answer += chunk["choices"][0]["delta"].get("content", "")
                            except (json.JSONDecodeError, KeyError):
                                continue

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_answer,
                            "sources": sources,
                        })

                except requests.exceptions.Timeout:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "⚠️ Request timed out. Go to Panel and switch to nemotron-4b (Fast).",
                        "sources": [],
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error: {e}",
                        "sources": [],
                    })
            st.rerun()
