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
    initial_sidebar_state="expanded",
)

# ─── APPLE-STYLE CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  /* ── Root Variables ── */
  :root {
    --bg-primary:     #000000;
    --bg-secondary:   #0a0a0a;
    --bg-card:        rgba(255,255,255,0.04);
    --bg-card-hover:  rgba(255,255,255,0.07);
    --border:         rgba(255,255,255,0.08);
    --border-active:  rgba(255,255,255,0.18);
    --text-primary:   #f5f5f7;
    --text-secondary: #a1a1a6;
    --text-tertiary:  #6e6e73;
    --accent:         #0071e3;
    --accent-hover:   #0077ed;
    --accent-glow:    rgba(0,113,227,0.25);
    --success:        #30d158;
    --warning:        #ffd60a;
    --danger:         #ff453a;
    --radius-sm:      8px;
    --radius-md:      14px;
    --radius-lg:      20px;
    --radius-xl:      28px;
    --font:           'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: var(--font) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  .main .block-container {
    background: var(--bg-primary);
    padding: 2rem 2.5rem 4rem;
    max-width: 1100px;
  }

  /* ── Hide Streamlit Chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-active); border-radius: 99px; }

  /* ── Hero Header ── */
  .hero-header {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    margin-bottom: 0.5rem;
  }
  .hero-eyebrow {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
  }
  .hero-title {
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.05;
    background: linear-gradient(135deg, #f5f5f7 0%, #a1a1a6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem;
  }
  .hero-subtitle {
    font-size: 17px;
    font-weight: 400;
    color: var(--text-secondary);
    letter-spacing: -0.01em;
    margin: 0;
  }

  /* ── Status Pill ── */
  .status-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 1.25rem 0 2.5rem;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.01em;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-secondary);
  }
  .status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 6px var(--success);
    animation: pulse-dot 2s ease-in-out infinite;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Chat Container ── */
  .chat-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(20px);
  }

  /* ── Chat Messages ── */
  .stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
  }
  [data-testid="stChatMessageContent"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.25rem !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    color: var(--text-primary) !important;
  }
  [data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
    background: var(--accent) !important;
    border-color: transparent !important;
  }

  /* ── Chat Input ── */
  .stChatInputContainer {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-active) !important;
    border-radius: var(--radius-xl) !important;
    padding: 0.15rem 0.5rem !important;
    backdrop-filter: blur(20px);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .stChatInputContainer:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
  }
  .stChatInputContainer textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-size: 15px !important;
    font-family: var(--font) !important;
  }
  .stChatInputContainer textarea::placeholder {
    color: var(--text-tertiary) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(10,10,10,0.95) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(30px);
  }
  [data-testid="stSidebar"] .block-container {
    padding: 2rem 1.25rem;
  }

  /* ── Sidebar Header ── */
  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-logo-icon {
    width: 36px; height: 36px;
    background: var(--accent);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .sidebar-logo-text { font-size: 15px; font-weight: 600; letter-spacing: -0.02em; }
  .sidebar-logo-sub  { font-size: 11px; color: var(--text-tertiary); margin-top: 1px; }

  /* ── Section Labels ── */
  .section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-tertiary);
    margin: 1.5rem 0 0.6rem;
  }

  /* ── Selectbox ── */
  .stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
    transition: border-color 0.2s ease;
  }
  .stSelectbox [data-baseweb="select"] > div:hover {
    border-color: var(--border-active) !important;
  }
  .stSelectbox label { font-size: 12px !important; color: var(--text-secondary) !important; }

  /* ── File Uploader ── */
  [data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-active) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s ease, background 0.2s ease;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(0,113,227,0.05) !important;
  }
  [data-testid="stFileUploaderDropzone"] p {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.55rem 1.1rem !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: var(--font) !important;
    letter-spacing: -0.01em !important;
    width: 100% !important;
    transition: background 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
    cursor: pointer !important;
  }
  .stButton > button:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }
  .stButton > button[kind="secondary"], button[data-testid*="reset"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    box-shadow: none !important;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
    background: rgba(255,69,58,0.06) !important;
  }

  /* ── Source Cards ── */
  .source-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    margin: 6px 0;
    font-size: 12px;
    color: var(--text-secondary);
    transition: border-color 0.2s;
  }
  .source-card:hover { border-color: var(--border-active); }
  .source-icon { font-size: 14px; opacity: 0.7; }

  /* ── Info / Warning / Error banners ── */
  .stAlert {
    background: var(--bg-card) !important;
    border-radius: var(--radius-md) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--text-primary) !important;
    font-size: 13px !important;
  }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: var(--accent) !important; }

  /* ── Progress bar ── */
  .stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px !important; }
  .stProgress > div > div { background: var(--border) !important; border-radius: 99px !important; }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

  /* ── Tooltip Tag ── */
  .model-tag {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(0,113,227,0.12);
    border: 1px solid rgba(0,113,227,0.25);
    border-radius: 99px;
    font-size: 11px;
    color: var(--accent);
    font-weight: 500;
    margin-left: 6px;
    vertical-align: middle;
  }

  /* ── Empty State ── */
  .empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: var(--text-tertiary);
  }
  .empty-state-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
  .empty-state-title { font-size: 18px; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.4rem; }
  .empty-state-body { font-size: 14px; line-height: 1.6; }

  /* ── Animations ── */
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fade-in { animation: fadeInUp 0.4s ease both; }
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
    "llama-3.1-70b · Balanced":        "meta/llama-3.1-70b-instruct",
    "nemotron-70b · Precision":        "nvidia/llama-3.1-nemotron-70b-instruct",
    "llama-3.1-405b · Max Power":      "meta/llama-3.1-405b-instruct",
    "nemotron-4b · Fast":              "nvidia/nemotron-4b-instruct-v1",
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
    """Batched NVIDIA embeddings — single API call per batch."""

    BATCH_SIZE = 32  # stay well within token limits

    def __init__(self, model_name: str):
        self.url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.model = model_name
        self.headers = {
            "Authorization": f"Bearer {NV_KEY}",
            "Content-Type":  "application/json",
        }

    def _call(self, texts: List[str], input_type: str) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            payload = {
                "input":            batch,
                "model":            self.model,
                "input_type":       input_type,
                "encoding_format":  "float",
            }
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            # sort by index to preserve order
            items = sorted(data["data"], key=lambda x: x["index"])
            results.extend([item["embedding"] for item in items])
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts, "passage")

    def embed_query(self, text: str) -> List[float]:
        return self._call([text], "query")[0]


# ─── SESSION STATE ────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "vector_db":    None,
        "filenames":    [],
        "chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">🛡️</div>
      <div>
        <div class="sidebar-logo-text">MilIntelX Pro</div>
        <div class="sidebar-logo-sub">Intelligence Analysis System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Models
    st.markdown('<div class="section-label">Search Engine</div>', unsafe_allow_html=True)
    embed_label = st.selectbox(
        "Embedding Model",
        options=list(EMBED_MODELS.keys()),
        label_visibility="collapsed",
    )
    embed_model = EMBED_MODELS[embed_label]

    st.markdown('<div class="section-label">Language Model</div>', unsafe_allow_html=True)
    chat_label = st.selectbox(
        "Chat Model",
        options=list(CHAT_MODELS.keys()),
        label_visibility="collapsed",
    )
    chat_model = CHAT_MODELS[chat_label]

    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Show already-indexed files
    if st.session_state.filenames:
        for fn in st.session_state.filenames:
            st.markdown(
                f'<div class="source-card"><span class="source-icon">📄</span>{fn}</div>',
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("⚡ Index", use_container_width=True)
    with col2:
        reset_btn = st.button("🗑 Reset", use_container_width=True)

    # ── Index ──
    if build_btn and uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.filenames]
        if not new_files:
            st.info("All files already indexed.")
        else:
            progress = st.progress(0, text="Reading documents…")
            all_chunks, all_meta = [], []

            for idx, file in enumerate(new_files):
                try:
                    reader = PdfReader(file)
                    text = "".join(
                        p.extract_text() or "" for p in reader.pages
                    )
                    if not text.strip():
                        st.warning(f"⚠️ {file.name}: no extractable text (scanned PDF?).")
                        continue
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=900, chunk_overlap=150, length_function=len
                    )
                    chunks = splitter.split_text(text)
                    all_chunks.extend(chunks)
                    all_meta.extend([{"source": file.name}] * len(chunks))
                    st.session_state.filenames.append(file.name)
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")

                progress.progress(
                    int((idx + 1) / len(new_files) * 60),
                    text=f"Parsed {idx + 1}/{len(new_files)} files…",
                )

            if all_chunks:
                progress.progress(70, text="Generating embeddings…")
                try:
                    embedder = NemotronEmbeddings(embed_model)
                    new_db = FAISS.from_texts(all_chunks, embedder, metadatas=all_meta)
                    if st.session_state.vector_db:
                        st.session_state.vector_db.merge_from(new_db)
                    else:
                        st.session_state.vector_db = new_db
                    progress.progress(100, text="Done.")
                    time.sleep(0.5)
                    progress.empty()
                    st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(new_files)} file(s).")
                except Exception as e:
                    st.error(f"Embedding failed: {e}")
    elif build_btn:
        st.warning("Upload at least one PDF first.")

    # ── Reset ──
    if reset_btn:
        st.session_state.vector_db = None
        st.session_state.filenames = []
        st.session_state.chat_history = []
        st.rerun()

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:var(--text-tertiary);text-align:center;">'
        'Powered by NVIDIA NIM · LangChain · FAISS'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────────

# Hero
kb_ready = st.session_state.vector_db is not None
status_label = "Knowledge Base Active" if kb_ready else "No Documents Indexed"
status_color = "#30d158" if kb_ready else "#6e6e73"

st.markdown(f"""
<div class="hero-header fade-in">
  <div class="hero-eyebrow">Intelligence Analysis Platform</div>
  <h1 class="hero-title">MilIntelX Pro</h1>
  <p class="hero-subtitle">Upload classified documents. Ask anything. Get answers grounded in your data.</p>
</div>
<div class="status-bar">
  <div class="status-pill">
    <div class="status-dot" style="background:{status_color};box-shadow:0 0 6px {status_color};"></div>
    {status_label}
  </div>
  <div class="status-pill">🧠 {chat_label.split('·')[0].strip()}</div>
  <div class="status-pill">🔍 {embed_label.split('·')[0].strip()}</div>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    unique_sources = sorted(set(msg["sources"]))
                    for src in unique_sources:
                        st.markdown(
                            f'<div class="source-card"><span class="source-icon">📄</span>{src}</div>',
                            unsafe_allow_html=True,
                        )
else:
    if not kb_ready:
        st.markdown("""
        <div class="empty-state fade-in">
          <div class="empty-state-icon">🛡️</div>
          <div class="empty-state-title">Ready for your intelligence</div>
          <div class="empty-state-body">
            Upload PDF documents in the sidebar,<br>
            then build the knowledge base to start querying.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state fade-in">
          <div class="empty-state-icon">💬</div>
          <div class="empty-state-title">Knowledge base active</div>
          <div class="empty-state-body">Ask any question about your indexed documents below.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Input ──
if prompt := st.chat_input("Ask about your documents…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not kb_ready:
        with st.chat_message("assistant"):
            st.info("Please upload and index documents first using the sidebar.")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Please upload and index documents first using the sidebar.",
            "sources": [],
        })
    else:
        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing…"):
                try:
                    # Retrieve top-k chunks
                    docs = st.session_state.vector_db.similarity_search(prompt, k=6)
                    context = "\n\n".join(
                        f"[SOURCE: {d.metadata['source']}]\n{d.page_content}" for d in docs
                    )
                    sources = [d.metadata["source"] for d in docs]

                    # Stream response
                    response = requests.post(
                        "https://integrate.api.nvidia.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {NV_KEY}",
                            "Content-Type":  "application/json",
                        },
                        json={
                            "model":    chat_model,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"},
                            ],
                            "temperature": 0.1,
                            "max_tokens":  1024,
                            "stream":      True,
                        },
                        stream=True,
                        timeout=90,
                    )

                    if response.status_code != 200:
                        err = f"Model error {response.status_code}: {response.text[:200]}"
                        st.error(err)
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": err, "sources": []
                        })
                    else:
                        placeholder = st.empty()
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
                                delta = chunk["choices"][0]["delta"].get("content", "")
                                full_answer += delta
                                placeholder.markdown(full_answer + "▍")
                            except (json.JSONDecodeError, KeyError):
                                continue
                        placeholder.markdown(full_answer)

                        # Source expander
                        if sources:
                            with st.expander("📎 Sources", expanded=False):
                                for src in sorted(set(sources)):
                                    st.markdown(
                                        f'<div class="source-card"><span class="source-icon">📄</span>{src}</div>',
                                        unsafe_allow_html=True,
                                    )

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": full_answer,
                            "sources": sources,
                        })

                except requests.exceptions.Timeout:
                    msg = "⚠️ Request timed out. Try a faster model (nemotron-4b)."
                    st.error(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg, "sources": []})
                except Exception as e:
                    msg = f"❌ Unexpected error: {e}"
                    st.error(msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": msg, "sources": []})
