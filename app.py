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
    initial_sidebar_state="collapsed",   # sidebar hidden — we use our own panel
)

# ─── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

  :root {
    --bg:     #000000;
    --bg2:    #0a0a0a;
    --card:   rgba(255,255,255,0.045);
    --card2:  rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.09);
    --brd2:   rgba(255,255,255,0.18);
    --text:   #f5f5f7;
    --text2:  #a1a1a6;
    --text3:  #6e6e73;
    --accent: #0071e3;
    --acc2:   #0077ed;
    --glow:   rgba(0,113,227,0.22);
    --green:  #30d158;
    --red:    #ff453a;
    --font:   'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
  }
  .main .block-container {
    background: var(--bg);
    padding: 1.5rem 1.5rem 5rem;
    max-width: 100%;
  }

  /* Hide Streamlit chrome + native sidebar */
  #MainMenu, footer, header,
  [data-testid="stSidebar"],
  [data-testid="collapsedControl"],
  .stDeployButton { display: none !important; }

  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-thumb { background: var(--brd2); border-radius: 99px; }

  /* ── LEFT PANEL ── */
  .sec-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text3);
    margin: 1.3rem 0 0.5rem;
  }
  .logo-row {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
  }
  .logo-icon {
    width: 38px; height: 38px;
    background: var(--accent); border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; flex-shrink: 0;
  }
  .logo-name { font-size: 15px; font-weight: 650; letter-spacing: -0.025em; }
  .logo-sub  { font-size: 11px; color: var(--text3); margin-top: 2px; }

  .file-tag {
    display: flex; align-items: center; gap: 7px;
    padding: 6px 10px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 12px; color: var(--text2);
    margin-bottom: 5px; word-break: break-all;
  }
  .panel-footer {
    margin-top: 1.5rem;
    padding-top: 1.2rem;
    font-size: 10.5px; color: var(--text3);
    text-align: center; line-height: 1.6;
    border-top: 1px solid var(--border);
  }

  /* ── HERO ── */
  .hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
  }
  .hero-eye {
    font-size: 12px; font-weight: 600;
    letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 0.8rem;
  }
  .hero-title {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700; letter-spacing: -0.035em; line-height: 1.05;
    background: linear-gradient(150deg, #f5f5f7 0%, #6e6e73 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.75rem;
  }
  .hero-sub { font-size: 15px; color: var(--text2); letter-spacing: -0.01em; margin: 0; }

  .pills {
    display: flex; flex-wrap: wrap;
    align-items: center; justify-content: center;
    gap: 7px; margin: 1.1rem 0 2rem;
  }
  .pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 11px; border-radius: 99px;
    font-size: 11.5px; font-weight: 500;
    border: 1px solid var(--border);
    background: var(--card); color: var(--text2);
  }
  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

  /* ── CHAT — NO AVATARS, NO BUBBLES ── */
  /* Nuke every possible avatar selector */
  [data-testid="stChatMessage"] > div:first-child { display: none !important; }
  [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"],
  [data-testid="chatAvatarIcon-user"],
  [data-testid="chatAvatarIcon-assistant"],
  .stChatMessage [class*="avatar"],
  .stChatMessage [class*="Avatar"] { display: none !important; }

  /* Strip all bubble styling */
  [data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important; box-shadow: none !important;
    padding: 0 !important; margin: 0 !important; gap: 0 !important;
  }
  [data-testid="stChatMessageContent"] {
    background: transparent !important;
    border: none !important; box-shadow: none !important;
    border-radius: 0 !important; padding: 0 !important;
    width: 100% !important; max-width: 100% !important;
  }

  /* ── Message labels ── */
  .msg-you {
    font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text3); text-align: right; margin-bottom: 3px;
  }
  .msg-user-text {
    text-align: right; color: var(--text2);
    font-size: 14.5px; font-style: italic;
    padding: 0.3rem 0 0.7rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.15rem;
  }
  .msg-ai {
    font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--accent); margin-bottom: 4px;
  }
  .msg-ai-text {
    font-size: 15px; line-height: 1.72;
    color: var(--text); padding: 0.4rem 0 1.1rem;
    border-bottom: 1px solid var(--border);
  }

  /* ── Empty state ── */
  .empty {
    text-align: center; padding: 5rem 2rem 3rem; color: var(--text3);
  }
  .empty-icon  { font-size: 2.6rem; opacity: .35; margin-bottom: 1rem; }
  .empty-title { font-size: 17px; font-weight: 600; color: var(--text2); margin-bottom: .4rem; }
  .empty-body  { font-size: 14px; line-height: 1.65; }

  /* ── Widgets ── */
  .stSelectbox [data-baseweb="select"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 13px !important;
    color: var(--text) !important; font-size: 13px !important;
  }
  .stSelectbox label { display: none !important; }

  [data-testid="stFileUploaderDropzone"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--brd2) !important;
    border-radius: 13px !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: rgba(0,113,227,.04) !important;
  }
  [data-testid="stFileUploaderDropzone"] p,
  [data-testid="stFileUploaderDropzone"] span {
    color: var(--text2) !important; font-size: 12.5px !important;
  }

  .stButton > button {
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 13px !important;
    padding: 0.5rem 1rem !important;
    font-size: 13px !important; font-weight: 500 !important;
    font-family: var(--font) !important; width: 100% !important;
    transition: background .2s, transform .1s, box-shadow .2s !important;
    box-shadow: 0 2px 10px var(--glow) !important; cursor: pointer !important;
  }
  .stButton > button:hover {
    background: var(--acc2) !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px var(--glow) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  [data-testid="stChatInput"],
  .stChatInputContainer {
    background: var(--card2) !important;
    border: 1px solid var(--brd2) !important;
    border-radius: 24px !important;
    transition: border-color .2s, box-shadow .2s;
  }
  [data-testid="stChatInput"]:focus-within,
  .stChatInputContainer:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
  }
  [data-testid="stChatInput"] textarea,
  .stChatInputContainer textarea {
    background: transparent !important; color: var(--text) !important;
    font-size: 15px !important; font-family: var(--font) !important;
  }
  [data-testid="stChatInput"] textarea::placeholder,
  .stChatInputContainer textarea::placeholder { color: var(--text3) !important; }

  .streamlit-expanderHeader {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; font-size: 12px !important; color: var(--text2) !important;
  }
  .stProgress > div > div > div { background: var(--accent) !important; border-radius: 99px !important; }
  .stProgress > div > div       { background: var(--border) !important; border-radius: 99px !important; }
  .stSpinner > div { border-top-color: var(--accent) !important; }
  .stAlert {
    background: var(--card) !important; border-radius: 13px !important;
    border-left: 3px solid var(--accent) !important; font-size: 13px !important;
  }
  hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

  .src-tag {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px; background: var(--card);
    border: 1px solid var(--border); border-radius: 8px;
    font-size: 12px; color: var(--text2); margin: 3px 0;
  }

  @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .fade { animation: fadeUp .4s ease both; }
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
            batch = texts[i : i + self.BATCH_SIZE]
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
for k, v in {"vector_db": None, "filenames": [], "chat_history": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

kb_ready = st.session_state.vector_db is not None


# ════════════════════════════════════════════════════════════════════════════════
#  TWO-COLUMN LAYOUT
# ════════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2.8], gap="medium")


# ─── LEFT PANEL ──────────────────────────────────────────────────────────────────
with left:
    st.markdown("""
    <div class="logo-row">
      <div class="logo-icon">🛡️</div>
      <div>
        <div class="logo-name">MilIntelX Pro</div>
        <div class="logo-sub">Intelligence Analysis System</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Search Engine</div>', unsafe_allow_html=True)
    embed_label = st.selectbox("embed", list(EMBED_MODELS.keys()), label_visibility="collapsed")
    embed_model = EMBED_MODELS[embed_label]

    st.markdown('<div class="sec-label">Language Model</div>', unsafe_allow_html=True)
    chat_label = st.selectbox("chat", list(CHAT_MODELS.keys()), label_visibility="collapsed")
    chat_model = CHAT_MODELS[chat_label]

    st.markdown('<div class="sec-label">Knowledge Base</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "pdfs", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
    )

    if st.session_state.filenames:
        for fn in st.session_state.filenames:
            st.markdown(f'<div class="file-tag">📄 {fn}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        build_btn = st.button("⚡ Index", use_container_width=True)
    with c2:
        reset_btn = st.button("🗑 Reset", use_container_width=True)

    if build_btn and uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.filenames]
        if not new_files:
            st.info("All files already indexed.")
        else:
            prog = st.progress(0, text="Reading…")
            all_chunks, all_meta = [], []
            for idx, file in enumerate(new_files):
                try:
                    reader = PdfReader(file)
                    text = "".join(p.extract_text() or "" for p in reader.pages)
                    if not text.strip():
                        st.warning(f"⚠️ {file.name}: no text found.")
                        continue
                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=900, chunk_overlap=150
                    ).split_text(text)
                    all_chunks.extend(chunks)
                    all_meta.extend([{"source": file.name}] * len(chunks))
                    st.session_state.filenames.append(file.name)
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                prog.progress(int((idx + 1) / len(new_files) * 60), text=f"Parsed {idx+1}/{len(new_files)}…")

            if all_chunks:
                prog.progress(70, text="Embedding…")
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
                    st.success(f"✅ {len(all_chunks)} chunks indexed.")
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
    <div class="panel-footer">
      Powered by<br>NVIDIA NIM · LangChain · FAISS
    </div>
    """, unsafe_allow_html=True)


# ─── RIGHT CONTENT ────────────────────────────────────────────────────────────────
with right:
    status_color = "#30d158" if kb_ready else "#6e6e73"
    status_label = "Knowledge Base Active" if kb_ready else "No Documents Indexed"

    st.markdown(f"""
    <div class="hero fade">
      <div class="hero-eye">Intelligence Analysis Platform</div>
      <h1 class="hero-title">MilIntelX Pro</h1>
      <p class="hero-sub">Upload classified documents. Ask anything. Get answers grounded in your data.</p>
    </div>
    <div class="pills">
      <div class="pill">
        <div class="dot" style="background:{status_color};box-shadow:0 0 5px {status_color}"></div>
        {status_label}
      </div>
      <div class="pill">🧠 {chat_label.split("·")[0].strip()}</div>
      <div class="pill">🔍 {embed_label.split("·")[0].strip()}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render chat history manually (no st.chat_message = no avatars) ──
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-you">You</div>'
                    f'<div class="msg-user-text">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="msg-ai">MilIntelX</div>', unsafe_allow_html=True)
                # Render markdown properly via st.markdown (not inside HTML)
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Sources", expanded=False):
                        for src in sorted(set(msg["sources"])):
                            st.markdown(f'<div class="src-tag">📄 {src}</div>', unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
    else:
        if not kb_ready:
            st.markdown("""
            <div class="empty fade">
              <div class="empty-icon">🛡️</div>
              <div class="empty-title">Ready for your intelligence</div>
              <div class="empty-body">Upload PDFs on the left panel,<br>then click ⚡ Index to build the knowledge base.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty fade">
              <div class="empty-icon">💬</div>
              <div class="empty-title">Knowledge base active</div>
              <div class="empty-body">Ask anything about your indexed documents below.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Chat input ──
    if prompt := st.chat_input("Ask about your documents…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if not kb_ready:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Please upload and index documents first using the left panel.",
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
                        headers={"Authorization": f"Bearer {NV_KEY}", "Content-Type": "application/json"},
                        json={
                            "model": chat_model,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"},
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
                        st.session_state.chat_history.append({"role": "assistant", "content": err, "sources": []})
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
                        "content": "⚠️ Request timed out. Try nemotron-4b (Fast).",
                        "sources": [],
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error: {e}",
                        "sources": [],
                    })
            st.rerun()
