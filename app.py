import streamlit as st
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import List

st.set_page_config(page_title="MilIntelX (beta)", layout="wide")
st.title("🛡️ MilIntelX (Beta)")

NV_KEY = st.secrets["NVIDIA_API_KEY"]

# --- 1. DYNAMIC EMBEDDING CLASS ---
class NemotronEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.model = model_name
        self.headers = {"Authorization": f"Bearer {NV_KEY}", "Content-Type": "application/json"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            payload = {"input": [text], "model": self.model, "input_type": "passage", "encoding_format": "float"}
            res = requests.post(self.url, headers=self.headers, json=payload).json()
            embeddings.append(res['data'][0]['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        payload = {"input": [text], "model": self.model, "input_type": "query", "encoding_format": "float"}
        res = requests.post(self.url, headers=self.headers, json=payload).json()
        return res['data'][0]['embedding']

# --- 2. SESSION STATE ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "filenames" not in st.session_state: st.session_state.filenames = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- 3. SIDEBAR: THE MODEL CHANGER ---
with st.sidebar:
    st.header("🧠 Select Brains")
    
    # Choose Embedding Model (The Searcher)
    # nv-embedqa-e5-v5 is very strong for QA
    embed_choice = st.selectbox("Search Model (Embedder):", [
        "nvidia/llama-nemotron-embed-1b-v2",
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/llama-3.2-nv-embedqa-1b-v2"
    ])
    
    # Choose Chat Model (The Thinker)
    # Nemotron-70B is specifically tuned to be "smarter" than standard Llama
    chat_choice = st.selectbox("Logic Model (LLM):", [
        "meta/llama-3.1-70b-instruct",
        "nvidia/nemotron-70b-instruct",
        "meta/llama-3.1-405b-instruct"
    ])
    
    st.warning("⚠️ If you change the 'Search Model', you MUST click Reset Brain and re-upload PDFs!")

    st.divider()
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🏗️ Build Knowledge Base") and uploaded_files:
        with st.spinner("Indexing..."):
            all_chunks = []
            for file in uploaded_files:
                if file.name not in st.session_state.filenames:
                    reader = PdfReader(file)
                    text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_text(text)
                    for c in chunks: all_chunks.append({"text": c, "source": file.name})
                    st.session_state.filenames.append(file.name)
            
            if all_chunks:
                embedder = NemotronEmbeddings(embed_choice)
                new_db = FAISS.from_texts([c["text"] for c in all_chunks], embedder, metadatas=[{"source": c["source"]} for c in all_chunks])
                if st.session_state.vector_db: st.session_state.vector_db.merge_from(new_db)
                else: st.session_state.vector_db = new_db
                st.success("Brain Ready!")

    if st.button("🗑️ Reset Brain"):
        st.session_state.vector_db = None
        st.session_state.filenames = []
        st.rerun()

# --- 4. CHAT ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your intel..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vector_db:
        # Step A: Retrieval
        docs = st.session_state.vector_db.similarity_search(prompt, k=5)
        context = "\n\n".join([f"FROM {d.metadata['source']}: {d.page_content}" for d in docs])

        # Step B: Logic (using chat_choice)
        with st.chat_message("assistant"):
            with st.spinner(f"Using {chat_choice}..."):
                res = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {NV_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": chat_choice,
                        "messages": [
                            {"role": "system", "content": "You are a military intelligence analyst. Use the provided context to answer. Be precise, technical, and strict."},
                            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"}
                        ],
                        "temperature": 0.1
                    }
                ).json()
                answer = res['choices'][0]['message']['content']
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.info("Index PDFs first.")
