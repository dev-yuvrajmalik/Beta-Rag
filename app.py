import streamlit as st
import requests
import json
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import List

# --- CONFIG ---
st.set_page_config(page_title="MilIntelX (Beta)", layout="wide")
st.title("🛡️ MilIntelX (Beta)")

# Fetch API key securely from Streamlit secrets
NV_KEY = st.secrets["NVIDIA_API_KEY"]

# --- 1. DYNAMIC EMBEDDING CLASS (WITH SAFE BATCHING) ---
class NemotronEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.model = model_name
        self.headers = {"Authorization": f"Bearer {NV_KEY}", "Content-Type": "application/json"}

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Embeds multiple documents in batches to avoid 'Payload Too Large' errors."""
        all_embeddings =[]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"input": batch, "model": self.model, "input_type": "passage", "encoding_format": "float"}
            res = requests.post(self.url, headers=self.headers, json=payload).json()
            
            # Extract embeddings from the batched response
            if 'data' in res:
                all_embeddings.extend([item['embedding'] for item in res['data']])
            else:
                raise ValueError(f"API Error during embedding: {res}")
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        payload = {"input": [text], "model": self.model, "input_type": "query", "encoding_format": "float"}
        res = requests.post(self.url, headers=self.headers, json=payload).json()
        return res['data'][0]['embedding']

# --- 2. SESSION STATE ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "filenames" not in st.session_state: st.session_state.filenames =[]
if "chat_history" not in st.session_state: st.session_state.chat_history =[]

# --- 3. STREAMING GENERATOR ---
def stream_nvidia_response(response):
    """Parses Server-Sent Events (SSE) from the NVIDIA API to yield tokens for streaming."""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line.replace("data: ", "")
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk['choices'][0]['delta'].get('content', '')
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue

# --- 4. SIDEBAR: UI & KNOWLEDGE BUILDER ---
with st.sidebar:
    st.header("🧠 Benchmarked Brains")
    
    embed_choice = st.selectbox("Search Model (Embedder):",[
        "nvidia/nv-embedqa-e5-v5",
        "nvidia/llama-nemotron-embed-1b-v2",
        "nvidia/llama-3.2-nv-embedqa-1b-v2"
    ])
    
    chat_choice = st.selectbox("Logic Model (LLM):",[
        "nvidia/llama-3.1-nemotron-70b-instruct", 
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-405b-instruct",
        "nvidia/nemotron-4b-instruct-v1"
    ])
    
    st.divider()
    uploaded_files = st.file_uploader("Upload Intel (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("🏗️ Build Knowledge Base") and uploaded_files:
        all_chunks =[]
        
        # UI Elements for UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Extracting and Chunking pages...")
        for file_idx, file in enumerate(uploaded_files):
            if file.name not in st.session_state.filenames:
                reader = PdfReader(file)
                
                # PAGE-LEVEL ITERATION FOR PRECISE CITATIONS
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        chunks = RecursiveCharacterTextSplitter(
                            chunk_size=900, 
                            chunk_overlap=150
                        ).split_text(page_text)
                        
                        for c in chunks: 
                            # Appending exact page source
                            all_chunks.append({"text": c, "source": f"{file.name} (Page {i+1})"})
                            
                st.session_state.filenames.append(file.name)
            
            # Update Progress Bar for extraction
            progress_bar.progress((file_idx + 1) / len(uploaded_files) * 0.5)
        
        if all_chunks:
            status_text.text(f"Embedding {len(all_chunks)} chunks in batches...")
            embedder = NemotronEmbeddings(embed_choice)
            
            # FAISS Indexing
            new_db = FAISS.from_texts(
                texts=[c["text"] for c in all_chunks], 
                embedding=embedder, 
                metadatas=[{"source": c["source"]} for c in all_chunks]
            )
            
            progress_bar.progress(1.0)
            status_text.text("Merging databases...")
            
            if st.session_state.vector_db: 
                st.session_state.vector_db.merge_from(new_db)
            else: 
                st.session_state.vector_db = new_db
                
            status_text.success("Brain Ready! Knowledge Base Indexed.")
        else:
            status_text.info("No new content found to index.")

    if st.button("🗑️ Purge Intel Base"):
        st.session_state.vector_db = None
        st.session_state.filenames =[]
        st.session_state.chat_history =[]
        st.rerun()

# --- 5. CHAT INTERFACE ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])

if prompt := st.chat_input("Query your intelligence base..."):
    # 1. Add user message to state and UI
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    # 2. Guardrails
    if not st.session_state.vector_db:
        st.warning("⚠️ Please upload and index PDFs before querying.")
        st.stop()

    # 3. Retrieve Context
    docs = st.session_state.vector_db.similarity_search(prompt, k=6)
    
    # Format context WITH precise citations
    context = "\n\n".join([f"SOURCE: [{d.metadata['source']}]\nCONTENT: {d.page_content}" for d in docs])
    
    # 4. Generate & Stream Output
    with st.chat_message("assistant"):
        try:
            payload = {
                "model": chat_choice,
                "messages":[
                    {
                        "role": "system", 
                        "content": (
                            "You are a precise military analyst. Use the provided context to answer the user's question. "
                            "ALWAYS cite the source file and page number inline when providing facts. "
                            "If the answer is not in the context, say 'I do not have this information in my current files.' "
                            "Do not hallucinate."
                        )
                    },
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"}
                ],
                "temperature": 0.1,
                "stream": True # Enable Streaming
            }
            
            res = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {NV_KEY}", "Content-Type": "application/json"},
                json=payload,
                stream=True
            )
            
            if res.status_code == 200:
                # Modern Streamlit Streaming
                full_answer = st.write_stream(stream_nvidia_response(res))
                st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
            else:
                st.error(f"Error {res.status_code}: Model '{chat_choice}' unavailable. Details: {res.text}")
                
        except Exception as e:
            st.error(f"Critical System Error: {e}")
