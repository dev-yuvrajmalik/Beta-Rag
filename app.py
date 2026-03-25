import streamlit as st
import requests
import json
import time
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import List

# \u2500\u2500\u2500 PAGE CONFIG \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.set_page_config(
    page_title="MilIntelX Pro",
    page_icon="\ud83d\udee1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# \u2500\u2500\u2500 APPLE-STYLE CSS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  /* \u2500\u2500 Root Variables \u2500\u2500 */
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

  /* \u2500\u2500 Global Reset \u2500\u2500 */
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

  /* \u2500\u2500 Hide Streamlit Chrome \u2500\u2500 */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* \u2500\u2500 Scrollbar \u2500\u2500 */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-active); border-radius: 99px; }

  /* \u2500\u2500 Hero Header \u2500\u2500 */
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

  /* \u2500\u2500 Status Pill \u2500\u2500 */
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

  /* \u2500\u2500 Chat Container \u2500\u2500 */
  .chat-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(20px);
  }

  /* \u2500\u2500 Chat Messages \u2500\u2500 */
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
    border
