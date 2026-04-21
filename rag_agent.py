import streamlit as st
from agno.agent import Agent
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Agentic RAG — Reasoning AI",
    page_icon="🧠",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    font-family: 'DM Mono', monospace !important;
    color: #e2e0ff !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(108, 60, 255, 0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(0, 210, 190, 0.10) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid rgba(108, 60, 255, 0.2) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'DM Mono', monospace !important;
    color: #c4bfff !important;
}

/* ── Title ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa, #00d2be, #6c3cff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -1px !important;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #a78bfa !important;
}

/* ── Text Input & TextArea ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #13131f !important;
    border: 1px solid rgba(108, 60, 255, 0.35) !important;
    border-radius: 10px !important;
    color: #e2e0ff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.12) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6c3cff, #a78bfa) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 22px !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.5px !important;
}

[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(108, 60, 255, 0.2) !important;
}

/* ── Info / Success boxes ── */
[data-testid="stAlert"] {
    background: #13131f !important;
    border: 1px solid rgba(108, 60, 255, 0.3) !important;
    border-radius: 12px !important;
    color: #c4bfff !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #13131f !important;
    border: 1px solid rgba(108, 60, 255, 0.2) !important;
    border-radius: 12px !important;
}

/* ── Expander Text Fix ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    font-family: 'DM Mono', monospace !important;
    color: #a78bfa !important;
    -webkit-text-fill-color: #a78bfa !important;
    font-size: 0.9rem !important;
    letter-spacing: normal !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #a78bfa !important;
}

/* ── Markdown text ── */
p, li, span {
    font-family: 'DM Mono', monospace !important;
    color: #c4bfff !important;
    line-height: 1.7 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #6c3cff; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h1>🧠 Agentic RAG — Reasoning AI</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#7c6fcd; font-size:0.9rem; margin-top:-8px;'>
Add any URL · Ask anything · Watch AI reason step-by-step
</p>
""", unsafe_allow_html=True)

st.divider()

# ─── API Key ──────────────────────────────────────────────────────────────────
google_key = st.text_input(
    "🔑 Google Gemini API Key",
    type="password",
    value="",
    help="Get your key from https://aistudio.google.com/apikey"
)

if google_key:

    if 'knowledge_urls' not in st.session_state:
        st.session_state.knowledge_urls = [
            "https://www.theunwindai.com/p/mcp-vs-a2a-complementing-or-supplementing"
        ]
    if 'urls_loaded' not in st.session_state:
        st.session_state.urls_loaded = set()

    @st.cache_resource(show_spinner="⚡ Initializing knowledge base...")
    def load_knowledge() -> Knowledge:
        return Knowledge(
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="agno_docs",
                search_type=SearchType.vector,
                embedder=GeminiEmbedder(api_key=google_key),
            ),
        )

    @st.cache_resource(show_spinner="🤖 Loading Gemini agent...")
    def load_agent(_kb: Knowledge) -> Agent:
        return Agent(
            model=Gemini(id="gemini-2.5-flash", api_key=google_key),
            knowledge=_kb,
            search_knowledge=True,
            tools=[ReasoningTools(add_instructions=True)],
            instructions=[
                "Include sources in your response.",
                "Always search your knowledge before answering.",
            ],
            markdown=True,
        )

    knowledge = load_knowledge()

    for url in st.session_state.knowledge_urls:
        if url not in st.session_state.urls_loaded:
            knowledge.add_content(url=url)
            st.session_state.urls_loaded.add(url)

    agent = load_agent(knowledge)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📚 Knowledge Sources")
        st.markdown("<p style='font-size:0.8rem;color:#7c6fcd;'>URLs loaded into AI memory:</p>", unsafe_allow_html=True)

        for i, url in enumerate(st.session_state.knowledge_urls):
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid rgba(108,60,255,0.25);
            border-radius:8px;padding:8px 12px;margin:6px 0;font-size:0.75rem;
            color:#a78bfa;word-break:break-all;'>
            {i+1}. {url}
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        new_url = st.text_input("➕ Add new URL", placeholder="https://example.com/article")

        if st.button("Add to Knowledge Base", type="primary"):
            if new_url:
                if new_url not in st.session_state.knowledge_urls:
                    st.session_state.knowledge_urls.append(new_url)
                with st.spinner("Loading..."):
                    if new_url not in st.session_state.urls_loaded:
                        knowledge.add_content(url=new_url)
                        st.session_state.urls_loaded.add(new_url)
                st.success("✅ Added!")
                st.rerun()
            else:
                st.error("Please enter a URL")

    # ── Query Section ─────────────────────────────────────────────────────────
    st.markdown("### 💬 Ask a Question")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔌 What is MCP?", use_container_width=True):
            st.session_state.query = "What is MCP (Model Context Protocol) and how does it work?"
    with col2:
        if st.button("⚡ MCP vs A2A", use_container_width=True):
            st.session_state.query = "How do MCP and A2A protocols differ?"
    with col3:
        if st.button("🤝 Agent Communication", use_container_width=True):
            st.session_state.query = "How do MCP and A2A work together in AI agent systems?"

    query = st.text_area(
        "Your question:",
        value=st.session_state.get("query", "What is the difference between MCP and A2A protocols?"),
        height=100,
    )

    if st.button("🚀 Get Answer with Reasoning", type="primary"):
        if query:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### 🧠 Reasoning Process")
                reasoning_placeholder = st.container().empty()

            with col2:
                st.markdown("### 💡 Answer")
                answer_placeholder = st.container().empty()

            citations = []
            answer_text = ""
            reasoning_text = ""

            with st.spinner("🔍 Thinking..."):
                for chunk in agent.run(query, stream=True, stream_events=True):
                    if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                        reasoning_text = chunk.reasoning_content
                        reasoning_placeholder.markdown(reasoning_text, unsafe_allow_html=True)

                    if hasattr(chunk, 'content') and chunk.content and isinstance(chunk.content, str):
                        answer_text += chunk.content
                        answer_placeholder.markdown(answer_text, unsafe_allow_html=True)

                    if hasattr(chunk, 'citations') and chunk.citations:
                        if hasattr(chunk.citations, 'urls') and chunk.citations.urls:
                            citations = chunk.citations.urls

            if citations:
                st.divider()
                st.markdown("### 📚 Sources")
                for cite in citations:
                    title = cite.title or cite.url
                    st.markdown(f"- [{title}]({cite.url})")
        else:
            st.error("Please enter a question")

else:
    st.markdown("""
    <div style='background:#13131f;border:1px solid rgba(108,60,255,0.3);
    border-radius:16px;padding:28px;margin-top:20px;'>
    <h3 style='color:#a78bfa;font-family:Syne,sans-serif;'>👋 Welcome!</h3>
    <p style='color:#7c6fcd;'>Enter your <b style='color:#a78bfa;'>Google Gemini API Key</b> above to get started.</p>
    <p style='color:#7c6fcd;font-size:0.85rem;'>Get your free key at 
    <a href='https://aistudio.google.com/apikey' style='color:#a78bfa;'>aistudio.google.com</a></p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='background:#13131f;border:1px solid rgba(108,60,255,0.2);
border-radius:12px;padding:20px 24px;margin-top:8px;'>
<p style='color:#a78bfa;font-family:Syne,sans-serif;font-size:1rem;font-weight:600;margin-bottom:12px;'>
📖 How This Works
</p>
<p style='color:#7c6fcd;font-size:0.85rem;margin:6px 0;'>
<b style='color:#a78bfa;'>1. Knowledge Loading</b> — URLs processed & stored in LanceDB vector database
</p>
<p style='color:#7c6fcd;font-size:0.85rem;margin:6px 0;'>
<b style='color:#a78bfa;'>2. Gemini Embeddings</b> — Text converted to vectors using Google Gemini
</p>
<p style='color:#7c6fcd;font-size:0.85rem;margin:6px 0;'>
<b style='color:#a78bfa;'>3. Reasoning Tools</b> — AI thinks step-by-step before answering
</p>
<p style='color:#7c6fcd;font-size:0.85rem;margin:6px 0;'>
<b style='color:#a78bfa;'>4. Gemini 2.5 Flash</b> — Generates final answer with citations
</p>
</div>
""", unsafe_allow_html=True)