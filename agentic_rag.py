import streamlit as st
from agno.agent import Agent
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.google import Gemini  # ✅ Groq → Gemini
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.lancedb import LanceDb, SearchType
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Agentic RAG with Reasoning",
    page_icon="🌿",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Root & Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f8faf6 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #1a2e1a !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 50% at 0% 0%, rgba(34, 139, 34, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 100% 100%, rgba(144, 238, 144, 0.10) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(255,255,255,0.6) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(34, 139, 34, 0.15) !important;
    box-shadow: 2px 0 20px rgba(34,139,34,0.06) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Outfit', sans-serif !important;
    color: #1a2e1a !important;
}

/* ── Title ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    color: #1a2e1a !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.2rem !important;
    position: relative !important;
}

h1 span.accent {
    color: #228b22 !important;
}

h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #228b22 !important;
}

/* ── Text Input & TextArea ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1.5px solid rgba(34, 139, 34, 0.25) !important;
    border-radius: 12px !important;
    color: #1a2e1a !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 2px 8px rgba(34,139,34,0.04) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #228b22 !important;
    box-shadow: 0 0 0 3px rgba(34, 139, 34, 0.10) !important;
}

[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: #2d5a2d !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: #228b22 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(34,139,34,0.2) !important;
}

[data-testid="stButton"] button:hover {
    background: #1a6e1a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(34,139,34,0.28) !important;
}

[data-testid="stButton"] button:active {
    transform: translateY(0px) !important;
}

/* ── Secondary buttons (quick question chips) ── */
[data-testid="stButton"]:nth-child(1) button,
[data-testid="stButton"]:nth-child(2) button,
[data-testid="stButton"]:nth-child(3) button {
    background: #ffffff !important;
    color: #228b22 !important;
    border: 1.5px solid rgba(34,139,34,0.3) !important;
    box-shadow: 0 2px 8px rgba(34,139,34,0.08) !important;
}

[data-testid="stButton"]:nth-child(1) button:hover,
[data-testid="stButton"]:nth-child(2) button:hover,
[data-testid="stButton"]:nth-child(3) button:hover {
    background: #f0faf0 !important;
    border-color: #228b22 !important;
    box-shadow: 0 4px 14px rgba(34,139,34,0.15) !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(34, 139, 34, 0.12) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    background: #f0faf0 !important;
    border: 1px solid rgba(34, 139, 34, 0.2) !important;
    border-radius: 12px !important;
    color: #1a2e1a !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid rgba(34, 139, 34, 0.15) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 10px rgba(34,139,34,0.05) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    font-family: 'Outfit', sans-serif !important;
    color: #228b22 !important;
    font-size: 0.9rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #228b22 !important;
}

/* ── Markdown text ── */
p, li, span {
    font-family: 'Outfit', sans-serif !important;
    color: #2d3a2d !important;
    line-height: 1.75 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f5f0; }
::-webkit-scrollbar-thumb { background: #90c890; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #228b22; }

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid rgba(34,139,34,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 4px 20px rgba(34,139,34,0.06);
}

/* ── Metric boxes ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(34,139,34,0.15) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:14px; margin-bottom:4px;'>
  <div style='width:48px;height:48px;background:linear-gradient(135deg,#228b22,#90ee90);
  border-radius:14px;display:flex;align-items:center;justify-content:center;
  font-size:24px;box-shadow:0 4px 14px rgba(34,139,34,0.25);'>🌿</div>
  <div>
    <h1 style='margin:0;font-family:Playfair Display,serif;font-size:2.2rem;
    font-weight:700;color:#1a2e1a;'>Agentic RAG
    <span style='color:#228b22;'>— Reasoning AI</span></h1>
    <p style='margin:0;color:#5a7a5a;font-size:0.88rem;font-family:Outfit,sans-serif;'>
    Add any URL · Ask anything · Watch AI reason step-by-step</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── API Key ──────────────────────────────────────────────────────────────────
gemini_key = st.text_input(  # ✅ variable renamed: groq_key → gemini_key
    "🔑 Gemini API Key",
    type="password",
    value="",
    help="Get your key from https://aistudio.google.com/apikey"
)

if gemini_key:

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
                embedder=FastEmbedEmbedder(),
            ),
        )

    @st.cache_resource(show_spinner="🤖 Loading Gemini agent...")  # ✅ was: Loading Groq agent
    def load_agent(_kb: Knowledge) -> Agent:
        return Agent(
            model=Gemini(id="gemini-2.0-flash", api_key=gemini_key),  # ✅ was: Groq(id="llama-3.3-70b-versatile", api_key=groq_key)
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
        st.markdown("""
        <div style='padding:8px 0 16px;'>
          <p style='font-family:Playfair Display,serif;font-size:1.2rem;
          font-weight:700;color:#1a2e1a;margin:0;'>📚 Knowledge Sources</p>
          <p style='font-size:0.78rem;color:#5a7a5a;margin:4px 0 0;'>URLs loaded into AI memory</p>
        </div>
        """, unsafe_allow_html=True)

        for i, url in enumerate(st.session_state.knowledge_urls):
            st.markdown(f"""
            <div style='background:#f0faf0;border:1px solid rgba(34,139,34,0.2);
            border-left:3px solid #228b22;
            border-radius:8px;padding:8px 12px;margin:6px 0;font-size:0.73rem;
            color:#2d5a2d;word-break:break-all;font-family:Outfit,sans-serif;'>
            <span style='color:#90c890;font-weight:600;'>{i+1}.</span> {url}
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
                st.success("✅ Added to knowledge base!")
                st.rerun()
            else:
                st.error("Please enter a URL")

    # ── Query Section ─────────────────────────────────────────────────────────
    st.markdown("""
    <p style='font-family:Playfair Display,serif;font-size:1.3rem;
    font-weight:600;color:#1a2e1a;margin-bottom:12px;'>💬 Ask a Question</p>
    """, unsafe_allow_html=True)

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
                st.markdown("""
                <div style='background:#f0faf0;border:1px solid rgba(34,139,34,0.15);
                border-radius:14px;padding:14px 18px;margin-bottom:12px;'>
                <p style='font-family:Playfair Display,serif;color:#228b22;
                font-size:1.1rem;font-weight:600;margin:0;'>🧠 Reasoning Process</p>
                </div>""", unsafe_allow_html=True)
                reasoning_placeholder = st.container().empty()

            with col2:
                st.markdown("""
                <div style='background:#f0faf0;border:1px solid rgba(34,139,34,0.15);
                border-radius:14px;padding:14px 18px;margin-bottom:12px;'>
                <p style='font-family:Playfair Display,serif;color:#228b22;
                font-size:1.1rem;font-weight:600;margin:0;'>💡 Answer</p>
                </div>""", unsafe_allow_html=True)
                answer_placeholder = st.container().empty()

            citations = []
            answer_text = ""
            reasoning_text = ""

            with st.spinner("🌿 Thinking..."):
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
                st.markdown("""
                <p style='font-family:Playfair Display,serif;color:#228b22;
                font-size:1.1rem;font-weight:600;'>📚 Sources</p>
                """, unsafe_allow_html=True)
                for cite in citations:
                    title = cite.title or cite.url
                    st.markdown(f"- [{title}]({cite.url})")
        else:
            st.error("Please enter a question")

else:
    st.markdown("""
    <div style='background:#ffffff;border:1px solid rgba(34,139,34,0.2);
    border-radius:20px;padding:36px;margin-top:24px;
    box-shadow:0 8px 32px rgba(34,139,34,0.08);'>
      <div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>
        <div style='width:44px;height:44px;background:linear-gradient(135deg,#228b22,#90ee90);
        border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;'>🌿</div>
        <p style='font-family:Playfair Display,serif;font-size:1.4rem;
        font-weight:700;color:#1a2e1a;margin:0;'>Welcome!</p>
      </div>
      <p style='color:#3d5c3d;font-family:Outfit,sans-serif;font-size:0.95rem;'>
      Enter your <b style='color:#228b22;'>Gemini API Key</b> above to get started.</p>
      <p style='color:#5a7a5a;font-size:0.85rem;font-family:Outfit,sans-serif;margin-top:8px;'>
      Get your free key at 
      <a href='https://aistudio.google.com/apikey'
      style='color:#228b22;font-weight:600;text-decoration:none;
      border-bottom:1px solid rgba(34,139,34,0.3);'>aistudio.google.com</p></a>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='background:#ffffff;border:1px solid rgba(34,139,34,0.12);
border-radius:16px;padding:24px 28px;margin-top:8px;
box-shadow:0 4px 20px rgba(34,139,34,0.05);'>
  <p style='font-family:Playfair Display,serif;color:#1a2e1a;
  font-size:1.05rem;font-weight:700;margin-bottom:16px;'>📖 How This Works</p>
  <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
    <div style='background:#f0faf0;border-radius:10px;padding:12px 16px;
    border-left:3px solid #228b22;'>
      <p style='color:#228b22;font-weight:600;font-size:0.85rem;margin:0 0 4px;
      font-family:Outfit,sans-serif;'>1. Knowledge Loading</p>
      <p style='color:#5a7a5a;font-size:0.8rem;margin:0;font-family:Outfit,sans-serif;'>
      URLs processed & stored in LanceDB vector database</p>
    </div>
    <div style='background:#f0faf0;border-radius:10px;padding:12px 16px;
    border-left:3px solid #4caf50;'>
      <p style='color:#228b22;font-weight:600;font-size:0.85rem;margin:0 0 4px;
      font-family:Outfit,sans-serif;'>2. FastEmbed Embedder</p>
      <p style='color:#5a7a5a;font-size:0.8rem;margin:0;font-family:Outfit,sans-serif;'>
      Text converted to vectors locally — no extra API key needed</p>
    </div>
    <div style='background:#f0faf0;border-radius:10px;padding:12px 16px;
    border-left:3px solid #66bb6a;'>
      <p style='color:#228b22;font-weight:600;font-size:0.85rem;margin:0 0 4px;
      font-family:Outfit,sans-serif;'>3. Reasoning Tools</p>
      <p style='color:#5a7a5a;font-size:0.8rem;margin:0;font-family:Outfit,sans-serif;'>
      AI thinks step-by-step before answering</p>
    </div>
    <div style='background:#f0faf0;border-radius:10px;padding:12px 16px;
    border-left:3px solid #81c784;'>
      <p style='color:#228b22;font-weight:600;font-size:0.85rem;margin:0 0 4px;
      font-family:Outfit,sans-serif;'>4. Gemini 2.0 Flash</p>
      <p style='color:#5a7a5a;font-size:0.8rem;margin:0;font-family:Outfit,sans-serif;'>
      Generates final answer with citations</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)