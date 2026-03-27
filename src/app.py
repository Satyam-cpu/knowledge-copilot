import streamlit as st
import os
import sys

# Path add karo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import KnowledgeCopilot

# ── PAGE CONFIG
st.set_page_config(
    page_title="Knowledge Copilot",
    page_icon="🧠",
    layout="wide"
)

# ── CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E2761, #028090);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .source-box {
        background: #f0f7ff;
        border-left: 3px solid #028090;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 12px;
    }
    .confidence-high { color: #047857; font-weight: bold; }
    .confidence-mid  { color: #D97706; font-weight: bold; }
    .confidence-low  { color: #B91C1C; font-weight: bold; }
    .gap-warning {
        background: #FEF3C7;
        border: 1px solid #D97706;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER
st.markdown("""
<div class="main-header">
    <h1>🧠 Enterprise Knowledge Copilot</h1>
    <p>RAG + Agentic AI | Powered by Llama 3.1</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD COPILOT
@st.cache_resource
def load_copilot():
    return KnowledgeCopilot()

copilot = load_copilot()

# ── SIDEBAR
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.success("✅ 5 Documents Indexed")
    st.info("📄 it_policy.txt\n📄 vpn_guide.txt\n📄 onboarding.txt\n📄 incident_sop.txt\n📄 leave_policy.txt")

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    
    questions = [
        "Password reset kaise karte hain?",
        "VPN setup kaise karo?",
        "Leave apply karne ka process?",
        "P0 incident mein kya karna chahiye?",
        "Naye employee ka Day 1 plan?",
    ]
    
    for q in questions:
        if st.button(q, use_container_width=True):
            st.session_state.selected_q = q

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_gaps" not in st.session_state:
        st.session_state.total_gaps = 0

    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Gaps", st.session_state.total_gaps)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

# ── CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purane messages dikhao
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Sources dikhao
        if "sources" in msg:
            with st.expander("📄 Sources dekho"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        📄 <b>{src['file']}</b><br>
                        {src['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Confidence dikhao
        if "confidence" in msg:
            conf = msg["confidence"]
            if conf >= 0.7:
                st.markdown(f"🟢 **Confidence: {conf:.0%}**")
            elif conf >= 0.4:
                st.markdown(f"🟡 **Confidence: {conf:.0%}**")
            else:
                st.markdown(f"🔴 **Confidence: {conf:.0%}** — Expert se verify karo!")

# ── SELECTED QUESTION (sidebar se)
if "selected_q" in st.session_state:
    question = st.session_state.selected_q
    del st.session_state.selected_q
else:
    question = None

# ── CHAT INPUT
user_input = st.chat_input("Kuch bhi poochho apni company ke baare mein...")

# Input determine karo
final_question = user_input or question

if final_question:
    # User message dikhao
    with st.chat_message("user"):
        st.write(final_question)

    st.session_state.messages.append({
        "role": "user",
        "content": final_question
    })

    # Answer generate karo
    with st.chat_message("assistant"):
        with st.spinner("🔍 Documents search ho rahe hain..."):
            result = copilot.ask(final_question)

        # Answer dikhao
        st.write(result["answer"])

        # Confidence
        conf = result["confidence"]
        if conf >= 0.7:
            st.markdown(f"🟢 **Confidence: {conf:.0%}**")
        elif conf >= 0.4:
            st.markdown(f"🟡 **Confidence: {conf:.0%}**")
        else:
            st.markdown(f"🔴 **Confidence: {conf:.0%}** — Expert se verify karo!")
            st.markdown("""
            <div class="gap-warning">
                ⚠️ <b>Knowledge Gap Detected!</b><br>
                Is topic pe documentation nahi mili.
                Knowledge Gap log ho gayi hai.
            </div>
            """, unsafe_allow_html=True)
            st.session_state.total_gaps += 1

        # Sources
        sources_data = []
        with st.expander("📄 Sources dekho"):
            for doc in result["sources"]:
                file = os.path.basename(
                    doc.metadata.get("source", "Unknown")
                )
                content = doc.page_content[:150] + "..."
                sources_data.append({
                    "file": file,
                    "content": content
                })
                st.markdown(f"""
                <div class="source-box">
                    📄 <b>{file}</b><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)

    # History mein save karo
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "confidence": conf,
        "sources": sources_data
    })

    # Stats update
    st.session_state.total_queries += 1