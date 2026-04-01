import streamlit as st
from graph import build_graph
from retriever import build_retriever
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Research RAG Agent", page_icon="🤖", layout="centered")

st.title("🤖 Self-Correcting RAG Agent")
st.caption("Powered by LangGraph · Groq · FAISS · Tavily")

@st.cache_resource
def get_retriever():
    with st.spinner("Loading knowledge base..."):
        return build_retriever()

@st.cache_resource
def get_graph(_retriever):
    return build_graph()

retriever = get_retriever()
graph = get_graph(retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources"):
                for i, chunk in enumerate(msg["sources"]):
                    src = chunk["source"]
                    st.markdown(f"**Chunk {i+1}** — `{src}`")
                    st.caption(chunk["content"])
            if msg.get("web_fallback"):
                st.info("⚠️ KB had low relevance — fell back to web search")
            if msg.get("retries", 0) > 0:
                st.info(f"🔄 Query rewritten {msg['retries']} time(s) for better retrieval")

question = st.chat_input("Ask anything about AI companies, models, or research...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving → Grading → Generating..."):
            result = graph.invoke({
                "question": question,
                "documents": [],
                "generation": "",
                "retries": 0,
                "web_fallback": False
            })

        answer = result["generation"]
        st.markdown(answer)

        sources = [
            {
                "source": doc.metadata.get("source", "KB"),
                "content": doc.page_content[:300]
            }
            for doc in result["documents"]
        ]

        with st.expander("📄 Sources"):
            for i, chunk in enumerate(sources):
                st.markdown(f"**Chunk {i+1}** — `{chunk['source']}`")
                st.caption(chunk["content"])

        if result.get("web_fallback"):
            st.info("⚠️ Knowledge base had low relevance — fell back to web search")
        if result.get("retries", 0) > 0:
            st.info(f"🔄 Query rewritten {result['retries']} time(s) for better retrieval")
            st.info(f"This is the rewritten query used for retrieval: `{result['question']}`")
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "web_fallback": result.get("web_fallback", False),
        "retries": result.get("retries", 0)
    })