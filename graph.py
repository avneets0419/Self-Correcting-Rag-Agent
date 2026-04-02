from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from retriever import build_retriever
from tools import web_search
from dotenv import load_dotenv
import json, re, os

load_dotenv()

# Two models — fast/cheap for grading, good for generation
grader_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
gen_llm    = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

retriever = build_retriever()

# --- State ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    retries: int
    web_fallback: bool

# --- Nodes ---
def retrieve(state):
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs, "retries": state.get("retries", 0)}

def grade_documents(state):
    question = state["question"]
    docs = state["documents"]

    prompt = f"""You are a relevance grader. Given a question and {len(docs)} document chunks,
reply with a JSON list of 'yes' or 'no' for each chunk in order.
Reply ONLY with a JSON array like: ["yes", "no", "yes", "yes"]
No explanation, no extra text.

Question: {question}

""" + "\n\n".join([f"Chunk {i+1}: {d.page_content[:300]}" for i, d in enumerate(docs)])

    result = grader_llm.invoke(prompt).content.strip()

    try:
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        grades = json.loads(match.group()) if match else []
        relevant = [doc for doc, grade in zip(docs, grades) if "yes" in str(grade).lower()]
    except:
        relevant = docs  # fallback: keep all if parsing fails

    return {**state, "documents": relevant}

def rewrite_query(state):
    prompt = f"""Rewrite this question to be more specific for document retrieval.
Return ONLY the rewritten question, nothing else.

Original: {state['question']}
Rewritten:"""
    new_q = grader_llm.invoke(prompt).content.strip()
    return {**state, "question": new_q, "retries": state["retries"] + 1}

def web_fallback_search(state):
    results = web_search(state["question"])
    fallback_doc = Document(page_content=results, metadata={"source": "web"})
    return {**state, "documents": [fallback_doc], "web_fallback": True}

def generate(state):
    context = "\n\n".join([d.page_content for d in state["documents"]])
    source = "web search" if state.get("web_fallback") else "knowledge base"

    prompt = f"""You are an expert AI assistant. Answer the question using only the context below.
Be clear, concise, and factual. If the context doesn't contain the answer, say so.

Context (from {source}):
{context}

Question: {state['question']}

Answer:"""

    answer = gen_llm.invoke(prompt).content
    return {**state, "generation": answer}

def check_hallucination(state):
    # Skip check entirely for web fallback — already grounded
    if state.get("web_fallback"):
        return {**state, "_grounded": True}

    context = "\n\n".join([d.page_content[:300] for d in state["documents"]])
    prompt = f"""Does this answer use only facts present in the context?
Reply with only 'yes' or 'no'.

Context: {context}

Answer: {state['generation']}

Grounded:"""

    result = grader_llm.invoke(prompt).content.strip().lower()
    return {**state, "_grounded": "yes" in result}

# --- Routing ---
def route_after_grading(state):
    if len(state["documents"]) >= 2:
        return "generate"
    elif state.get("retries", 0) < 2:
        return "rewrite"
    else:
        return "web_fallback"

def route_after_hallucination_check(state):
    if state.get("_grounded", True):
        return END
    return "generate"

# --- Build Graph ---
def build_graph():
    g = StateGraph(GraphState)

    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_fallback", web_fallback_search)
    g.add_node("generate", generate)
    g.add_node("check_hallucination", check_hallucination)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")

    g.add_conditional_edges("grade_documents", route_after_grading, {
        "generate":    "generate",
        "rewrite":     "rewrite_query",
        "web_fallback":"web_fallback",
    })

    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("web_fallback", "generate")
    g.add_edge("generate", "check_hallucination")

    g.add_conditional_edges("check_hallucination", route_after_hallucination_check, {
        END:        END,
        "generate": "generate",
    })

    return g.compile()