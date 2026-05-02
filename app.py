"""
BrightSmile Dental Support Agent — Streamlit Cloud Safe Version
Run: streamlit run app.py
"""

import streamlit as st
import os, re, tempfile, base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from gtts import gTTS

# ─────────────────────────────────────────────
# ENV + STREAMLIT SECRETS
# ─────────────────────────────────────────────
load_dotenv()

if hasattr(st, "secrets"):
    for k, v in st.secrets.items():
        os.environ[k] = v

st.set_page_config(page_title="BrightSmile Dental AI", page_icon="🦷")

st.title("🦷 BrightSmile Dental Support")
st.caption("Book appointments · Get info · 24/7 AI Assistant")
st.divider()

# ─────────────────────────────────────────────
# SPEECH → TEXT (Groq Whisper)
# ─────────────────────────────────────────────
def speech_to_text(audio_bytes: bytes) -> str:
    try:
        from groq import Groq

        client = Groq(api_key=os.getenv("llm_api"))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name

        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f
            )

        os.unlink(path)
        return result.text.strip()

    except Exception as e:
        return f"❌ Transcription error: {e}"


# ─────────────────────────────────────────────
# TEXT → SPEECH
# ─────────────────────────────────────────────
def text_to_speech(text: str) -> str:
    try:
        clean = re.sub(r"[#*_`]", "", text)[:500]
        tts = gTTS(clean, lang="en")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            path = f.name

        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        os.unlink(path)
        return audio_b64

    except:
        return ""


# ─────────────────────────────────────────────
# LOAD AGENT
# ─────────────────────────────────────────────
@st.cache_resource
def load_agent():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import create_retriever_tool, StructuredTool
    from langchain.chat_models import init_chat_model
    from langgraph.graph import StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing import TypedDict, Sequence, Annotated
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel

    # ── PDF ──
    loader = PyPDFLoader("BrightSmile_Dental_Hospital.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)

    vector = FAISS.from_documents(
        docs,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    retriever_tool = create_retriever_tool(
        vector.as_retriever(),
        "retriever_tool",
        "Search dental hospital information"
    )

    # ── Calendar tools ──
    from calendar_mcp_server import handle_book, handle_cancel, handle_availability

    class BookInput(BaseModel):
        patient_name: str
        date: str
        time: str
        treatment: str = "Dental Appointment"
        patient_email: str = ""

    class CancelInput(BaseModel):
        event_id: str

    class AvailabilityInput(BaseModel):
        date: str

    def book_fn(**kwargs):
        return handle_book(kwargs)

    def cancel_fn(event_id: str):
        return handle_cancel({"event_id": event_id})

    def availability_fn(date: str):
        return handle_availability({"date": date})

    tools = [
        retriever_tool,

        StructuredTool.from_function(
            book_fn,
            name="book_appointment",
            description="Book dental appointment",
            args_schema=BookInput,
        ),

        StructuredTool.from_function(
            cancel_fn,
            name="cancel_appointment",
            description="Cancel appointment",
            args_schema=CancelInput,
        ),

        StructuredTool.from_function(
            availability_fn,
            name="check_availability",
            description="Check available slots",
            args_schema=AvailabilityInput,
        ),
    ]

    llm = init_chat_model(
        model="groq:openai/gpt-oss-120b",
        api_key=os.getenv("llm_api")
    )

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def assistant(state):
        sys = SystemMessage(content="You are a dental receptionist assistant.")
        res = llm.bind_tools(tools).invoke([sys] + list(state["messages"]))
        return {"messages": [res]}

    graph = StateGraph(State)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None


# ─────────────────────────────────────────────
# CHAT UI
# ─────────────────────────────────────────────
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ─────────────────────────────────────────────
# INPUTS (TEXT + AUDIO)
# ─────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

audio = st.audio_input("🎙️ Speak")

if audio:
    audio_bytes = audio.read()
    audio_id = hash(audio_bytes)

    if audio_id != st.session_state.last_audio:
        st.session_state.last_audio = audio_id
        with st.spinner("Transcribing..."):
            user_input = speech_to_text(audio_bytes)

# ─────────────────────────────────────────────
# MAIN AGENT FLOW
# ─────────────────────────────────────────────
def run(query):
    st.session_state.history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = load_agent()
                st.session_state.messages.append(HumanMessage(content=query))
                result = graph.invoke({"messages": st.session_state.messages})
                answer = result["messages"][-1].content
                st.session_state.messages = list(result["messages"])
            except Exception as e:
                answer = f"⚠️ Error: {e}"

        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})

    audio_b64 = text_to_speech(answer)
    if audio_b64:
        st.markdown(
            f'<audio autoplay src="data:audio/mp3;base64,{audio_b64}"></audio>',
            unsafe_allow_html=True
        )

    st.rerun()


if user_input:
    run(user_input)
