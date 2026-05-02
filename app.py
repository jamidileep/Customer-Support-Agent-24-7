"""
app.py — BrightSmile Dental Support Agent
Run: streamlit run app.py
"""

import streamlit as st
import json, re, os, tempfile, base64
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from gtts import gTTS

# ✅ Load env (local) + Streamlit secrets (cloud)
load_dotenv()

if hasattr(st, "secrets"):
    for key, val in st.secrets.items():
        os.environ[key] = val

st.set_page_config(page_title="BrightSmile Support", page_icon="🦷", layout="centered")

# ---------------- UI ----------------
st.title("🦷 BrightSmile Dental Support")
st.caption("Book appointments · Get info · 24/7 support")
st.divider()

# ---------------- Speech to Text ----------------
def speech_to_text(audio_bytes) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("llm_api"))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3"
            )

        os.unlink(tmp_path)
        return transcription.text.strip()

    except Exception as e:
        return f"❌ Transcription error: {str(e)}"

# ---------------- Text to Speech ----------------
def text_to_speech(text: str) -> str:
    try:
        clean = re.sub(r"[#*_`]", "", text)
        clean = re.sub(r"http\S+", "link", clean)
        clean = clean[:500]

        tts = gTTS(text=clean, lang="en", slow=False)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            tmp_path = f.name

        with open(tmp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        os.unlink(tmp_path)
        return audio_b64

    except Exception:
        return ""

# ---------------- Load Agent ----------------
@st.cache_resource
def load_agent():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.tools import create_retriever_tool, StructuredTool
    from langchain.chat_models import init_chat_model
    from langgraph.graph import END, StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing import Annotated, Sequence, Optional
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field

    # ---- Load PDF ----
    loader = PyPDFLoader("BrightSmile_Dental_Hospital.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        splits,
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "retriever_tool",
        "Search hospital info"
    )

    # ---- Calendar tools ----
    from calendar_mcp_server import handle_book, handle_cancel, handle_availability

    class BookInput(BaseModel):
        patient_name: str
        date: str
        time: str
        treatment: str = "Dental Appointment"
        patient_email: Optional[str] = ""

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
        StructuredTool.from_function(book_fn, "book_appointment", args_schema=BookInput),
        StructuredTool.from_function(cancel_fn, "cancel_appointment", args_schema=CancelInput),
        StructuredTool.from_function(availability_fn, "check_availability", args_schema=AvailabilityInput),
    ]

    # ---- LLM ----
    llm = init_chat_model(
        model="groq:openai/gpt-oss-120b",
        api_key=os.getenv("llm_api")
    )

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def assistant(state):
        system = SystemMessage(content="You are a helpful dental receptionist.")
        response = llm.bind_tools(tools).invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")

    return graph.compile()

# ---------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# ---------------- Chat UI ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

# ---------------- Main Logic ----------------
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = load_agent()

                st.session_state.agent_messages.append(HumanMessage(content=user_input))

                result = graph.invoke({"messages": st.session_state.agent_messages})

                response = result["messages"][-1].content
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

                st.session_state.agent_messages = list(result["messages"])

            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # ---- Audio output ----
    audio_b64 = text_to_speech(response)
    if audio_b64:
        st.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

    st.rerun()
