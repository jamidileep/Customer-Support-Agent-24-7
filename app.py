"""
app.py — BrightSmile Dental Support Agent
Run: streamlit run app.py
"""

import streamlit as st
import re, os, tempfile, base64
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from gtts import gTTS

# ---------------- ENV + SECRETS ----------------
load_dotenv()

if hasattr(st, "secrets"):
    for key, val in st.secrets.items():
        os.environ[key] = val

st.set_page_config(page_title="BrightSmile Support", page_icon="🦷")

st.title("🦷 BrightSmile Dental Support")
st.caption("Book appointments · Get info · 24/7 support")
st.divider()

# ---------------- TEXT TO SPEECH ----------------
def text_to_speech(text: str) -> str:
    try:
        clean = re.sub(r"[#*_`]", "", text)
        clean = clean[:500]

        tts = gTTS(text=clean, lang="en")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            tmp = f.name

        with open(tmp, "rb") as f:
            audio = base64.b64encode(f.read()).decode()

        os.unlink(tmp)
        return audio

    except:
        return ""

# ---------------- LOAD AGENT ----------------
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
    from typing import Annotated, Sequence, Optional
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel

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

    # ---- Calendar functions ----
    from calendar_mcp_server import handle_book, handle_cancel, handle_availability

    class BookInput(BaseModel):
        patient_name: str
        date: str
        time: str
        treatment: Optional[str] = "Dental Appointment"
        patient_email: Optional[str] = ""

    class CancelInput(BaseModel):
        event_id: str

    class AvailabilityInput(BaseModel):
        date: str

    def book_fn(**kwargs):
        """Book a dental appointment in Google Calendar."""
        return handle_book(kwargs)

    def cancel_fn(event_id: str):
        """Cancel an appointment using event ID."""
        return handle_cancel({"event_id": event_id})

    def availability_fn(date: str):
        """Check booked slots for a given date."""
        return handle_availability({"date": date})

    tools = [
        retriever_tool,

        StructuredTool.from_function(
            func=book_fn,
            name="book_appointment",
            description="Book a dental appointment at BrightSmile Dental Hospital",
            args_schema=BookInput
        ),

        StructuredTool.from_function(
            func=cancel_fn,
            name="cancel_appointment",
            description="Cancel an appointment using event ID",
            args_schema=CancelInput
        ),

        StructuredTool.from_function(
            func=availability_fn,
            name="check_availability",
            description="Check appointment slots for a given date",
            args_schema=AvailabilityInput
        ),
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

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- CHAT UI ----------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

# ---------------- MAIN ----------------
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = load_agent()

                st.session_state.messages.append(HumanMessage(content=user_input))
                result = graph.invoke({"messages": st.session_state.messages})

                response = result["messages"][-1].content
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

                st.session_state.messages = list(result["messages"])

            except Exception as e:
                response = f"⚠️ Error: {str(e)}"

        st.markdown(response)

    st.session_state.history.append({"role": "assistant", "content": response})

    # ---- Audio ----
    audio = text_to_speech(response)
    if audio:
        st.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

    st.rerun()
