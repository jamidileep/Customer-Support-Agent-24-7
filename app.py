"""
app.py — BrightSmile Dental Support Agent
Run: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
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

# ── Pin the mic iframe fixed to bottom-right beside chat send button ──
st.markdown("""
<style>
iframe[title="mic_component"] {
    position: fixed !important;
    bottom: 10px !important;
    right: 68px !important;
    width: 46px !important;
    height: 46px !important;
    border: none !important;
    background: transparent !important;
    z-index: 9999 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🦷 BrightSmile Dental Support")
st.caption("Book appointments · Get info · 24/7 support")
st.divider()

# ---------------- SPEECH TO TEXT ----------------
def speech_to_text(audio_bytes: bytes) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("llm_api"))
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        with open(tmp_path, "rb") as f:
            transcription = client.audio.translations.create(model="whisper-large-v3", file=f)
        os.unlink(tmp_path)
        return transcription.text.strip()
    except Exception as e:
        return f"❌ Transcription error: {str(e)}"

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

    loader = PyPDFLoader("BrightSmile_Dental_Hospital.pdf")
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(
        splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(), "retriever_tool", "Search hospital info"
    )

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
        try:
            return handle_book(kwargs)
        except Exception as e:
            return f"❌ BOOKING ERROR: {str(e)}"

    def cancel_fn(event_id: str):
        return handle_cancel({"event_id": event_id})

    def availability_fn(date: str):
        return handle_availability({"date": date})

    tools = [
        retriever_tool,
        StructuredTool.from_function(func=book_fn, name="book_appointment",
            description="Book a dental appointment at BrightSmile Dental Hospital",
            args_schema=BookInput),
        StructuredTool.from_function(func=cancel_fn, name="cancel_appointment",
            description="Cancel an appointment using event ID", args_schema=CancelInput),
        StructuredTool.from_function(func=availability_fn, name="check_availability",
            description="Check appointment slots for a given date", args_schema=AvailabilityInput),
    ]

    llm = init_chat_model(model="groq:openai/gpt-oss-120b", api_key=os.getenv("llm_api"))

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
for key, default in [("history", []), ("messages", []), ("last_audio_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- CHAT UI ----------------
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- STANDARD PINNED CHAT INPUT ----------------
user_input = st.chat_input("Ask me anything...")

# ---------------- PURE JS MIC COMPONENT ----------------
# No third-party package. Single 🎙️ button. Click = start, click again = stop + send.
# JS encodes the recorded audio as base64 and passes it back as the component value.
mic_html = """
<style>
  body { margin:0; padding:0; background:transparent; overflow:hidden; }
  #btn {
    width:40px; height:40px; border-radius:50%;
    border: 1.5px solid #3f3f3f;
    cursor:pointer; font-size:18px;
    background:#2f2f2f; color:#ececec;
    display:flex; align-items:center; justify-content:center;
  }
  #btn.rec { background:#c62828; border-color:#c62828; }
</style>
<button id="btn" title="Click to record">🎙️</button>
<script>
let mr, chunks=[], going=false;
document.getElementById('btn').onclick = async () => {
  const btn = document.getElementById('btn');
  if (!going) {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    mr = new MediaRecorder(stream);
    chunks = [];
    mr.ondataavailable = e => chunks.push(e.data);
    mr.onstop = () => {
      const blob = new Blob(chunks, {type:'audio/webm'});
      const reader = new FileReader();
      reader.onloadend = () => {
        const b64 = reader.result.split(',')[1];
        window.parent.postMessage({isStreamlitMessage:true, type:'streamlit:setComponentValue', value:b64}, '*');
      };
      reader.readAsDataURL(blob);
      stream.getTracks().forEach(t=>t.stop());
    };
    mr.start();
    going=true; btn.textContent='⏹️'; btn.classList.add('rec');
  } else {
    mr.stop();
    going=false; btn.textContent='🎙️'; btn.classList.remove('rec');
  }
};
</script>
"""

mic_b64 = components.html(mic_html, height=46, key="mic_component")

# ---------------- HANDLE MIC RECORDING ----------------
if mic_b64 and isinstance(mic_b64, str):
    audio_id = hash(mic_b64)
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        audio_bytes = base64.b64decode(mic_b64)
        with st.spinner("Transcribing..."):
            transcribed = speech_to_text(audio_bytes)
        if transcribed and not transcribed.startswith("❌"):
            user_input = transcribed
        else:
            st.warning(transcribed)

# ---------------- MAIN FLOW ----------------
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

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

    # Auto-play TTS for every response (mic & typed)
    audio_b64 = text_to_speech(response)
    if audio_b64:
        st.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

    st.rerun()
