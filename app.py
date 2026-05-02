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

st.set_page_config(page_title="BrightSmile Support", page_icon="🦷", layout="centered")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #212121; color: #ececec; }
.block-container { padding: 1rem 1rem 0 1rem !important; max-width: 760px !important; }
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 8px 0 !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown p {
    background: #2f2f2f; border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; display: inline-block; max-width: 80%; float: right;
}
[data-testid="stBottom"] {
    background: #212121 !important;
    border-top: 1px solid #2f2f2f !important;
    padding: 8px 0 !important;
}
[data-testid="stChatInput"] textarea {
    background: #2f2f2f !important; border: 1px solid #3f3f3f !important;
    border-radius: 12px !important; color: #ececec !important; font-size: 15px !important;
    padding-left: 52px !important;   /* room for mic icon on left  */
    padding-right: 52px !important;  /* room for speaker icon on right */
}
[data-testid="stChatInput"] textarea:focus { border-color: #555 !important; box-shadow: none !important; }
.stSpinner > div { border-top-color: #19c37d !important; }
hr { border-color: #2f2f2f !important; margin: 4px 0 !important; }
h1 { color: #ececec !important; font-size: 20px !important; margin-bottom: 0 !important; }
.stCaption { color: #8e8ea0 !important; }

/* ── Hide the big audio-input recorder UI, show only the mic icon ── */
[data-testid="stAudioInput"] > div > div:last-child { display: none !important; }
[data-testid="stAudioInput"] > div {
    border: none !important; background: transparent !important;
    padding: 0 !important; margin: 0 !important;
}
[data-testid="stAudioInput"] > div > div:first-child button {
    background: #2f2f2f !important; border: 1px solid #3f3f3f !important;
    border-radius: 50% !important; width: 38px !important; height: 38px !important;
    color: #ececec !important; font-size: 16px !important;
}
[data-testid="stAudioInput"] > div > div:first-child button:hover {
    background: #3f3f3f !important;
}
[data-testid="stAudioInput"] label { display: none !important; }

/* ── Speaker button styling ── */
.stButton > button {
    background: #2f2f2f !important; border: 1px solid #3f3f3f !important;
    color: #ececec !important; border-radius: 50% !important;
    width: 38px !important; height: 38px !important;
    padding: 0 !important; font-size: 16px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
}
.stButton > button:hover { background: #3f3f3f !important; border-color: #555 !important; }

/* ── Keep the bottom columns tight and vertically centered ── */
[data-testid="stBottom"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 6px !important;
}
[data-testid="stBottom"] [data-testid="stColumn"] {
    padding: 0 !important;
    min-width: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SPEECH TO TEXT ----------------
def speech_to_text(audio_bytes) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("llm_api"))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
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
        clean = re.sub(r"http\S+", "link", clean)
        clean = re.sub(r"📌.*", "", clean)
        clean = clean[:500]
        tts = gTTS(text=clean, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            tmp_path = f.name
        with open(tmp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp_path)
        return audio_b64
    except Exception as e:
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
    from langgraph.graph import END, StateGraph, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from typing import Annotated, Sequence, Optional
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field

    loader = PyPDFLoader("BrightSmile_Dental_Hospital.pdf")
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(
        splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(), "retriever_tool",
        "Search BrightSmile hospital info — services, timings, fees, doctors"
    )

    from calendar_mcp_server import handle_book, handle_cancel, handle_availability

    class BookInput(BaseModel):
        patient_name: str = Field(description="Full name of the patient")
        date: str = Field(description="Appointment date in YYYY-MM-DD format")
        time: str = Field(description="Appointment time in HH:MM 24-hour format")
        treatment: str = Field(default="Dental Appointment", description="Type of dental treatment")
        patient_email: Optional[str] = Field(default="", description="Patient email address")

    class CancelInput(BaseModel):
        event_id: str = Field(description="Google Calendar Event ID to cancel")

    class AvailabilityInput(BaseModel):
        date: str = Field(description="Date to check in YYYY-MM-DD format")

    def book_fn(patient_name: str, date: str, time: str,
                treatment: str = "Dental Appointment", patient_email: str = "") -> str:
        try:
            return handle_book({
                "patient_name": patient_name, "date": date, "time": time,
                "treatment": treatment, "patient_email": patient_email,
            })
        except Exception as e:
            return f"❌ Error: {e}"

    def cancel_fn(event_id: str) -> str:
        try:
            return handle_cancel({"event_id": event_id.strip()})
        except Exception as e:
            return f"❌ Error: {e}"

    def availability_fn(date: str) -> str:
        try:
            return handle_availability({"date": date.strip()})
        except Exception as e:
            return f"❌ Error: {e}"

    tools = [
        retriever_tool,
        StructuredTool.from_function(func=book_fn, name="book_appointment",
            description="Book a dental appointment at BrightSmile Dental Hospital.",
            args_schema=BookInput),
        StructuredTool.from_function(func=cancel_fn, name="cancel_appointment",
            description="Cancel an existing appointment using its Google Calendar Event ID.",
            args_schema=CancelInput),
        StructuredTool.from_function(func=availability_fn, name="check_availability",
            description="Check which appointment slots are already booked on a given date.",
            args_schema=AvailabilityInput),
    ]

    llm = init_chat_model(model="groq:openai/gpt-oss-120b", api_key=os.getenv("llm_api"))

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def assistant(state):
        system = SystemMessage(content="""You are a receptionist for BrightSmile Dental Hospital.

Tools available:
- retriever_tool: look up hospital info (services, timings, fees, doctors)
- book_appointment: fields — patient_name, date (YYYY-MM-DD), time (HH:MM), treatment, patient_email
- cancel_appointment: field — event_id
- check_availability: field — date (YYYY-MM-DD)

Be warm, professional, and always give a clear final answer.""")
        response = llm.bind_tools(tools).invoke([system] + list(state["messages"]))
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", assistant)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges("assistant", tools_condition)
    workflow.add_edge("tools", "assistant")
    return workflow.compile()


# ---------------- SESSION ----------------
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0
if "speaker_key" not in st.session_state:
    st.session_state.speaker_key = 0

# ---------------- PAGE ----------------
st.title("🦷 BrightSmile Dental Support")
st.caption("Book appointments · Get info · 24/7 support")
st.divider()

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🦷"):
        st.markdown(msg["content"])

# ---------------- BOTTOM BAR: mic | chat input | speaker ----------------
# Streamlit automatically moves everything after this point into stBottom.
# Columns here render INSIDE the fixed bottom bar — they don't push content up.
col_mic, col_input, col_speaker = st.columns([1, 10, 1])

with col_mic:
    audio = st.audio_input(" ", label_visibility="collapsed",
                           key=f"mic_{st.session_state.mic_key}")
with col_input:
    user_input = st.chat_input("Ask me anything...")

with col_speaker:
    speak_clicked = st.button("🔊", key=f"speaker_{st.session_state.speaker_key}",
                              help="Read last reply aloud")

# ---------------- SPEAKER ----------------
if speak_clicked:
    st.session_state.speaker_key += 1
    last_agent = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
    if last_agent:
        audio_b64 = text_to_speech(last_agent[-1]["content"])
        if audio_b64:
            st.markdown(
                f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )

# ---------------- MIC ----------------
if audio:
    audio_id = hash(audio.read())
    audio.seek(0)
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        with st.spinner("Transcribing..."):
            transcribed = speech_to_text(audio.read())
        st.session_state.mic_key += 1
        if transcribed and not transcribed.startswith("❌"):
            user_input = transcribed
        else:
            st.warning(transcribed)

# ---------------- MAIN FLOW ----------------
if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🦷"):
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

    # Auto-play TTS after mic input
    if audio:
        audio_b64 = text_to_speech(response)
        if audio_b64:
            st.markdown(
                f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True
            )

    st.rerun()
