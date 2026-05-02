"""
calendar_mcp_server.py — BrightSmile Dental Hospital
"""

import asyncio, os, sys, json, re
from datetime import datetime, timedelta

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
TIMEZONE = "Asia/Kolkata"
CALENDAR_ID = "primary"

# ── Auth — reads GOOGLE_TOKEN env var (Streamlit Cloud) or falls back to token.json ──
def get_calendar_service():
    creds = None

    token_env = os.getenv("GOOGLE_TOKEN")
    if token_env:
        # Running on Streamlit Cloud — load from secret
        try:
            creds = Credentials.from_authorized_user_info(json.loads(token_env), SCOPES)
        except Exception as e:
            return None, f"❌ Failed to load GOOGLE_TOKEN: {e}"
    elif os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds:
        return None, "❌ No credentials found. Set GOOGLE_TOKEN secret or provide token.json"

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Save refreshed token back if running locally
            if not os.getenv("GOOGLE_TOKEN") and os.path.exists(TOKEN_FILE):
                with open(TOKEN_FILE, "w") as f:
                    f.write(creds.to_json())
        except Exception as e:
            return None, f"❌ Token refresh failed: {e}"

    return build("calendar", "v3", credentials=creds), None


# ── MCP Server ────────────────────────────────────────────────
app = Server("brightsmile-calendar-server")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="book_appointment",
            description="Book a dental appointment at BrightSmile Dental Hospital.",
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_name": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "time": {"type": "string", "description": "HH:MM 24hr"},
                    "treatment": {"type": "string", "default": "Dental Appointment"},
                    "duration_minutes": {"type": "integer", "default": 30},
                    "patient_email": {"type": "string", "default": ""}
                },
                "required": ["patient_name", "date", "time"]
            }
        ),
        types.Tool(
            name="cancel_appointment",
            description="Cancel an appointment by Event ID.",
            inputSchema={
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"]
            }
        ),
        types.Tool(
            name="check_availability",
            description="Check booked slots on a given date.",
            inputSchema={
                "type": "object",
                "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                "required": ["date"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "book_appointment":
        result = handle_book(arguments)
    elif name == "cancel_appointment":
        result = handle_cancel(arguments)
    elif name == "check_availability":
        result = handle_availability(arguments)
    else:
        result = f"❌ Unknown tool: {name}"
    return [types.TextContent(type="text", text=result)]


# ── Email ─────────────────────────────────────────────────────
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_booking_email(patient_email, booking_details):
    try:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(model="groq:llama-3.3-70b-versatile", api_key=os.getenv("llm_api"))
        email_content = llm.invoke(f"""
Write a professional appointment confirmation email for BrightSmile Dental Hospital.
Booking details: {booking_details}
Write a warm, professional email body only. No subject line. No <think> tags.
""").content
        email_content = re.sub(r"<think>.*?</think>", "", email_content, flags=re.DOTALL).strip()

        sender = os.getenv("EMAIL_SENDER")
        password = os.getenv("EMAIL_PASSWORD")
        msg = MIMEMultipart()
        msg["Subject"] = "✅ Appointment Confirmed — BrightSmile Dental Hospital"
        msg["From"] = f"BrightSmile Dental <{sender}>"
        msg["To"] = patient_email
        msg.attach(MIMEText(email_content, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, patient_email, msg.as_string())
        return True
    except Exception as e:
        print(f"⚠️ Email failed: {e}", file=sys.stderr)
        return False


# ── Tool Logic ────────────────────────────────────────────────
def handle_book(args: dict) -> str:
    service, err = get_calendar_service()
    if err:
        return err
    try:
        name = args["patient_name"]
        date = args["date"]
        time = args["time"]
        treatment = args.get("treatment", "Dental Appointment")
        duration = args.get("duration_minutes", 30)
        patient_email = args.get("patient_email", "")

        start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=duration)

        event = {
            "summary": f"🦷 {treatment} — {name}",
            "description": f"Patient: {name}\nTreatment: {treatment}\nBooked via BrightSmile Support Agent",
            "start": {"dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": TIMEZONE},
            "end": {"dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": TIMEZONE},
            "reminders": {"useDefault": False, "overrides": [
                {"method": "email", "minutes": 60},
                {"method": "popup", "minutes": 15},
            ]},
        }

        created = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
        event_link = created.get("htmlLink", "N/A")
        event_id = created.get("id", "N/A")

        result_text = (
            f"✅ Appointment Booked Successfully!\n"
            f"👤 Patient   : {name}\n"
            f"🦷 Treatment : {treatment}\n"
            f"📅 Date      : {date}\n"
            f"⏰ Time      : {time} IST\n"
            f"⏱️ Duration  : {duration} minutes\n"
            f"🔗 Calendar  : {event_link}\n"
            f"📌 Event ID  : {event_id}\n"
            f"💡 Save the Event ID to cancel later."
        )

        if patient_email:
            sent = send_booking_email(patient_email, result_text)
            result_text += f"\n📧 Confirmation email {'sent to ' + patient_email if sent else 'could not be sent.'}"

        return result_text

    except ValueError:
        return "❌ Wrong format. Date must be YYYY-MM-DD and time must be HH:MM (24hr)"
    except Exception as e:
        return f"❌ Booking failed: {str(e)}"


def handle_cancel(args: dict) -> str:
    service, err = get_calendar_service()
    if err:
        return err
    try:
        event_id = args["event_id"]
        service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
        return f"✅ Appointment Cancelled.\nEvent ID: {event_id} has been removed."
    except Exception as e:
        return f"❌ Cancellation failed: {str(e)}"


def handle_availability(args: dict) -> str:
    service, err = get_calendar_service()
    if err:
        return err
    try:
        date = args["date"]
        result = service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=f"{date}T00:00:00+05:30",
            timeMax=f"{date}T23:59:59+05:30",
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = result.get("items", [])
        if not events:
            return f"📅 {date} is fully open — no appointments booked yet."

        lines = [f"📅 Booked slots on {date}:"]
        for ev in events:
            start = ev["start"].get("dateTime", ev["start"].get("date"))
            t = datetime.fromisoformat(start).strftime("%H:%M")
            lines.append(f"  🔴 {t} — {ev.get('summary', 'Appointment')}  (ID: {ev.get('id','')})")
        lines.append("\n✅ All other time slots are available.")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Failed to check availability: {str(e)}"


# ── Entry Point ───────────────────────────────────────────────
async def main():
    print("🦷 BrightSmile Calendar MCP Server running...", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
