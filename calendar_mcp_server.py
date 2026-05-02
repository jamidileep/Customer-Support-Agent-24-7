"""
calendar_mcp_server.py

MCP Server for BrightSmile Dental Hospital — Appointment Booking.
Place this file in your project root (same level as support_Agent.ipynb).

Run it as:   python calendar_mcp_server.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── Config ────────────────────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"   # already exists in your project ✅
TOKEN_FILE = "token.json"               # auto-created after first login
TIMEZONE = "Asia/Kolkata"
CALENDAR_ID = "primary"

# ── Google Calendar Auth ──────────────────────────────────────────────────────

def get_calendar_service():
    """Load credentials and return Google Calendar service."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Refresh or re-authenticate if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ── MCP Server ────────────────────────────────────────────────────────────────

app = Server("brightsmile-calendar-server")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Expose 3 tools to the AI agent."""
    return [
        types.Tool(
            name="book_appointment",
            description=(
                "Book a dental appointment at BrightSmile Dental Hospital. "
                "Use this when the user wants to schedule, book, or fix an appointment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Full name of the patient"
                    },
                    "date": {
                        "type": "string",
                        "description": "Appointment date in YYYY-MM-DD format"
                    },
                    "time": {
                        "type": "string",
                        "description": "Appointment time in HH:MM 24-hour format (e.g. 10:30)"
                    },
                    "treatment": {
                        "type": "string",
                        "description": "Type of dental treatment (e.g. cleaning, checkup, root canal)"
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Duration in minutes. Default is 30.",
                        "default": 30
                    }
                },
                "required": ["patient_name", "date", "time"]
            }
        ),
        types.Tool(
            name="cancel_appointment",
            description=(
                "Cancel an existing appointment using the Event ID. "
                "The Event ID is given to the patient when they booked."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The Google Calendar Event ID to cancel"
                    }
                },
                "required": ["event_id"]
            }
        ),
        types.Tool(
            name="check_availability",
            description=(
                "Check what appointment slots are already booked on a given date. "
                "Use this when user asks 'are you free on X date' or 'what slots are available'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to check in YYYY-MM-DD format"
                    }
                },
                "required": ["date"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Route tool calls."""
    if name == "book_appointment":
        result = handle_book(arguments)
    elif name == "cancel_appointment":
        result = handle_cancel(arguments)
    elif name == "check_availability":
        result = handle_availability(arguments)
    else:
        result = f"❌ Unknown tool: {name}"

    return [types.TextContent(type="text", text=result)]



#-------------Email----------------------------------------------------------
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_booking_email(patient_email, booking_details):
    """LLM writes the email, we just send it."""
    try:
        from langchain.chat_models import init_chat_model
        from dotenv import load_dotenv
        load_dotenv()

        llm = init_chat_model(
            model="groq:llama-3.3-70b-versatile",
            api_key=os.getenv("llm_api")
        )

        # LLM writes the email
        email_content = llm.invoke(f"""
Write a professional appointment confirmation email for BrightSmile Dental Hospital.

Booking details:
{booking_details}

Write a warm, professional email. Include all booking details clearly.
No subject line, just the email body. No <think> tags.
""").content

        # Strip <think> tags if any
        import re
        email_content = re.sub(r"<think>.*?</think>", "", email_content, flags=re.DOTALL).strip()

        # Send it
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

        print(f"✅ Email sent to {patient_email}", file=sys.stderr)
        return True

    except Exception as e:
        print(f"⚠️ Email failed: {e}", file=sys.stderr)
        return False    


# ── Tool Logic ────────────────────────────────────────────────────────────────

def handle_book(args: dict) -> str:
    try:
        service = get_calendar_service()

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
            "start": {
                "dateTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": TIMEZONE,
            },
            "end": {
                "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "timeZone": TIMEZONE,
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 60},
                    {"method": "popup", "minutes": 15},
                ],
            },
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

        # ── LLM writes and sends email automatically ──
        if patient_email:
            sent = send_booking_email(patient_email, result_text)
            result_text += f"\n📧 Confirmation email {'sent to ' + patient_email if sent else 'could not be sent.'}"

        return result_text

    except ValueError:
        return "❌ Wrong format. Date must be YYYY-MM-DD and time must be HH:MM (24hr)"
    except FileNotFoundError as e:
        return f"❌ Auth Error: {str(e)}"
    except Exception as e:
        return f"❌ Booking failed: {str(e)}"
    
    


def handle_cancel(args: dict) -> str:
    try:
        service = get_calendar_service()
        event_id = args["event_id"]
        service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
        return (
            f"✅ Appointment Cancelled.\n"
            f"Event ID: {event_id} has been removed from the calendar."
        )
    except Exception as e:
        return f"❌ Cancellation failed: {str(e)}"


def handle_availability(args: dict) -> str:
    try:
        service = get_calendar_service()
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


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main():
    print("🦷 BrightSmile Calendar MCP Server running...", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())