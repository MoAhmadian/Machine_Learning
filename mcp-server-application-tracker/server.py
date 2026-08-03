from mcp.server import MCPServer
from typing import List, Optional, Dict
from datetime import datetime
import json
import os

# Persistent storage location: a JSON file next to this script.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "applications_data.json")

# Seed data used only the very first time the app runs (no data file yet).
_SEED_APPLICATIONS: Dict[str, dict] = {
    "APP001": {
        "company": "Google",
        "role": "Software Engineer",
        "status": "Applied",
        "date_applied": "2026-07-10",
        "interview_datetime": None,
        "notes": "Referred by a friend on the search team.",
        "history": [
            {"status": "Applied", "timestamp": "2026-07-10 09:00"}
        ],
    },
    "APP002": {
        "company": "Anthropic",
        "role": "Applied Scientist",
        "status": "Interview Scheduled",
        "date_applied": "2026-07-15",
        "interview_datetime": "2026-08-10 14:00",
        "notes": "Onsite loop, 4 rounds.",
        "history": [
            {"status": "Applied", "timestamp": "2026-07-15 10:30"},
            {"status": "Interview Scheduled", "timestamp": "2026-07-28 16:45"},
        ],
    },
}


def _load_applications() -> Dict[str, dict]:
    """Load applications from the JSON data file, or fall back to seed data
    if the file doesn't exist yet (first run)."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(_SEED_APPLICATIONS)


def _save_applications() -> None:
    """Persist the current in-memory applications dict to disk."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(applications, f, indent=2)


# In-memory database of job applications, loaded from disk on startup.
# Each entry is keyed by an application_id like "APP001".
applications: Dict[str, dict] = _load_applications()

# Counter used to generate the next application ID (APP001, APP002, ...)
# Derived from existing IDs so it keeps incrementing correctly across restarts.
def _load_next_id_counter() -> int:
    if not applications:
        return 1
    numeric_parts = []
    for app_id in applications:
        digits = "".join(ch for ch in app_id if ch.isdigit())
        if digits:
            numeric_parts.append(int(digits))
    return (max(numeric_parts) + 1) if numeric_parts else (len(applications) + 1)


_next_id_counter = _load_next_id_counter()

# Statuses considered valid for an application, in a roughly typical order
VALID_STATUSES = [
    "Applied",
    "Online Assessment",
    "Phone Screen",
    "Interview Scheduled",
    "Interviewed",
    "Offer",
    "Rejected",
    "Withdrawn",
]

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

mcp = MCPServer("JobApplicationTracker")


def _now() -> str:
    return datetime.now().strftime(DATETIME_FORMAT)


def _generate_id() -> str:
    global _next_id_counter
    new_id = f"APP{_next_id_counter:03d}"
    _next_id_counter += 1
    return new_id


@mcp.tool()
def add_application(company: str, role: str, date_applied: str, notes: str = "") -> str:
    """
    Add a new job application to track.
    date_applied must be in YYYY-MM-DD format (e.g., "2026-08-01").
    Returns the generated application ID, which you'll use for all future
    lookups and updates on this application.
    """
    try:
        datetime.strptime(date_applied, DATE_FORMAT)
    except ValueError:
        return f"Invalid date format: '{date_applied}'. Please use YYYY-MM-DD."

    app_id = _generate_id()
    applications[app_id] = {
        "company": company,
        "role": role,
        "status": "Applied",
        "date_applied": date_applied,
        "interview_datetime": None,
        "notes": notes,
        "history": [{"status": "Applied", "timestamp": _now()}],
    }
    _save_applications()
    return f"Application {app_id} added: {role} at {company}, applied {date_applied}."


@mcp.tool()
def update_status(application_id: str, new_status: str, note: str = "") -> str:
    """
    Update the status of an application (e.g., "Phone Screen", "Offer", "Rejected").
    Valid statuses: Applied, Online Assessment, Phone Screen, Interview Scheduled,
    Interviewed, Offer, Rejected, Withdrawn.
    """
    if application_id not in applications:
        return f"Application ID '{application_id}' not found."
    if new_status not in VALID_STATUSES:
        return f"Invalid status '{new_status}'. Valid options: {', '.join(VALID_STATUSES)}."

    app = applications[application_id]
    app["status"] = new_status
    app["history"].append({"status": new_status, "timestamp": _now()})
    if note:
        app["notes"] = (app["notes"] + " | " if app["notes"] else "") + note

    _save_applications()
    return f"{application_id} ({app['company']} - {app['role']}) updated to '{new_status}'."


@mcp.tool()
def schedule_interview(application_id: str, interview_datetime: str, notes: str = "") -> str:
    """
    Schedule or reschedule an interview for an application.
    interview_datetime must be in YYYY-MM-DD HH:MM 24-hour format
    (e.g., "2026-08-15 14:00"). Automatically sets status to "Interview Scheduled".
    """
    if application_id not in applications:
        return f"Application ID '{application_id}' not found."
    try:
        datetime.strptime(interview_datetime, DATETIME_FORMAT)
    except ValueError:
        return f"Invalid datetime format: '{interview_datetime}'. Please use YYYY-MM-DD HH:MM."

    app = applications[application_id]
    app["interview_datetime"] = interview_datetime
    app["status"] = "Interview Scheduled"
    app["history"].append({"status": "Interview Scheduled", "timestamp": _now()})
    if notes:
        app["notes"] = (app["notes"] + " | " if app["notes"] else "") + notes

    _save_applications()
    return f"Interview for {application_id} ({app['company']}) scheduled at {interview_datetime}."


@mcp.tool()
def get_application(application_id: str) -> str:
    """Get full details for a single application by its ID."""
    app = applications.get(application_id)
    if not app:
        return f"Application ID '{application_id}' not found."

    interview = app["interview_datetime"] or "Not scheduled"
    return (
        f"{application_id}: {app['role']} at {app['company']}\n"
        f"Status: {app['status']}\n"
        f"Applied: {app['date_applied']}\n"
        f"Interview: {interview}\n"
        f"Notes: {app['notes'] or 'None'}"
    )


@mcp.tool()
def list_applications(company: Optional[str] = None, status: Optional[str] = None) -> str:
    """
    List all applications, optionally filtered by company (e.g., "Meta")
    and/or status (e.g., "Interview Scheduled").
    """
    results = []
    for app_id, app in applications.items():
        if company and company.lower() not in app["company"].lower():
            continue
        if status and status.lower() != app["status"].lower():
            continue
        results.append(f"{app_id}: {app['role']} at {app['company']} — {app['status']}")

    if not results:
        return "No applications match those filters."
    return "\n".join(results)


@mcp.tool()
def get_upcoming_interviews() -> str:
    """List all applications with a scheduled interview, soonest first."""
    upcoming = [
        (app_id, app)
        for app_id, app in applications.items()
        if app["interview_datetime"]
    ]
    if not upcoming:
        return "No interviews currently scheduled."

    upcoming.sort(key=lambda item: item[1]["interview_datetime"])
    lines = [
        f"{app_id}: {app['role']} at {app['company']} — {app['interview_datetime']}"
        for app_id, app in upcoming
    ]
    return "\n".join(lines)


@mcp.tool()
def get_status_history(application_id: str) -> str:
    """Get the full status change history for an application, in order."""
    app = applications.get(application_id)
    if not app:
        return f"Application ID '{application_id}' not found."

    lines = [f"{h['timestamp']} — {h['status']}" for h in app["history"]]
    return f"History for {application_id} ({app['company']}):\n" + "\n".join(lines)


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting for the job application tracker."""
    return f"Hi {name}! I can help you track your job applications — ask me about statuses, upcoming interviews, or add a new one."


if __name__ == "__main__":
    mcp.run()
