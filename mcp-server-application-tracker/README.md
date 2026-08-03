# Job Application Tracker — MCP Server

A simple [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server for tracking job
applications: company, role, status, interview dates, and notes. Built with the
[MCP Python SDK v2](https://py.sdk.modelcontextprotocol.io/), designed to be used from
Claude Desktop, Claude Code, or any other MCP-compatible host.

Data is stored in a local JSON file (`applications_data.json`) next to the server script, so
your applications persist across restarts. This file is not tracked in git — it's your personal
data.

## Requirements

- Python 3.10+
- The `mcp` package, version 2.0.0 (see [`requirements.txt`](requirements.txt))

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/MoAhmadian/Machine_Learning.git
   cd job-application-tracker
   ```

2. **Create and activate a virtual environment**

   Windows (PowerShell):
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

   macOS / Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the server

The server communicates over **stdio** — it's designed to be launched by an MCP host
(Claude Desktop, Claude Code, etc.), not run standalone in a terminal for interactive use.

Running it directly will just sit there waiting for a client to speak first:

```bash
python server.py
```

That's expected — it means it's working and waiting for a host to connect.

## Testing without a full MCP host

Use the included test script to call tools directly, in-memory, without any transport or
separate process:

```bash
python test_client.py
```

This lists current applications, adds a test entry, then lists again so you can confirm
everything works end to end.

## Connecting to Claude Desktop

1. Open (or create) your Claude Desktop config file:

   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add an entry pointing at your venv's Python interpreter and the server script. Use
   **absolute paths**:

   ```json
   {
     "mcpServers": {
       "job-application-tracker": {
         "command": "C:\\path\\to\\job-application-tracker\\venv\\Scripts\\python.exe",
         "args": ["C:\\path\\to\\job-application-tracker\\server.py"]
       }
     }
   }
   ```

   macOS/Linux example:
   ```json
   {
     "mcpServers": {
       "job-application-tracker": {
         "command": "/path/to/job-application-tracker/venv/bin/python",
         "args": ["/path/to/job-application-tracker/server.py"]
       }
     }
   }
   ```

3. **Fully restart** Claude Desktop (quit from the system tray / menu bar, not just close the
   window).

4. Look for the tools icon in the chat box to confirm `job-application-tracker` shows up as
   connected.

5. Just ask Claude naturally — e.g. *"Add a new application for Meta, Software Engineer, applied
   today"* or *"Show me all my job applications"* — and it will call the right tool.

## Connecting to Claude Code

```bash
claude mcp add job-application-tracker -- \
  /path/to/job-application-tracker/venv/bin/python \
  /path/to/job-application-tracker/server.py
```

(Windows: use the `venv\Scripts\python.exe` path as in the Desktop config above.)

## Available tools

| Tool | Description |
|---|---|
| `add_application(company, role, date_applied, notes="")` | Add a new application. Returns its generated ID (e.g. `APP003`). |
| `update_status(application_id, new_status, note="")` | Update an application's status. |
| `schedule_interview(application_id, interview_datetime, notes="")` | Schedule/reschedule an interview; sets status to "Interview Scheduled". |
| `get_application(application_id)` | Get full details for one application. |
| `list_applications(company=None, status=None)` | List all applications, optionally filtered. |
| `get_upcoming_interviews()` | List applications with a scheduled interview, soonest first. |
| `get_status_history(application_id)` | Full status change history for an application. |

Valid statuses: `Applied`, `Online Assessment`, `Phone Screen`, `Interview Scheduled`,
`Interviewed`, `Offer`, `Rejected`, `Withdrawn`.

Date format: `YYYY-MM-DD`. Datetime format (for interviews): `YYYY-MM-DD HH:MM` (24-hour).

## Data persistence

On first run, the server seeds itself with two example applications. From then on, every add,
status update, or interview scheduling call writes the full application set to
`applications_data.json` in the same folder as `server.py`. Deleting that file resets you back to
the seed data.

## Notes on scope

This server is intentionally local and single-user:

- Data lives in a plain JSON file, not a database — fine for personal use, not built for
  concurrent multi-user access.
- It runs over stdio, meaning only the machine it's installed on (and whatever MCP host is
  configured there) can use it. To make it reachable from other devices (e.g. a phone), you'd
  need to switch the transport to Streamable HTTP and host it somewhere network-reachable —
  that's a separate, larger step outside the scope of this repo.

## License

Add a license of your choice (MIT is a common default for small personal tools).
