from icalendar import Calendar
import caldav
import os 
from datetime import datetime

os.environ["CALDAV_URL"] = "https://caldav.icloud.com"
os.environ["CALDAV_USERNAME"] = "bljordan4@gmail.com"
os.environ["CALDAV_PASSWORD"] = "dtoo-iqwv-stsm-qeuy"

url = "https://caldav.icloud.com"
username = "bljordan4@gmail.com"
app_passwd = "dtoo-iqwv-stsm-qeuy"
#!/usr/bin/env python3
import json
import sys
import time
import math
import re
from datetime import datetime, time as dtime, timedelta
from dateutil import tz

LOCAL_TZ = tz.tzlocal()

import caldav

# Your project helper that returns an authenticated DAVClient
import caldav.davclient  # expects caldav.davclient.get_davclient()


# -------- Formatting helpers (Waybar) --------


def to_aware_dt(v):
    """
    Convert DTSTART/DTEND dt value to a timezone-aware datetime.
    Handles date or datetime, naive or aware.
    """
    if v is None:
        return None

    # date -> datetime at midnight
    if not isinstance(v, datetime):
        v = datetime.combine(v, dtime.min)

    # naive -> assume local timezone
    if v.tzinfo is None:
        v = v.replace(tzinfo=LOCAL_TZ)

    return v

def join(*args):
    return " ".join(str(e) for e in args if e)

def truncate(s: str, length: int) -> str:
    ellipsis = " ..."
    if s is None:
        return ""
    if len(s) <= length:
        return s
    return s[: length - len(ellipsis)] + ellipsis

def summary(text: str) -> str:
    if not text:
        return ""
    # Keep your old "remove X..." behavior
    return truncate(re.sub(r"X[0-9A-Za-z]+", "", text).strip(), 50)

def formatdd(begin: datetime, end: datetime) -> str:
    minutes = math.ceil((end - begin).total_seconds() / 60)

    if minutes <= 0:
        return "0 min"
    if minutes == 1:
        return "1 minute"
    if minutes < 60:
        return f"{minutes} min"

    hours = minutes // 60
    rest = minutes % 60

    if hours > 5 or rest == 0:
        return f"{hours} hr"

    return f"{hours}:{rest:02d} hr"


def location(loc: str) -> str:
    """
    Keep your original behavior:
    - if location contains "(...)", show the content inside parentheses
    - otherwise show nothing
    """
    if not loc:
        return ""
    m = re.search(r"\((.*)\)", loc)
    if not m:
        return ""
    return f"in {m.group(1)}"

def to_dt(v):
    """
    CalDAV DTSTART/DTEND may be date or datetime.
    Convert date -> datetime at midnight for comparisons.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    # date -> datetime
    return datetime.combine(v, dtime.min)

def build_status_text(events, now: datetime):
    current = next((e for e in events if e["start"] < now < e["end"]), None)

    if not current:
        nxt = next((e for e in events if now <= e["start"]), None)
        if nxt:
            return (
                join(
                    summary(nxt["summary"]),
                    "in",
                    formatdd(now, nxt["start"]),
                    location(nxt.get("location")),
                ),
                "idle"
            )
        return ("", "dim")

    nxt = next((e for e in events if e["start"] >= current["end"]), None)

    if not nxt:
        return (
            join("Ends in", formatdd(now, current["end"]) + "!"),
            "busy"
        )

    if current["end"] == nxt["start"]:
        return (
            join(
                "Ends in",
                formatdd(now, current["end"]) + ".",
                "Next:",
                summary(nxt["summary"]),
                location(nxt.get("location")),
            ),
            "busy"
        )

    return (
        join(
            "Ends in",
            formatdd(now, current["end"]) + ".",
            "Next:",
            summary(nxt["summary"]),
            location(nxt.get("location")),
            "after a break of",
            formatdd(current["end"], nxt["start"]),
        ),
        "busy"
    )



# -------- CalDAV fetching --------

def find_school_calendar(client):
    principal = client.principal()
    calendars = principal.calendars()
    school_cals = [c for c in calendars if (c.name or "").startswith("School")]

    if not school_cals:
        print("No calendars found starting with 'School'", file=sys.stderr)
        return None

    # Use the first match (safe). Change to [1] if you truly want second.
    return school_cals[1]

def get_todays_events(cal, now: datetime):
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)

    raw = cal.search(
        start=start_of_day,
        end=end_of_day,
        expand=True
    )

    events = []
    for ev in raw:
        comp = ev.component
        dtstart = to_aware_dt(getattr(comp.get("dtstart"), "dt", None))
        dtend = to_aware_dt(getattr(comp.get("dtend"), "dt", None))
        if not dtstart:
            continue
        # Some events may omit DTEND; assume 1 hour if missing (better than crashing)
        if not dtend:
            dtend = dtstart + timedelta(hours=1)

        events.append({
            "summary": str(comp.get("summary") or ""),
            "location": str(comp.get("location") or ""),
            "start": dtstart,
            "end": dtend,
        })

    events.sort(key=lambda e: e["start"])
    return events


# -------- Main loop (Waybar tail) --------

def emit(text: str, css_class: str, tooltip: str = ""):
    # Waybar: one JSON object per line, nothing else on stdout
    obj = {
        "text": text,
        "class": css_class,
        "tooltip": tooltip if tooltip else text
    }
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def main():
    REFRESH_EVENTS_EVERY = 5 * 60   # re-fetch from server every 5 minutes
    UPDATE_EVERY = 60              # update display every 60 seconds

    client = caldav.davclient.get_davclient()
    cal = find_school_calendar(client)
    if not cal:
        # still keep running so module doesn't die; just show nothing
        while True:
            emit("", "dim", "No School calendar found")
            time.sleep(UPDATE_EVERY)

    last_fetch = 0
    events = []

    while True:
        now = datetime.now(tz=LOCAL_TZ)

        # periodic refresh
        if time.time() - last_fetch >= REFRESH_EVENTS_EVERY:
            try:
                events = get_todays_events(cal, now)
                last_fetch = time.time()
                print(f"Fetched {len(events)} events", file=sys.stderr)
            except Exception as e:
                # keep running even if server hiccups
                print(f"Fetch error: {e}", file=sys.stderr)

        text, css_class = build_status_text(events, now)
        emit(text, css_class)
        time.sleep(UPDATE_EVERY)


if __name__ == "__main__":
    main()
