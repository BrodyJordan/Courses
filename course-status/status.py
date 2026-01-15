#!/usr/bin/env python3
import json
import math
import os
import re
from datetime import datetime, time as dtime
from dateutil import tz

LOCAL_TZ = tz.tzlocal()
CACHE_PATH = os.path.expanduser("~/.cache/calendar/today.json")

SEP = " · "

def minutes_until(a: datetime, b: datetime) -> int:
    return max(0, math.ceil((b - a).total_seconds() / 60))

def fmt_duration(minutes: int) -> str:
    if minutes <= 0:
        return "0 min"

    h = minutes // 60
    m = minutes % 60

    if h == 0:
        return f"{m} min"
    if m == 0:
        return f"{h} hr"
    return f"{h} hr {m:02d} min"

def parse_course(summary: str) -> str:
    """
    'CRSE-XXX - COURSE NAME' → 'CRSE-XXX'
    """
    s = (summary or "").strip()
    m = re.match(r"^\s*([A-Za-z]{2,}-\w+)\s*-\s*", s)
    if m:
        return m.group(1).upper()
    # fallback
    return s.split("-", 1)[0].strip().upper()[:12] if s else "CLASS"

def fmt_location(loc: str) -> str:
    return re.sub(r"\s+", " ", (loc or "").strip())

def format_upcoming(event: dict, now: datetime) -> str:
    code = parse_course(event.get("summary", ""))
    t = fmt_duration(minutes_until(now, event["start"]))
    loc = fmt_location(event.get("location", ""))

    # Example: "MECH-321 · 12 min · ENGR 101"
    parts = [code, f"in {t}"]
    if loc:
        parts.append(loc)
    return SEP.join(parts)

def format_current(event: dict, now: datetime) -> str:
    code = parse_course(event.get("summary", ""))
    t = fmt_duration(minutes_until(now, event["end"]))

    # Example: "MECH-321 · 38 min"
    return SEP.join([code, f"{t} left"])

def build_status_text(events: list[dict], now: datetime) -> tuple[str, str]:
    current = next((e for e in events if e["start"] <= now < e["end"]), None)
    if current:
        return format_current(current, now), "busy"

    nxt = next((e for e in events if now < e["start"]), None)
    if nxt:
        return format_upcoming(nxt, now), "idle"

    return "No more classes", "dim"




def to_aware_dt(s: str) -> datetime:
    # cache stores ISO strings
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt

def load_events():
    if not os.path.exists(CACHE_PATH):
        return None

    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    # if cache is for a different day, treat as missing
    today = datetime.now(tz=LOCAL_TZ).date().isoformat()
    if payload.get("date") != today:
        return None

    events = []
    for e in payload.get("events", []):
        events.append({
            "summary": e.get("summary", ""),
            "location": e.get("location", ""),
            "start": to_aware_dt(e["start"]),
            "end": to_aware_dt(e["end"]),
        })
    events.sort(key=lambda x: x["start"])
    return events

def emit(text: str, cls: str, tooltip: str):
    print(json.dumps({"text": text, "class": cls, "tooltip": tooltip}, ensure_ascii=False))

def main():
    now = datetime.now(tz=LOCAL_TZ)
    events = load_events()

    if not events:
        emit("Calendar not refreshed", "dim", "Click to refresh calendar cache")
        return

    text, cls = build_status_text(events, now)
    emit(text, cls, text or "No more events today")

if __name__ == "__main__":
    main()