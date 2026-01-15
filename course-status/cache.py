#!/usr/bin/env python3

import json
import os
import sys
from datetime import datetime, time as dtime, timedelta
from dateutil.tz import tzlocal

import caldav
import caldav.davclient

os.environ["CALDAV_URL"] = "https://caldav.icloud.com"
os.environ["CALDAV_USERNAME"] = "bljordan4@gmail.com"
os.environ["CALDAV_PASSWORD"] = "dtoo-iqwv-stsm-qeuy"

LOCAL_TZ = tzlocal()
CACHE_DIR = os.path.expanduser("~/.cache/calendar")
CACHE_PATH = os.path.join(CACHE_DIR, "today.json")

def to_aware_dt(v):
    if v is None:
        return None
    if not isinstance(v, datetime):
        v = datetime.combine(v, dtime.min)
    if v.tzinfo is None:
        v = v.replace(tzinfo=LOCAL_TZ)
    return v

def find_school_calendar(client):
    principal = client.principal()
    calendars = principal.calendars()
    school_cals = [c for c in calendars if (c.name or "").startswith("School")]
    return school_cals[1] if school_cals else None

def get_todays_events(cal, now: datetime):
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=0)

    raw = cal.search(start=start_of_day, end=end_of_day, expand=True)

    events = []
    for ev in raw:
        comp = ev.component
        dtstart = to_aware_dt(getattr(comp.get("dtstart"), "dt", None))
        dtend = to_aware_dt(getattr(comp.get("dtend"), "dt", None))

        if not dtstart:
            continue
        if not dtend:
            dtend = dtstart + timedelta(hours=1)

        events.append({
            "summary": str(comp.get("summary") or ""),
            "location": str(comp.get("location") or ""),
            "start": dtstart.isoformat(),
            "end": dtend.isoformat(),
        })

    events.sort(key=lambda e: e["start"])
    return events


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    now = datetime.now(tz=LOCAL_TZ)

    client = caldav.davclient.get_davclient()
    cal = find_school_calendar(client)
    if not cal:
        print("No 'School*' calendar found.", file=sys.stderr)
        return 1

    events = get_todays_events(cal, now)

    payload = {
        "date": now.date().isoformat(),
        "fetched_at": now.isoformat(),
        "events": events,
    }

    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_PATH)

    print(f"Wrote {len(events)} events to {CACHE_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
