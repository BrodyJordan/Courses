#!/usr/bin/python3
from caldav.davclient import get_davclient
import json
from datetime import datetime, timedelta
from config import CALDAV_URL, USERNAME, PASSWORD
import os

os.environ["CALDAV_URL"] = "https://caldav.icloud.com"
os.environ["CALDAV_USERNAME"] = "bljordan4@gmail.com"
os.environ["CALDAV_PASSWORD"] = "dtoo-iqwv-stsm-qeuy"

def get_next_event():
    with get_davclient() as client:
    #with caldav.DAVClient(url=CALDAV_URL, username=USERNAME, password=PASSWORD) as client:
        # Fetch the primary calendar
        principal = client.principal()
        calendar = principal.calendars()[0] 
        
        # Search for events in the next 24 hours
        now = datetime.now()
        events = calendar.search(start=now, end=now + timedelta(days=1), event=True, expand=True)
        
        if not events:
            return {"text": "No Events", "tooltip": "No upcoming classes"}

        # Sort by start time using the internal icalendar component
        events.sort(key=lambda x: x.icalendar_component.get('dtstart').dt)
        next_ev = events[0].icalendar_component
        
        start = next_ev.get('dtstart').dt
        summary = str(next_ev.get('summary'))
        delta = (start - now).total_seconds() // 60
        
        return {
            "text": f"󰃭 {summary} in {int(delta)}m",
            "tooltip": f"Course: {summary}\nStart: {start.strftime('%H:%M')}\nLocation: {next_ev.get('location')}",
            "class": "upcoming" if delta < 15 else ""
        }

if __name__ == "__main__":
    print(json.dumps(get_next_event()))