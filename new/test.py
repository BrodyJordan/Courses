#!/usr/bin/python3
from caldav.davclient import get_davclient
import json
from datetime import datetime, timedelta
import os

os.environ["CALDAV_URL"] = "https://p152-caldav.icloud.com:443/20372772933/calendars/725E3EC3-372D-42D9-9033-C9AE2C474AA7/"
os.environ["CALDAV_USERNAME"] = "bljordan4@gmail.com"
os.environ["CALDAV_PASSWORD"] = "dtoo-iqwv-stsm-qeuy"

def fill_event(component, calendar) -> dict[str, str]:
    ## quite some data is tossed away here - like, the recurring rule.
    cur = {}
    cur["calendar"] = f"{calendar}"
    cur["summary"] = component.get("summary")
    cur["description"] = component.get("description")
    ## month/day/year time? Never ever do that!
    ## It's one of the most confusing date formats ever!
    ## Use year-month-day time instead ... https://xkcd.com/1179/
    cur["start"] = component.start.strftime("%m/%d/%Y %H:%M")
    endDate = component.end
    if endDate:
        cur["end"] = endDate.strftime("%m/%d/%Y %H:%M")
    ## For me the following line breaks because some imported calendar events
    ## came without dtstamp.  But dtstamp is mandatory according to the RFC
    cur["datestamp"] = component.get("dtstamp").dt.strftime("%m/%d/%Y %H:%M")
    return cur

def print_calendars_demo(calendars):
    if not calendars:
        return
    events = []
    for calendar in calendars:
        for event in calendar.events():
            ## Most calendar events will have only one component,
            ## and it can be accessed simply as event.component
            ## The exception is special recurrences, to handle those
            ## we may need to do the walk:
            for component in event.icalendar_instance.walk():
                print(component)
                if component.name != "VEVENT":
                    continue
                events.append(fill_event(component, calendar))
    print(json.dumps(events, indent=2, ensure_ascii=False))

with get_davclient() as client:
    print("Connecting to the caldav server")
    my_principal = client.principal()

    ## The principals calendars can be fetched like this:
    calendars = my_principal.calendars()
    # print_calendars_demo(calendars)
    print_calendars_demo(client.principal().calendars())

# https://p152-caldav.icloud.com:443/20372772933/calendars/725E3EC3-372D-42D9-9033-C9AE2C474AA7/
# https://p152-caldav.icloud.com:443/20372772933/calendars/06D4B3C7-0ECC-4DC0-B768-541A40478873/
