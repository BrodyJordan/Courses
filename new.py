passwd = "dtoo-iqwv-stsm-qeuy"

import datetime
from pyicloud import PyiCloudService
# ... other imports from your original script

def authenticate_icloud():
    # Replace with your Apple ID
    api = PyiCloudService('bljordan4@gmail.com', passwd)
    
    if api.requires_2fa:
        print("Two-factor authentication required.")
        # Logic to handle 2FA (usually only needed once)
        code = input("Enter the code you received of one of your approved devices: ")
        result = api.validate_2fa_code(code)
        if not result:
            print("Failed to verify 2FA code")
            return None
    return api

def get_apple_events(api, start, end):
    # api.calendar.get_events returns events for a timeframe
    events = api.calendar.get_events(from_dt=start, to_dt=end)
    return [
        {
            'summary': e['title'],
            'location': e.get('location'),
            'start': e['startDate'][1:7], # Format depends on library version
            'end': e['endDate'][1:7]
        }
        for e in events
    ]

