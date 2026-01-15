from pathlib import Path

# Apple Calendar / iCloud Settings
# Server URL for iCloud is typically https://caldav.icloud.com/


CALDAV_URL = "https://caldav.icloud.com/"
USERNAME = "bljordan4@gmail.com"
PASSWORD = "dtoo-iqwv-stsm-qeuy"  # App-Specific Password

# Paths
BASE_DIR = Path('~/Documents/University').expanduser()
CURRENT_SYM = Path('~/current_course').expanduser()
WATCH_FILE = Path('/tmp/current_course')