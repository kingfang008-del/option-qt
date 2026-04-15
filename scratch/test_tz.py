import pandas as pd
import datetime
import pytz

try:
    timestamp = 1713010000 # Example ts
    ts_ny = pd.Timestamp(timestamp, unit='s', tz='UTC').tz_convert('America/New_York')
    print(f"TS NY: {ts_ny}")
    search_date = ts_ny.replace(hour=0, minute=0, second=0, microsecond=0).tz_localize(None)
    print(f"Search date: {search_date}")
except Exception as e:
    print(f"ERROR: {e}")
