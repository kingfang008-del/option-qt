import pandas as pd
import numpy as np
from pathlib import Path

def check_duplicates(symbol):
    root = Path("/Users/fangshuai/Documents/GitHub/option-qt/train_data/quote_features_val")
    stock_dir = root / symbol
    files = list(stock_dir.glob("**/regular/**/1min/*.parquet"))
    
    for f in files:
        df = pd.read_parquet(f)
        if 'timestamp' not in df.columns: df.reset_index(inplace=True)
        # Standardize timestamp
        ts = pd.to_datetime(df['timestamp'])
        if ts.dt.tz is None: ts = ts.dt.tz_localize('UTC')
        ts = ts.dt.tz_convert('America/New_York')
        df['align_ts'] = ts.astype(np.int64)
        
        counts = df['align_ts'].value_counts()
        dupes = counts[counts > 1]
        if not dupes.empty:
            print(f"File {f.name} has {len(dupes)} duplicate timestamps!")
            print(dupes.head())
            
            # Check if data in duplicates is different
            first_dupe_ts = dupes.index[0]
            dupe_rows = df[df['align_ts'] == first_dupe_ts]
            print(f"Data for duplicated TS {first_dupe_ts}:")
            print(dupe_rows.iloc[:, :5]) # Check first few cols
        else:
            print(f"File {f.name} has no duplicates.")

if __name__ == '__main__':
    check_duplicates('AMD')
