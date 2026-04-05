import pyarrow.parquet as pq

# Let's inspect ONE parquet file to see what data is inside it.
parquet_file = "/Users/fangshuai/quant_project/data/rl_feed_parquet_batch/AAPL.parquet"
try:
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    print("Columns:", df.columns.tolist())
    print("\nDescribe spy_roc_5min:")
    print(df['spy_roc_5min'].describe())
    
except Exception as e:
    print("Error:", e)
