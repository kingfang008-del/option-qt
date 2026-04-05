from config import PG_DB_URL
# parse dbname=quant_trade user=postgres password=postgres host=...
parts = dict(p.split("=") for p in PG_DB_URL.split())
uri = f"postgresql://{parts['user']}:{parts['password']}@{parts['host']}:{parts['port']}/{parts['dbname']}"
print(uri)
