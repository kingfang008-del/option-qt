import psycopg2
import sys
import os
sys.path.append('/Users/fangshuai/Documents/GitHub/option-qt/production/baseline')
from config import PG_DB_URL

def list_tables():
    conn = psycopg2.connect(PG_DB_URL)
    c = conn.cursor()
    c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = c.fetchall()
    for t in sorted([row[0] for row in tables]):
        print(t)
    conn.close()

if __name__ == '__main__':
    list_tables()
