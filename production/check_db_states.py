import psycopg2
import json
import sys
sys.path.append('/Users/fangshuai/Documents/GitHub/option-qt/production/baseline')
from config import PG_DB_URL

def check_db_states():
    conn = psycopg2.connect(PG_DB_URL)
    c = conn.cursor()
    c.execute("SELECT symbol, data FROM symbol_state")
    rows = c.fetchall()
    conn.close()
    
    active = 0
    locked_cash = 0.0
    
    print("--- Current Saved DB States ---")
    for sym, data_str in rows:
        if sym == '_GLOBAL_STATE_':
            state = json.loads(data_str)
            print(f"Global Mock Cash: ${float(state.get('mock_cash', 0)):.2f}")
            continue
            
        st = json.loads(data_str)
        pos = st.get('position', 0)
        qty = st.get('qty', 0)
        pending = st.get('is_pending', False)
        
        if pos != 0 or pending or qty > 0:
            active += 1
            cost = st.get('entry_price', 0.0) * float(qty) * 100
            locked_cash += cost
            print(f"Symbol: {sym} | Pos: {pos} | Qty: {qty} | Pending: {pending} | Entry: {st.get('entry_price')} | Cost: ${cost:.2f} | is_pending stored?: {'is_pending' in st}")

    print(f"\nTotal Active Count (via state restore): {active}")
    print(f"Total Locked Cash: ${locked_cash:.2f}")

if __name__ == '__main__':
    check_db_states()
