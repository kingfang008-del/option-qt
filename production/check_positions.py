import pickle
import time
from pathlib import Path

state_file = Path('/Users/fangshuai/Documents/GitHub/option-qt/production/orchestrator_state_v8.pkl')
if state_file.exists():
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
        
    print(f"State saved at: {time.ctime(state.get('ts', 0))}")
    states = state.get('states', {})
    
    active = 0
    locked_cash = 0.0
    for sym, dict_state in states.items():
        pos = dict_state.get('position', 0)
        qty = dict_state.get('qty', 0)
        pending = dict_state.get('is_pending', False)
        if pos != 0 or pending or qty > 0:
            active += 1
            cost = dict_state.get('entry_price', 0.0) * float(qty) * 100
            locked_cash += cost
            print(f"Symbol: {sym} | Pos: {pos} | Qty: {qty} | Pending: {pending} | Entry: {dict_state.get('entry_price')} | Cost: ${cost:.2f}")
            
    print(f"\nTotal Active Count: {active}")
    print(f"Total Locked Cash: ${locked_cash:.2f}")
    print(f"Mock Cash: ${state.get('mock_cash', 0.0):.2f}")
else:
    print("State file not found.")
