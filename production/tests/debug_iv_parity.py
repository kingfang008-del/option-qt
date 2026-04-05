import sys
import numpy as np
import pandas as pd
from py_vollib_vectorized import vectorized_implied_volatility

# Hardcode identical inputs
S = 192.10
K = 192.5
r = 0.0365

# calc T exactly like real dataset
expiry_dt = pd.to_datetime('260109', format='%y%m%d').tz_localize('America/New_York') + pd.Timedelta(hours=16)
ts_ny = pd.Timestamp(1767366060, unit='s', tz='UTC').tz_convert('America/New_York').floor('1min')
T = max(1e-6, (expiry_dt - ts_ny).total_seconds() / 31557600.0)

# Scenerio 1: 1s engine Price
iv_1s = vectorized_implied_volatility(np.array([3.925]), np.array([S]), np.array([K]), np.array([T]), np.array([r]), 'c', return_as='numpy')
print('1s engine IV (Price=3.925):', iv_1s[0])

# Scenerio 2: 1m engine Price
iv_1m = vectorized_implied_volatility(np.array([3.900]), np.array([S]), np.array([K]), np.array([T]), np.array([r]), 'c', return_as='numpy')
print('1m engine IV (Price=3.900):', iv_1m[0])
