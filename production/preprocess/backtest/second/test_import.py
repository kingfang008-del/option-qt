import sys
import os
from pathlib import Path

PROJECT_ROOT = Path("/Users/fangshuai/Documents/GitHub/option-qt")
sys.path.append(str(PROJECT_ROOT / "production"))
sys.path.append(str(PROJECT_ROOT / "production/baseline"))
sys.path.append(str(PROJECT_ROOT / "production/baseline/DAO"))

try:
    from realtime_feature_engine import RealTimeFeatureEngine
    print("✅ RealTimeFeatureEngine imported successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
