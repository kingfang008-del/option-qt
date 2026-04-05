import numpy as np
import sys
import os
from unittest.mock import MagicMock

# 模拟缺失的依赖，确保测试脚本能跑起来
mock_scipy = MagicMock()
sys.modules["scipy"] = mock_scipy
sys.modules["scipy.stats"] = MagicMock()
sys.modules["scipy.optimize"] = MagicMock()
sys.modules["py_vollib_vectorized"] = MagicMock()

# 强制将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入我们的生产代码逻辑
# 注意：由于 greeks_math.py 内部 import 了 scipy，我们需要在 import 它之前 mock 掉依赖
from production.utils.greeks_math import calculate_bucket_greeks

def run_flow_logic_tests():
    print("\n" + "="*60)
    print("🚀 STARTING MOCKED GREEKS FLOW LOGIC TEST")
    print("目标：验证 0.5 占位符是否被彻底清除")
    print("="*60)
    
    # 模拟输入参数
    S = 450.0
    r = 0.045
    T = 0.01 
    
    IDX_IV = 7
    IDX_PRICE = 0
    IDX_STRIKE = 5

    def get_dirty_buckets():
        # 构造一个 6x12 的矩阵，预填满 0.5 作为脏数据
        b = np.full((6, 12), 0.5, dtype=np.float32)
        # 设置一些基础价格和行权价，否则逻辑可能过早返回
        b[:, IDX_PRICE] = 1.0
        b[:, IDX_STRIKE] = 450.0
        return b

    # --- TEST CASE 1: 验证无合约行是否清零 ---
    print("\n[TEST 1] 验证空合约行是否被强制清零 (Dead Rows)")
    buckets = get_dirty_buckets()
    # 只有前两行有合约，中间有 None，有空字符串
    contracts = ["TSLA250102C00450000", "TSLA250102C00455000", "", None, "INVALID", ""]
    
    # 执行计算逻辑
    calculate_bucket_greeks(buckets, S, T, r, contracts)
    
    success_1 = True
    for i in range(len(buckets)):
        iv_val = buckets[i, IDX_IV]
        if i < 2:
            # 前两行应该被处理过（因为 mock 了计算，它们会被覆盖为 mock 返回的默认 0.0 或其他值）
            print(f"Row {i} (Active): {iv_val}")
        else:
            # 重点：第 2, 3, 4, 5 行本该跳过，验证它们是否变为了 0.0 而不是保留 0.5
            if abs(iv_val - 0.5) < 1e-6:
                print(f"❌ FAILED: Row {i} (Ticker: {contracts[i]}) 仍然保留了 0.5!")
                success_1 = False
            else:
                print(f"✅ SUCCESS: Row {i} (Ticker: {contracts[i]}) 已被清零")

    # --- TEST CASE 2: 验证 S=0 或 T=0 时的熔断清零 ---
    print("\n[TEST 2] 验证无效 S/T 时的全局清零")
    buckets = get_dirty_buckets()
    contracts = ["TSLA250102C00450000"] * 6
    
    # 测试 T=0 时的行为
    calculate_bucket_greeks(buckets, S, 0.0, r, contracts)
    if np.any(np.isclose(buckets[:, IDX_IV], 0.5)):
        print("❌ FAILED: S=450, T=0 时依然残留了 0.5!")
        success_2 = False
    else:
        print("✅ SUCCESS: S=450, T=0 已触发全局清零")
        success_2 = True

    # --- TEST CASE 3: 零价格处理 ---
    print("\n[TEST 3] 验证零价格是否覆盖 0.5")
    buckets = get_dirty_buckets()
    buckets[:, IDX_PRICE] = 0.0 # 价格为 0
    contracts = ["TSLA250102C00450000"] * 6
    
    calculate_bucket_greeks(buckets, S, T, r, contracts)
    if np.any(np.isclose(buckets[:, IDX_IV], 0.5)):
        print("❌ FAILED: 零价格行依然残留了 0.5!")
        success_3 = False
    else:
        print("✅ SUCCESS: 零价格正确将 0.5 覆盖为 0.0")
        success_3 = True

    print("\n" + "="*60)
    print(f"🏁 结论: {'✅ 通过' if (success_1 and success_2 and success_3) else '❌ 失败'}")
    print("="*60)

if __name__ == "__main__":
    run_flow_logic_tests()
