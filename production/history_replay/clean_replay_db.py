import os
# 🚀 [强制] 确保清理脚本运行在回放/仿真模式，从而准确定位到 Redis DB 1
os.environ["RUN_MODE"] = "LIVEREPLAY"

import psycopg2
import redis
from datetime import datetime
import pytz
from config import PG_DB_URL, IS_SIMULATED, get_redis_db, REDIS_CFG

# 设定你要回放的目标日期 (例如 2026-01-02)
target_date = datetime(2026, 1, 2, tzinfo=pytz.timezone('America/New_York'))
start_ts = int(target_date.timestamp())

print(f"🧨 清理 {target_date.strftime('%Y-%m-%d')} 及之后的未来数据 [PostgreSQL & Redis]...")

conn = psycopg2.connect(PG_DB_URL)
c = conn.cursor()

# 1. 精准删除回放当天的行情和特征，但保留之前的用于预热！
c.execute("DELETE FROM market_bars_1m WHERE ts >= %s", (start_ts,))
c.execute("DELETE FROM market_bars_5m WHERE ts >= %s", (start_ts,))
c.execute("DELETE FROM option_snapshots_1m WHERE ts >= %s", (start_ts,))
c.execute("DELETE FROM option_snapshots_5m WHERE ts >= %s", (start_ts,))
c.execute("DELETE FROM alpha_logs WHERE ts >= %s", (start_ts,))

# 2. 全局清空状态与流水
c.execute("DELETE FROM trade_logs_backtest;")
c.execute("DELETE FROM symbol_state;")

conn.commit()
conn.close()
print("✅ PostgreSQL 数据清理完成！")

# 3. 清理 Redis (动态识别 DB)
try:
    target_db = get_redis_db()
    r = redis.Redis(host=REDIS_CFG['host'], port=REDIS_CFG['port'], db=target_db)
    
    # 获取常用的流和同步键
    streams = ['fused_market_stream', 'trade_log_stream', 'unified_inference_stream', 'live_option_snapshot']
    sync_keys = r.keys("sync:*")
    
    print(f"🧹 正在清理 Redis DB {target_db}...")
    for s in streams:
        if r.exists(s):
            r.delete(s)
            print(f"  - 已删除 Stream: {s}")
    
    for k in sync_keys:
        r.delete(k)
        print(f"  - 已删除 Sync Key: {k.decode() if isinstance(k, bytes) else k}")
    
    # 额外清理 5min/1min 的缓存状态 (由 FeatureService 维护)
    r.delete("feature_service_state")
        
    print(f"✨ Redis DB {target_db} 清理完成！")
except Exception as e:
    print(f"❌ Redis 清理失败: {e}")

print("✅ 环境已 100% 纯净，可以安全启动量化系统！")