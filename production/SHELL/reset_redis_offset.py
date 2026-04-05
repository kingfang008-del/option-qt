import redis

# 配置
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
DB = 0

# 定义需要重置的流和对应的消费组
# 格式: ('流名称', '消费组名称')
TARGETS = [
    ('fused_market_stream', 'persistence_group'),      # 持久化服务
    ('unified_inference_stream', 'persistence_group'), # 持久化服务存特征
    # 如果 Orchestrator 也有消费组，在这里添加，例如:
    # ('unified_inference_stream', 'strategy_group'),
]

def reset_offsets():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=DB)
    
    print(f"🔌 Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    
    for stream, group in TARGETS:
        try:
            # 检查流是否存在
            if not r.exists(stream):
                print(f"⚠️  Stream not found: {stream}")
                continue
                
            # 重置 ID 为 '$' (即当前最新消息)
            # 这意味着消费者只会收到脚本执行之后产生的新消息
            r.xgroup_setid(stream, group, '$')
            print(f"✅ Reset {stream} :: {group} -> Latest ($)")
            
        except redis.exceptions.ResponseError as e:
            if "no such key" in str(e):
                print(f"⚠️  Stream {stream} does not exist yet.")
            elif "NOGROUP" in str(e):
                print(f"⚠️  Group {group} does not exist. It will be created on service start.")
            else:
                print(f"❌ Error resetting {stream}: {e}")

if __name__ == "__main__":
    reset_offsets()