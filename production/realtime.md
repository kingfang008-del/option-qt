```plaintext

[5s Bar Stream] 
      |
      v
[FeatureComputeService] (维护 200分钟的 1min Buffer)
      | <--- 每分钟触发 (09:31, 09:32...)
      |
      +---> 1. 切片 Fast: 取最后 10 个 1min Bar -> 算 Fast Feats
      |
      +---> 2. 重采样 Slow: 将 200min Buffer resample('5T') -> 取最后 30 个 -> 算 Slow Feats
      |
      +---> 3. 聚合 Option: 获取最新的期权快照
      |
      v
[Redis Stream: 'unified_feature_stream'] (包含 FastTensor, SlowTensor, OptionData)
      |
      v
[SystemOrchestrator]
      |---> Model: Fast(10) + Slow(30) -> Student -> Alpha
      |---> RL: Alpha + Portfolio + Option -> Action
```
