import asyncio
import pickle
import time
import logging
from datetime import datetime
from production.baseline.system_orchestrator_v8 import V8Orchestrator
from production.history_replay.mock_ibkr_historical import MockIBKRHistorical

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("S5_HiFi_Replay")

class HighFidelityOrchestrator(V8Orchestrator):
    """
    [NEW] 高仿真回测引擎
    继承 V8Orchestrator，但改造了 process_batch 以支持秒级 Tick 注入。
    """
    async def process_batch(self, payload: dict):
        r_type = payload.get('replay_type', 'slow')
        curr_ts = payload.get('ts')
        
        if r_type == 'slow':
            # 1. SLOW (1min) 模式：运行完整的 Alpha 信号计算、开仓探测和常规平仓
            # logger.debug(f"⏳ Processing SLOW batch at {curr_ts}")
            await super().process_batch(payload)
            
        elif r_type == 'fast':
            # 2. FAST (1sec) 模式：模拟实盘中的高频 Fused Tick 监听
            # 仅触发 _process_fast_fused_tick 进行止盈止损监控
            # logger.debug(f"⚡ Processing FAST tick at {curr_ts}")
            
            # 由于回放 packet 是按 batch 组织的，我们需要拆解给 _process_fast_fused_tick
            symbols = payload.get('symbols', [])
            for i, sym in enumerate(symbols):
                # 构造符合 _process_fast_fused_tick 结构的 payload
                # 预提取该标的多维盘口数据
                c_price = payload.get('feed_call_price', [0]*len(symbols))[i]
                c_bid = payload.get('feed_call_bid', [0]*len(symbols))[i]
                c_ask = payload.get('feed_call_ask', [0]*len(symbols))[i]
                
                p_price = payload.get('feed_put_price', [0]*len(symbols))[i]
                p_bid = payload.get('feed_put_bid', [0]*len(symbols))[i]
                p_ask = payload.get('feed_put_ask', [0]*len(symbols))[i]
                
                tick_payload = {
                    'symbol': sym,
                    'ts': curr_ts,
                    'stock': {'close': payload['stock_price'][i]},
                    'option_buckets': [
                        # Index 0: PUT_ATM (根据 config.TAG_TO_INDEX)
                        [p_price, 0, 0, 0, 0, 0, 0, 0, p_bid, p_ask],
                        # Index 1: 其他...
                        [],
                        # Index 2: CALL_ATM
                        [c_price, 0, 0, 0, 0, 0, 0, 0, c_bid, c_ask]
                    ]
                }
                # 注意：_process_fast_fused_tick 内部会根据 st.position 自动选 bucket
                await self._process_fast_fused_tick(tick_payload)
            
            # 同时将秒级价格喂给 MockIBKR 以便记录 market_history (用于 +1s 延迟成交匹配)
            self.mock_ibkr.record_market_data(payload)

async def main():
    # 1. 强制设为 'backtest' 模式
    orchestrator = HighFidelityOrchestrator(mode='backtest')
    
    # 2. 注入增强版的 MockIBKR (支持秒级延迟)
    mock_ibkr = MockIBKRHistorical()
    mock_ibkr.execution_delay_seconds = 1 # 开启 1 秒精准执行延迟
    orchestrator.mock_ibkr = mock_ibkr
    
    logger.info("🚀 High-Fidelity Second-Level Replay Started.")
    await orchestrator.run()
    
    # 3. 结束后打印报告
    orchestrator.mock_ibkr.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
