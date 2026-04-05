import logging
import math

logger = logging.getLogger("LiquidityManager")

class LiquidityRiskManager:
    """
    [终极形态] L2 深度盘口驱动期权订单规模风控引擎
    根据实时盘口卖一挂单量 (Ask Size)，动态计算免拆单资金上限。
    如果缺乏实盘 L2 数据（如回测模式），自动退化为 Tier 静态限额保护。
    """
    
    # === 静态分级免拆上限 (用于回测兜底) ===
    TIER_1_SYMBOLS = {'SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'IWM'}
    TIER_1_NO_CHUNK_CAP = 30000.0  
    
    TIER_2_SYMBOLS = {'META', 'AMZN', 'MSFT', 'AMD', 'MSTR', 'COIN', 'SMCI'}
    TIER_2_NO_CHUNK_CAP = 10000.0  
    
    TIER_3_NO_CHUNK_CAP = 3000.0   
    
    @classmethod
    def evaluate_order(cls, sym: str, target_alloc: float, option_price: float, mode: str = 'backtest', ask_size: float = 0.0) -> dict:
        """
        评估订单流动性风险，输出修正后的金额与拆单建议。
        """
        if target_alloc <= 0 or option_price <= 0 or math.isnan(option_price) or math.isnan(target_alloc):
            return {'final_alloc': 0.0, 'chunks': 0, 'reason': "Invalid inputs (NaN or <= 0)"}


            
        target_qty = int(target_alloc // (option_price * 100))
        if target_qty < 1:
            return {
                'final_alloc': 0.0, 'chunks': 0, 
                'reason': f"Alloc too small for 1 contract (Price: ${option_price:.2f})"
            }

        # ========================================================
        # 👑 [L2 动态盘口容量判定] (仅限实盘)
        # ========================================================
        if ask_size > 0:
            # 使用 L2 的卖一挂单量作为单笔免拆上限。
            # 为了极端安全，我们只吃当下的挂单量，绝不吃穿到下一档！
            safe_qty_cap = int(ask_size)
            chunk_cap_alloc = safe_qty_cap * option_price * 100
            tier_name = f"L2 Depth(Ask:{ask_size:,.0f}手)"
            
        # ========================================================
        # 🧱 [静态 Tier 兜底判定] (回测或盘前实盘缺失盘口数据时)
        # ========================================================
        else:
            if sym in cls.TIER_1_SYMBOLS:
                tier_name = "Static-Tier1"
                chunk_cap_alloc = cls.TIER_1_NO_CHUNK_CAP
            elif sym in cls.TIER_2_SYMBOLS:
                tier_name = "Static-Tier2"
                chunk_cap_alloc = cls.TIER_2_NO_CHUNK_CAP
            else:
                tier_name = "Static-Tier3"
                chunk_cap_alloc = cls.TIER_3_NO_CHUNK_CAP
                
        # 确保单笔最小分拆容量不低于 1 手的价格，否则无法发单
        min_alloc = option_price * 100
        chunk_cap_alloc = max(chunk_cap_alloc, min_alloc)
        
        # --- Chunking 拆单切片逻辑 ---
        capped_alloc = target_alloc # 总金额不截断，全部交给冰山吃进
        
        if capped_alloc > chunk_cap_alloc:
            chunks = math.ceil(capped_alloc / chunk_cap_alloc)
            reason = f"{tier_name} Cap: ${chunk_cap_alloc:,.0f} | Iceberg Split: {chunks} chunks"
        else:
            chunks = 1
            reason = f"{tier_name} Pass (Alloc < Cap ${chunk_cap_alloc:,.0f})"
            
        return {
            'final_alloc': capped_alloc,
            'chunks': chunks,
            'reason': reason
        }