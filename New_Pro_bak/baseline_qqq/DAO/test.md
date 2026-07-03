
```sql
SELECT
    COUNT(*) AS '总平仓笔数',

     
    ROUND(SUM(json_extract(details_json, '$.pnl')), 2) AS '总净盈亏($)',
    
     
    SUM(CASE WHEN json_extract(details_json, '$.pnl') > 0 THEN 1 ELSE 0 END) AS '盈利笔数',
    SUM(CASE WHEN json_extract(details_json, '$.pnl') <= 0 THEN 1 ELSE 0 END) AS '亏损笔数',
    ROUND(SUM(CASE WHEN json_extract(details_json, '$.pnl') > 0 THEN 1.0 ELSE 0.0 END) / COUNT(*) * 100, 2) || '%' AS '胜率',

     
    ROUND(SUM(CASE WHEN json_extract(details_json, '$.pnl') > 0 THEN json_extract(details_json, '$.pnl') ELSE 0 END), 2) AS '总盈利($)',
    ROUND(SUM(CASE WHEN json_extract(details_json, '$.pnl') < 0 THEN json_extract(details_json, '$.pnl') ELSE 0 END), 2) AS '总亏损($)',
    
    
    ROUND(
        (SUM(CASE WHEN json_extract(details_json, '$.pnl') > 0 THEN json_extract(details_json, '$.pnl') ELSE 0 END) / 
         NULLIF(SUM(CASE WHEN json_extract(details_json, '$.pnl') > 0 THEN 1 ELSE 0 END), 0)) 
        / 
        ABS(SUM(CASE WHEN json_extract(details_json, '$.pnl') < 0 THEN json_extract(details_json, '$.pnl') ELSE 0 END) / 
            NULLIF(SUM(CASE WHEN json_extract(details_json, '$.pnl') < 0 THEN 1 ELSE 0 END), 0)), 
    2) AS '单笔盈亏比'

FROM trade_logs_backtest
WHERE action = 'CLOSE';
```