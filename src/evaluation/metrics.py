"""
src/evaluation/metrics.py
核心量化收益、波动率、夏普比率与最大回撤指标计算引擎
"""
import numpy as np
import pandas as pd

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    计算连续对数收益率: r_t = ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1))

def compute_simple_returns(prices: pd.Series) -> pd.Series:
    """
    计算简单收益率: R_t = (P_t - P_{t-1}) / P_{t-1}
    """
    return prices.pct_change()

def compute_equity_curve(returns: pd.Series, is_log: bool = False) -> pd.Series:
    """
    计算标准化累计净值曲线 (初始本金设为 1.0)
    """
    clean_ret = returns.fillna(0.0)
    if is_log:
        return np.exp(clean_ret.cumsum())
    return (1.0 + clean_ret).cumprod()

def compute_annualized_metrics(returns: pd.Series, periods_per_year: int = 8760) -> dict:
    """
    计算年化收益率、年化波动率与夏普比率
    参数:
        returns: 单周期收益率序列 (如 1h 收益率)
        periods_per_year: 一年内的周期数。加密货币 7x24 小时无休，1h 颗粒度为 24 * 365 = 8760
    """
    clean_ret = returns.dropna()
    if len(clean_ret) < 2:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0}
    
    mean_ret = clean_ret.mean()
    std_ret = clean_ret.std()
    
    # 期望按周期数线性放大
    ann_return = mean_ret * periods_per_year
    # 波动率按根号周期数放大 (平方根时间法则)
    ann_vol = std_ret * np.sqrt(periods_per_year)
    # 夏普比率 (无风险利率设为 0)
    sharpe = (ann_return / ann_vol) if ann_vol > 1e-8 else 0.0
    
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe)
    }

def compute_drawdown_series(equity_curve: pd.Series) -> tuple[float, pd.Series]:
    """
    计算最大回撤 (MDD) 与水下曲线 (Underwater Drawdown)
    返回:
        (max_drawdown, drawdown_series)
    """
    # 历史滚动最高点
    running_max = equity_curve.cummax()
    # 相对最高点的跌幅比率
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())  # 最小值即为最大负偏离
    return max_dd, drawdown