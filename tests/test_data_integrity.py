"""
tests/test_data_integrity.py
保证底层时序数据的单调性与数值完整性
"""
from pathlib import Path
import pandas as pd
import pytest

PROCESSED_DIR = Path("data/processed")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

@pytest.mark.parametrize("symbol", SYMBOLS)
def test_kline_parquet_integrity(symbol):
    file_path = PROCESSED_DIR / f"{symbol}_kline_1h.parquet"
    assert file_path.exists(), f"未找到文件: {file_path}"
    
    df = pd.read_parquet(file_path)
    assert not df.empty, f"{symbol} 数据为空"
    # 校验时间戳单调递增 (绝对不能有时间倒流)
    assert df.index.is_monotonic_increasing, f"{symbol} 时间戳不是单调递增"
    # 校验无重复时间戳
    assert not df.index.has_duplicates, f"{symbol} 存在重复时间戳"
    
    # 价格与成交量必须全部合法
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"缺少必要列: {col}"
        assert df[col].isna().sum() == 0, f"{symbol} 的 {col} 存在 NaN 缺失值"
        assert (df[col] >= 0).all(), f"{symbol} 的 {col} 存在负数非法值"

@pytest.mark.parametrize("symbol", SYMBOLS)
def test_funding_parquet_integrity(symbol):
    file_path = PROCESSED_DIR / f"{symbol}_funding.parquet"
    assert file_path.exists(), f"未找到文件: {file_path}"
    
    df = pd.read_parquet(file_path)
    assert not df.empty, f"{symbol} 资金费率数据为空"
    assert df.index.is_monotonic_increasing, f"{symbol} 资金费率时间戳未单调递增"
    assert not df.index.has_duplicates, f"{symbol} 资金费率存在重复时间戳"
    assert "funding_rate" in df.columns, "缺少 funding_rate 列"
    assert df["funding_rate"].isna().sum() == 0, f"{symbol} 资金费率存在 NaN"