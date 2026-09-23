"""
src/data/download.py
"""
import io
import zipfile
from pathlib import Path
import pandas as pd
import requests

# 标的：三大主流币（合约代码）
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
# 时间区间：2024年1月 到 2026年8月
START_YEAR, START_MONTH = 2024, 1
END_YEAR, END_MONTH = 2026, 8

BASE_URL = "https://data.binance.vision/data"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 币安官方导出的 1h K 线列名定义
KLINE_SCHEMA = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]

def generate_year_months(s_y, s_m, e_y, e_m):
    """生成 (年, 月) 元组列表"""
    ym_list = []
    for y in range(s_y, e_y + 1):
        m_start = s_m if y == s_y else 1
        m_end = e_m if y == e_y else 12
        for m in range(m_start, m_end + 1):
            ym_list.append((y, m))
    return ym_list

def fetch_zip_csv(url: str, is_kline: bool = True) -> pd.DataFrame:
    """下载 ZIP 文件到内存流中，直接解压读取 CSV 为 DataFrame"""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                first_line = f.readline().decode("utf-8")
                f.seek(0)
                # 检查第一行是否包含表头
                has_header = "open_time" in first_line or "calc_time" in first_line
                
                if is_kline:
                    df = pd.read_csv(f, header=0 if has_header else None)
                    df.columns = KLINE_SCHEMA
                    return df
                else:
                    df = pd.read_csv(f)
                    time_col = "timestamp" if "timestamp" in df.columns else "calc_time"
                    rate_col = "funding_rate" if "funding_rate" in df.columns else "last_funding_rate"
                    return df[[time_col, rate_col]].rename(
                        columns={time_col: "timestamp", rate_col: "funding_rate"}
                    )
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return pd.DataFrame()

def run_download():
    ym_tuples = generate_year_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    
    for symbol in SYMBOLS:
        print(f"[*] 正在下载 {symbol} 1h K线数据...")
        klines_accum = []
        for y, m in ym_tuples:
            url = f"{BASE_URL}/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{y}-{m:02d}.zip"
            sub_df = fetch_zip_csv(url, is_kline=True)
            if not sub_df.empty:
                klines_accum.append(sub_df)
                
        if klines_accum:
            df_k = pd.concat(klines_accum, ignore_index=True)
            # 将毫秒时间戳转为 UTC 的 DatetimeIndex
            df_k["timestamp"] = pd.to_datetime(df_k["open_time"], unit="ms", utc=True)
            df_k.set_index("timestamp", inplace=True)
            df_k.sort_index(inplace=True)
            # 去除可能存在的重复时间戳
            df_k = df_k[~df_k.index.duplicated(keep="first")]
            
            # 转为浮点数
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
            df_k[numeric_cols] = df_k[numeric_cols].astype("float64")
            
            # 导出 Parquet 格式
            out_file = PROCESSED_DIR / f"{symbol}_kline_1h.parquet"
            df_k[numeric_cols].to_parquet(out_file, engine="pyarrow", compression="snappy")
            print(f"    ✓ 已保存: {out_file} (共 {len(df_k)} 行)")

        print(f"[*] 正在下载 {symbol} 8h 资金费率...")
        funding_accum = []
        for y, m in ym_tuples:
            url = f"{BASE_URL}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip"
            sub_df = fetch_zip_csv(url, is_kline=False)
            if not sub_df.empty:
                funding_accum.append(sub_df)
                
        if funding_accum:
            df_f = pd.concat(funding_accum, ignore_index=True)
            df_f["timestamp"] = pd.to_datetime(df_f["timestamp"], unit="ms", utc=True)
            df_f.set_index("timestamp", inplace=True)
            df_f.sort_index(inplace=True)
            df_f = df_f[~df_f.index.duplicated(keep="first")]
            df_f["funding_rate"] = df_f["funding_rate"].astype("float64")
            
            out_file = PROCESSED_DIR / f"{symbol}_funding.parquet"
            df_f[["funding_rate"]].to_parquet(out_file, engine="pyarrow", compression="snappy")
            print(f"    ✓ 已保存: {out_file} (共 {len(df_f)} 次结算)")

if __name__ == "__main__":
    run_download()