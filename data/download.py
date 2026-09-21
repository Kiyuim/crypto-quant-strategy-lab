import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# 目标标的与时间范围 (2024-01 至 2026-08)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_YEAR, START_MONTH = 2024, 1
END_YEAR, END_MONTH = 2026, 8

BASE_URL = "https://data.binance.vision/data"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 标准化 K 线列名 (Binance 官方导出标准)
KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def generate_year_months(s_y, s_m, e_y, e_m):
    ym_list = []
    for y in range(s_y, e_y + 1):
        m_start = s_m if y == s_y else 1
        m_end = e_m if y == e_y else 12
        for m in range(m_start, m_end + 1):
            ym_list.append((y, m))
    return ym_list


def fetch_and_parse_zip(url: str, is_kline: bool = True) -> pd.DataFrame:
    """下载 ZIP 到内存并解析 CSV"""
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return pd.DataFrame()
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            # 兼容部分月份带 header、部分不带 header 的历史遗留情况
            first_line = f.readline().decode("utf-8")
            f.seek(0)
            has_header = "open_time" in first_line or "calc_time" in first_line

            if is_kline:
                df = pd.read_csv(
                    f,
                    header=0 if has_header else None,
                    names=KLINE_COLS if not has_header else None,
                )
                if has_header:
                    df.columns = KLINE_COLS
            else:
                df = pd.read_csv(f)
                # 资金费率列名重命名为标准格式
                df.rename(
                    columns={
                        "calc_time": "timestamp",
                        "funding_rate": "funding_rate",
                        "last_funding_rate": "funding_rate",
                    },
                    inplace=True,
                )
            return df


def process_klines(market: str = "futures/um"):
    """
    market: 'futures/um' (合约) 或 'spot' (现货)
    """
    for symbol in SYMBOLS:
        print(f"[*] Processing {market} 1h Klines for {symbol}...")
        dfs = []
        for y, m in generate_year_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
            url = f"{BASE_URL}/{market}/monthly/klines/{symbol}/1h/{symbol}-1h-{y}-{m:02d}.zip"
            try:
                sub_df = fetch_and_parse_zip(url, is_kline=True)
                if not sub_df.empty:
                    dfs.append(sub_df)
            except Exception as e:
                print(f"    Failed to fetch {url}: {e}")

        if not dfs:
            print(f"[!] No data collected for {symbol}")
            continue

        full_df = pd.concat(dfs, ignore_index=True)

        # 字段类型转换与时间索引建立
        full_df["open_time"] = pd.to_datetime(full_df["open_time"], unit="ms", utc=True)
        full_df.set_index("open_time", inplace=True)
        full_df.sort_index(inplace=True)
        full_df = full_df[~full_df.index.duplicated(keep="first")]

        # 转为高效数值类型
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
        ]
        full_df[numeric_cols] = full_df[numeric_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        market_tag = "perp" if "futures" in market else "spot"
        out_path = OUTPUT_DIR / f"{symbol}_{market_tag}_1h.parquet"
        full_df[numeric_cols].to_parquet(
            out_path, engine="pyarrow", compression="snappy"
        )
        print(f"[✓] Saved {symbol} klines: {out_path} ({len(full_df)} rows)")


def process_funding_rates():
    """获取 U 本位合约资金费率"""
    for symbol in SYMBOLS:
        print(f"[*] Processing Funding Rate for {symbol}...")
        dfs = []
        for y, m in generate_year_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
            url = f"{BASE_URL}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{y}-{m:02d}.zip"
            try:
                sub_df = fetch_and_parse_zip(url, is_kline=False)
                if not sub_df.empty:
                    dfs.append(sub_df)
            except Exception as e:
                print(f"    Failed to fetch {url}: {e}")

        if not dfs:
            continue

        full_df = pd.concat(dfs, ignore_index=True)
        time_col = "timestamp" if "timestamp" in full_df.columns else "calc_time"
        full_df["timestamp"] = pd.to_datetime(full_df[time_col], unit="ms", utc=True)
        full_df.set_index("timestamp", inplace=True)
        full_df.sort_index(inplace=True)
        full_df = full_df[~full_df.index.duplicated(keep="first")]

        rate_col = (
            "funding_rate" if "funding_rate" in full_df.columns else "last_funding_rate"
        )
        full_df["funding_rate"] = pd.to_numeric(full_df[rate_col], errors="coerce")

        out_path = OUTPUT_DIR / f"{symbol}_funding.parquet"
        full_df[["funding_rate"]].to_parquet(
            out_path, engine="pyarrow", compression="snappy"
        )
        print(f"[✓] Saved {symbol} funding: {out_path} ({len(full_df)} rows)")


if __name__ == "__main__":
    # 1. 抓取 U 本位合约 1h K 线（核心）
    process_klines(market="futures/um")
    # 2. 抓取 U 本位资金费率（核心）
    process_funding_rates()
    # 3. 如果需要现货 K 线用于计算期现基差，取消下行注释：
    # process_klines(market="spot")
