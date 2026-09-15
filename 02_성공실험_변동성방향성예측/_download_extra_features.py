# -*- coding: utf-8 -*-
"""Download additional financial features via yfinance to boost ETF return prediction.

Sources:
- yfinance (Yahoo Finance) — free, no API key
- Date range: 2013-12-01 to 2024-12-31 (extends existing 2014-01-01 to 2022-06-26)

Output:
- USD/extra_features.csv  (US market additions)
- UK/extra_features.csv   (UK market additions)
"""
import os
import sys
import time
import pandas as pd
import yfinance as yf

START = "2013-12-01"
END   = "2024-12-31"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USD_DIR  = os.path.join(BASE_DIR, "USD")
UK_DIR   = os.path.join(BASE_DIR, "UK")

# ---------------------------------------------------------------------------
# Symbol catalog
# ---------------------------------------------------------------------------
US_SYMBOLS = {
    # Broad market context
    "spy":      "SPY",       # S&P 500 ETF
    "iwm":      "IWM",       # Russell 2000 small-cap
    "dia":      "DIA",       # Dow Jones ETF
    # Sector ETFs (QQQ tech tilt)
    "xlk":      "XLK",       # Tech sector
    "soxx":     "SOXX",      # Semiconductors
    "smh":      "SMH",       # Semiconductor (alt)
    "xlf":      "XLF",       # Financials
    "xle":      "XLE",       # Energy
    "xlv":      "XLV",       # Healthcare
    # Dollar / credit / duration
    "uup":      "UUP",       # USD index ETF (DXY proxy)
    "hyg":      "HYG",       # High yield bond ETF
    "lqd":      "LQD",       # Investment grade bond
    "tlt":      "TLT",       # 20+ year treasury
    # Yield curve
    "us5y":     "^FVX",      # 5Y treasury yield
    "us30y":    "^TYX",      # 30Y treasury yield
    # Volatility regime
    "vix9d":    "^VIX9D",    # 9-day VIX
    "vvix":     "^VVIX",     # VIX of VIX
    # Asian overnight signal (key for US opens)
    "nikkei":   "^N225",
    "hsi":      "^HSI",
    # Crypto risk sentiment
    "eth":      "ETH-USD",
    # QQQ OHLCV (for technical features)
    "qqq_ohlcv": "QQQ",
}

UK_SYMBOLS = {
    # Broad European context (already have DAX/CAC/STOXX)
    "ftas":     "^FTAS",     # FTSE All-Share
    "ftmc":     "^FTMC",     # FTSE 250 (alt — should match existing ftse250)
    # Cross-Atlantic lead
    "spy":      "SPY",       # US lead-lag effect
    "iwm":      "IWM",       # US small-cap risk gauge
    # Sector ETFs (UK)
    "xlk":      "XLK",       # Global tech (FTSE100 has limited tech)
    "xle":      "XLE",       # Energy (FTSE100 heavy in energy)
    "xlf":      "XLF",       # Financials (FTSE100 heavy in financials)
    # Dollar / credit / duration
    "uup":      "UUP",
    "hyg":      "HYG",
    "lqd":      "LQD",
    "tlt":      "TLT",
    # Yield curve
    "us5y":     "^FVX",
    "us30y":    "^TYX",
    # Volatility regime
    "vix9d":    "^VIX9D",
    "vvix":     "^VVIX",
    # Asian overnight
    "nikkei":   "^N225",
    "hsi":      "^HSI",
    # Crypto risk sentiment
    "eth":      "ETH-USD",
    # FTSE100 ETF OHLCV (ISF.L is the iShares FTSE100 ETF)
    "isf_ohlcv": "ISF.L",
}


def fetch(symbol: str, retries: int = 3, sleep_s: float = 1.5) -> pd.DataFrame:
    """Fetch one symbol from yfinance with retries."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, start=START, end=END,
                auto_adjust=True, progress=False, threads=False,
            )
            if df is None or len(df) == 0:
                raise RuntimeError("empty frame")
            # Flatten possible MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            last_err = e
            print(f"   ! attempt {attempt} failed for {symbol}: {e}")
            time.sleep(sleep_s * attempt)
    raise RuntimeError(f"failed to fetch {symbol}: {last_err}")


def build_market(symbols: dict, label: str) -> pd.DataFrame:
    """Download all symbols, build wide dataframe indexed by Date."""
    frames = []
    for col_prefix, sym in symbols.items():
        print(f"[{label}] downloading {sym} -> {col_prefix} ...", flush=True)
        try:
            df = fetch(sym)
        except Exception as e:
            print(f"   X  skipping {sym}: {e}")
            continue

        # OHLCV symbols keep all columns (high-low range, volume)
        if col_prefix.endswith("_ohlcv"):
            base = col_prefix.replace("_ohlcv", "")
            sub = pd.DataFrame(index=df.index)
            sub[f"{base}_close"]  = df["Close"]
            sub[f"{base}_open"]   = df["Open"]
            sub[f"{base}_high"]   = df["High"]
            sub[f"{base}_low"]    = df["Low"]
            sub[f"{base}_volume"] = df["Volume"]
            # range and gap features
            sub[f"{base}_hl_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, pd.NA)
            sub[f"{base}_oc_gap"]   = (df["Close"] - df["Open"]) / df["Open"].replace(0, pd.NA)
            frames.append(sub)
        else:
            sub = pd.DataFrame(index=df.index)
            sub[col_prefix] = df["Close"]
            frames.append(sub)

    out = pd.concat(frames, axis=1)
    out.index.name = "Date"
    out = out.sort_index()
    out = out.reset_index()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


def main():
    print(f"Date range: {START} to {END}")
    print(f"Output dir: {BASE_DIR}\n")

    # US
    us = build_market(US_SYMBOLS, "US")
    us_path = os.path.join(USD_DIR, "extra_features.csv")
    us.to_csv(us_path, index=False)
    print(f"\n[US] saved -> {us_path}")
    print(f"     rows: {len(us):,}, cols: {len(us.columns)}")
    print(f"     columns: {list(us.columns)}\n")

    # UK
    uk = build_market(UK_SYMBOLS, "UK")
    uk_path = os.path.join(UK_DIR, "extra_features.csv")
    uk.to_csv(uk_path, index=False)
    print(f"\n[UK] saved -> {uk_path}")
    print(f"     rows: {len(uk):,}, cols: {len(uk.columns)}")
    print(f"     columns: {list(uk.columns)}")


if __name__ == "__main__":
    main()
