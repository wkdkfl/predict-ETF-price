# -*- coding: utf-8 -*-
"""GDELT 기반 일별 뉴스 피처 수집 — 기존 뉴스 데이터의 커버리지 공백을 메우기 위한 스크립트.

왜 필요한가
-----------
기존 news_per_headline.csv 는 2017~2025 기간에 대해 고유 날짜가
미국 221일 / 영국 713일뿐이다(연도별 약 3,000건에서 수집이 끊긴 흔적).
모델 학습 구간 기준으로 실제 뉴스가 존재하는 날의 비율은
미국 3.2% / 영국 12.9% 에 불과하며, 나머지는 직전 가용일 값이 이월된다.
따라서 "뉴스가 예측에 기여하지 않는다"는 현재 결과는 뉴스의 정보력이 아니라
데이터 부재를 반영한 것일 수 있다.

사용법
------
  python _collect_gdelt_news.py            # timeline 모드 (권장, 요청 수십 건)
  python _collect_gdelt_news.py --artlist  # 일별 헤드라인 수집 (요청 수천 건, 느림)
    python _collect_gdelt_news.py --smoke-test  # 단일 기간 요청으로 연결·추출만 점검

주의
----
GDELT API 는 IP 단위로 강하게 레이트 리밋한다. 공용/사내망 공유 IP 에서는
429 가 지속될 수 있으므로 개인 네트워크에서 실행할 것.

출력
----
  {USD,UK}/gdelt_daily_tone.csv     일자별 뉴스 톤/볼륨 시계열 (timeline 모드)
  {USD,UK}/gdelt_headlines.csv      일자별 헤드라인 (artlist 모드)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
API = "https://api.gdeltproject.org/api/v2/doc/doc?"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

START = date(2017, 1, 1)
END = date(2025, 12, 31)

MARKETS = {
    "USD": {
        "label": "US",
        "queries": {
            "market":    '("stock market" OR "S&P 500" OR "Wall Street" OR "Dow Jones") sourcecountry:US sourcelang:eng',
            "macro":     '("Federal Reserve" OR "interest rate" OR inflation OR "CPI report") sourcecountry:US sourcelang:eng',
            "risk":      '(recession OR "market selloff" OR volatility OR "market crash") sourcecountry:US sourcelang:eng',
            "earnings":  '(earnings OR "quarterly results" OR "profit warning") sourcecountry:US sourcelang:eng',
        },
    },
    "UK": {
        "label": "UK",
        "queries": {
            "market":    '("FTSE 100" OR "London stock" OR "UK shares" OR "London market") sourcecountry:UK sourcelang:eng',
            "macro":     '("Bank of England" OR "UK inflation" OR "interest rate" OR "UK economy") sourcecountry:UK sourcelang:eng',
            "risk":      '(recession OR "market selloff" OR volatility OR "pound falls") sourcecountry:UK sourcelang:eng',
            "earnings":  '(earnings OR "trading update" OR "profit warning") sourcecountry:UK sourcelang:eng',
        },
    },
}

MAX_TRIES = 8
BASE_WAIT = 20


def call(params: dict):
    url = API + urllib.parse.urlencode(params)
    for i in range(MAX_TRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            wait = BASE_WAIT * (i + 1)
            print("      재시도 %d/%d (%ds 대기): %s" % (i + 1, MAX_TRIES, wait, str(e)[:60]))
            time.sleep(wait)
    return None


def timeline_series(query: str, mode: str, start: date, end: date, chunk_days=365):
    """기간을 잘라 timeline 모드로 호출. 한 청크당 요청 1건."""
    rows = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        d = call({"query": query, "mode": mode, "format": "json",
                  "startdatetime": cur.strftime("%Y%m%d") + "000000",
                  "enddatetime": stop.strftime("%Y%m%d") + "235959",
                  "timelinesmooth": 0})
        if d:
            for s in d.get("timeline", []):
                for p in s.get("data", []):
                    rows.append({"date": p.get("date"), "value": p.get("value")})
            print("      %s ~ %s : %d 포인트 누적" % (cur, stop, len(rows)))
        else:
            print("      %s ~ %s : 실패" % (cur, stop))
        cur = stop + timedelta(days=1)
        time.sleep(8)
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date"]).drop_duplicates("date")


def run_timeline():
    for sub, cfg in MARKETS.items():
        print("=" * 66)
        print("  %s — timeline 수집" % cfg["label"])
        merged = None
        for topic, q in cfg["queries"].items():
            for mode, suffix in [("timelinetone", "tone"), ("timelinevolraw", "vol")]:
                print("    [%s / %s]" % (topic, suffix))
                s = timeline_series(q, mode, START, END)
                if s.empty:
                    continue
                s = s.rename(columns={"value": f"{topic}_{suffix}"})
                merged = s if merged is None else merged.merge(s, on="date", how="outer")
        if merged is None:
            print("    수집 실패 — 네트워크/레이트리밋 확인 필요")
            continue
        merged = merged.sort_values("date").reset_index(drop=True)
        out = BASE / sub / "gdelt_daily_tone.csv"
        merged.to_csv(out, index=False)
        print("    저장: %s  (%d일, %d열)" % (out, len(merged), merged.shape[1] - 1))


def run_artlist():
    for sub, cfg in MARKETS.items():
        print("=" * 66)
        print("  %s — 일별 헤드라인 수집 (느림)" % cfg["label"])
        q = cfg["queries"]["market"]
        rows, cur = [], START
        while cur <= END:
            if cur.weekday() < 5:
                d = call({"query": q, "mode": "artlist", "format": "json",
                          "maxrecords": 250, "sort": "hybridrel",
                          "startdatetime": cur.strftime("%Y%m%d") + "000000",
                          "enddatetime": cur.strftime("%Y%m%d") + "235959"})
                if d:
                    for a in d.get("articles", []):
                        rows.append({"date": cur.isoformat(),
                                     "headline": a.get("title", ""),
                                     "source": a.get("domain", "")})
                    print("    %s : 누적 %d건" % (cur, len(rows)))
                time.sleep(8)
            cur += timedelta(days=1)
        if rows:
            out = BASE / sub / "gdelt_headlines.csv"
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
            print("    저장: %s (%d건)" % (out, len(rows)))


def run_smoke_test():
    """Issue two small requests only; do not write data files or start a collection."""
    start = date(2024, 1, 2)
    end = date(2024, 1, 3)
    print("GDELT 연결·추출 점검: %s ~ %s" % (start, end))
    for sub, cfg in MARKETS.items():
        response = call({
            "query": cfg["queries"]["market"],
            "mode": "artlist",
            "format": "json",
            "maxrecords": 10,
            "sort": "hybridrel",
            "startdatetime": start.strftime("%Y%m%d") + "000000",
            "enddatetime": end.strftime("%Y%m%d") + "235959",
        })
        if response is None:
            print("  %s: 실패 (네트워크 또는 GDELT 레이트 리밋)" % cfg["label"])
            continue
        articles = response.get("articles", [])
        print("  %s: 성공, %d건 수신" % (cfg["label"], len(articles)))
        for article in articles[:3]:
            print("    - %s [%s]" % (article.get("title", "")[:120], article.get("domain", "")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artlist", action="store_true", help="일별 헤드라인 수집 모드")
    ap.add_argument("--smoke-test", action="store_true", help="GDELT 연결·추출 점검만 실행")
    a = ap.parse_args()
    if a.smoke_test:
        run_smoke_test()
    elif a.artlist:
        run_artlist()
    else:
        run_timeline()
