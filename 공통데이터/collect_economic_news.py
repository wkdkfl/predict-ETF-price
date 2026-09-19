# -*- coding: utf-8 -*-
"""2014~2025 미국·영국 일별 경제 뉴스 헤드라인 수집 (매일 최소 1건 보장).

왜 GDELT 가 아닌가
------------------
02_성공실험_변동성방향성예측/_collect_gdelt_news.py 는 GDELT DOC API 를 쓰는데,
이 API 는 2017-01-01 이전 시작일을 거부하고("Invalid query start date"), IP 단위 레이트 리밋(429)이
심해 4,383일 × 2개국 조회가 사실상 불가능하다. 그래서 스크립트의 구조(시장별 설정, 재시도/백오프,
--smoke-test, 일자 루프)는 유지하고 소스만 일별 조회가 되는 아카이브로 교체했다.
자세한 실측 근거는 plan/20260919_economic_news_collection.md 참고.

소스
----
  US : NYT 일별 사이트맵   https://www.nytimes.com/sitemap/YYYY/MM/DD/            (business/your-money/upshot)
  UK : Guardian 비즈니스   https://www.theguardian.com/business/YYYY/mon/DD/all

사용법
------
  python collect_economic_news.py --smoke-test     # 표본 6일 조회만, 파일 미생성
  python collect_economic_news.py                  # 전체 수집(US+UK) 후 선별
  python collect_economic_news.py --market US      # 한 시장만
  python collect_economic_news.py --select-only    # 재크롤링 없이 후보 CSV 에서 재선별
  python collect_economic_news.py --verify         # 저장 결과 검증 + 리포트 생성
  python collect_economic_news.py --refetch-empty  # 후보 0건이었던 날짜만 다시 조회

중단 후 다시 실행하면 _news_cache/ 의 진행 기록을 읽어 이어서 수집한다.

출력 (공통데이터/{USD,UK}/)
--------------------------
  {US,UK}_economic_news_daily.csv       일별 선택본 (year,month,date,headline,source,section,url,
                                        relevance_score,day_rank,is_fallback)
  {US,UK}_economic_news_candidates.csv  점수화된 후보 전체 (감사·재선별용)
  공통데이터/economic_news_collection_report.md   --verify 결과
"""
from __future__ import annotations

import argparse
import csv
import html
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent
CACHE = BASE / "_news_cache"
START = date(2014, 1, 1)
END = date(2025, 12, 31)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-crawler/1.0 (academic thesis; low-rate)"

MAX_PER_DAY = 10     # 날짜당 최대 선택 건수
THRESH = 3           # 이 점수 이상이면 "의미 있는 뉴스"
MAX_TRIES = 6
ABORT_AFTER = 8      # 연속 실패 날짜 수가 이 값이면 차단으로 보고 중단

NYT_SECTIONS = {"business", "your-money", "upshot"}
GUARDIAN_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# 기존 파이프라인(_build_multi_headline_features.py)과 동일한 "장 마감 요약" 패턴 — 정보 누출 방지
CLOSE_PATTERNS = re.compile(
    r"\b(closed at|ended (the )?(day|session|week)|"
    r"(close|closes|closing) (higher|lower|up|down|mixed|flat)|"
    r"finished (higher|lower|up|down|mixed)|"
    r"end of (the )?(day|session|week)|"
    r"after the (close|bell)|"
    r"close: |stocks close|market close|wall street close)\b",
    re.IGNORECASE,
)

# 시장 영향도가 큰 주제 (가중 3) / 일반 경제어 (가중 1)
HIGH_KW = [re.compile(p, re.I) for p in (
    r"\bfederal reserve\b|\bthe fed\b|\bfed (chair|raises|cuts|holds|signals|officials?)\b|\bfomc\b|\bpowell\b|\byellen\b",
    r"\bbank of england\b|\bboe\b|\becb\b|\bcentral banks?\b|\bbailey\b|\blagarde\b",
    r"\binterest rates?\b|\brate (cut|cuts|hike|hikes|rise|rises|decision)\b|\bmonetary policy\b|\bquantitative easing\b",
    r"\binflation\b|\bdeflation\b|\bcpi\b|\bconsumer prices?\b",
    r"\bgdp\b|\brecession\b|\beconomic (growth|output|slowdown|contraction|outlook)\b",
    r"\bunemployment\b|\bjobless\b|\bpayrolls?\b|\bjobs (report|data|numbers)\b|\blabou?r market\b",
    r"\btariffs?\b|\btrade (war|deal|talks|deficit|dispute|tensions)\b|\bsanctions\b",
    r"\bstock markets?\b|\bwall street\b|\bs&p 500\b|\bdow\b|\bnasdaq\b|\bftse\b|\bstocks\b|\bshares\b|\bequities\b|\bsell-?off\b|\brally\b",
    r"\bearnings\b|\bprofits?\b|\brevenues?\b|\bquarterly results\b|\bprofit warning\b|\btrading update\b",
    r"\boil prices?\b|\bcrude\b|\bbrent\b|\bopec\b|\bgas prices?\b|\benergy prices?\b",
    r"\bdollar\b|\bsterling\b|\bpound\b|\beuro\b|\byuan\b|\byen\b|\bcurrency\b|\bexchange rate\b",
    r"\bbonds?\b|\bgilts?\b|\btreasur(y|ies)\b|\byields?\b",
    r"\bbudget\b|\bdeficit\b|\bnational debt\b|\bdebt (ceiling|crisis|limit)\b|\bausterity\b|\bstimulus\b|\bchancellor\b",
    r"\bbrexit\b|\bfinancial crisis\b|\bbailout\b|\bbankruptcy\b|\bdefault\b|\bcollapse\b",
    r"\bmergers?\b|\bacquisitions?\b|\btakeover\b|\bbuyout\b|\bipo\b|\bacquire[sd]?\b",
)]
MED_KW = [re.compile(p, re.I) for p in (
    r"\bbanks?\b|\bbanking\b|\blenders?\b|\bloans?\b",
    r"\bhousing\b|\bmortgages?\b|\bproperty\b|\bhome prices?\b|\brents?\b",
    r"\bretail(ers)?\b|\bconsumers?\b|\bspending\b|\bsales\b",
    r"\bmanufactur\w*\b|\bfactory\b|\bpmi\b|\bsupply chains?\b|\bexports?\b|\bimports?\b",
    r"\beconom(y|ic|ics|ists?)\b|\bmarkets?\b|\binvestors?\b|\bcompan(y|ies)\b|\bindustry\b",
    r"\bwages?\b|\blayoffs?\b|\bjob cuts\b|\bhiring\b|\bworkers?\b|\bpay\b",
    r"\bprices?\b|\bgrowth\b|\bfunds?\b|\bcrypto\w*\b|\bbitcoin\b|\bpensions?\b|\btax(es)?\b",
)]
NOISE = re.compile(
    r"\b(obituar\w*|crossword|podcast|quiz|newsletter|cartoon|horoscope|recipes?|celebrity|"
    r"movies?|album|oscars?|olympics?|football|cricket|tennis|golf|review:|match report)\b",
    re.I,
)
BOILERPLATE = {"corrections", "brief", "editors' note", "editor's note", "today's headlines"}

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
NYT_RE = re.compile(r'<li><a href="https://www\.nytimes\.com/(\d{4})/(\d{2})/(\d{2})/([^"]+)"[^>]*>([^<]+)</a>')
GUARDIAN_RE = re.compile(
    r'<h[1-4] class="fc-item__title"><a href="(https://www\.theguardian\.com/[^"]+)"[^>]*>(.*?)</a></h[1-4]>'
    r'(.*?)(?=<h[1-4] class="fc-item__title">|$)', re.S)
GUARDIAN_DATE = re.compile(r"/(\d{4})/([a-z]{3})/(\d{2})/")


def clean(s: str) -> str:
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s))).strip()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


# ---------------------------------------------------------------- 점수화
def score_item(market: str, headline: str, summary: str, section: str) -> int:
    text = headline + " " + summary
    high = sum(1 for p in HIGH_KW if p.search(text))
    med = sum(1 for p in MED_KW if p.search(text))
    s = 3 * min(high, 4) + min(med, 3)
    top, sub = (section.split("/") + [""])[:2]
    if market == "US":
        if section == "business/economy":
            s += 3
        elif section == "business/dealbook":
            s += 2
        elif top == "business":
            s += 1
        elif top in ("your-money", "upshot"):
            s += 1
    else:
        if top == "business":
            s += 2
        elif top == "money":
            s += 1
    if NOISE.search(headline):
        s -= 6
    return s


def make_row(market, d, headline, url, section, summary=""):
    return {
        "date": d.isoformat(), "headline": headline, "source": MARKETS[market]["source"],
        "section": section, "url": url, "summary": summary,
        "score": score_item(market, headline, summary, section),
        "is_close_summary": int(bool(CLOSE_PATTERNS.search(headline))),
    }


# ---------------------------------------------------------------- 파서
def parse_nyt(page: str, d: date):
    rows = []
    for y, m, dd, path, title in NYT_RE.findall(page):
        # 사이트맵 날짜 D 를 기사 날짜로 쓴다. 일요일/토요일 지면 기사는 URL 날짜가 하루 뒤로 올라오므로
        # (예: 2014-03-09 사이트맵의 비즈니스 기사 전부가 2014/03/10 URL) URL 날짜 ±1일까지 허용.
        if abs((date(int(y), int(m), int(dd)) - d).days) > 1:
            continue
        segs = path.split("/")
        section = segs[0] + ("/" + segs[1] if len(segs) > 2 else "")
        rows.append(make_row("US", d, clean(title), f"https://www.nytimes.com/{y}/{m}/{dd}/{path}", section))
    eligible = [r for r in rows if r["section"].split("/")[0] in NYT_SECTIONS]
    if eligible:
        return eligible
    # 경제 섹션 기사가 0건인 날 → 전체 섹션 중 점수 상위 5건만 후보로 (fallback 후보)
    return sorted(rows, key=lambda r: -r["score"])[:5]


def parse_guardian(page: str, d: date):
    rows = []
    for url, head_html, rest in GUARDIAN_RE.findall(page):
        m = GUARDIAN_DATE.search(url)
        if not m or (int(m.group(1)), m.group(2), int(m.group(3))) != (d.year, GUARDIAN_MONTHS[d.month - 1], d.day):
            continue
        hm = re.search(r'js-headline-text">(.*?)</span>', head_html, re.S)
        headline = clean(hm.group(1) if hm else head_html)
        sm = re.search(r'fc-item__standfirst">(.*?)</div>', rest, re.S)
        summary = clean(sm.group(1)) if sm else ""
        section = url.split("theguardian.com/", 1)[1].split("/")[0]
        rows.append(make_row("UK", d, headline, url, section, summary))
    return rows


def guardian_url(section: str, d: date) -> str:
    return f"https://www.theguardian.com/{section}/{d.year}/{GUARDIAN_MONTHS[d.month - 1]}/{d.day:02d}/all"


MARKETS = {
    "US": {"dir": "USD", "source": "nytimes.com", "parse": parse_nyt,
           "url": lambda d: f"https://www.nytimes.com/sitemap/{d.year}/{d.month:02d}/{d.day:02d}/"},
    "UK": {"dir": "UK", "source": "theguardian.com", "parse": parse_guardian,
           "url": lambda d: guardian_url("business", d)},
}
# Guardian 은 크리스마스 등 일부 휴일에 비즈니스 섹션 기사가 0건이다(실측: 2018-12-25). 그런 날만 아래 섹션에서 보충한다.
UK_FALLBACK_SECTIONS = ["money", "politics", "uk-news", "technology", "world"]


def paths(market: str):
    cfg = MARKETS[market]
    out = BASE / cfg["dir"]
    return {
        "candidates": out / f"{market}_economic_news_candidates.csv",
        "daily": out / f"{market}_economic_news_daily.csv",
        "progress": CACHE / f"{market}_days.csv",
    }


# ---------------------------------------------------------------- 네트워크
class Throttle:
    """스레드 공용 최소 요청 간격."""
    def __init__(self, min_interval: float):
        self.lock, self.next, self.min = threading.Lock(), 0.0, min_interval

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.min
        if t > now:
            time.sleep(t - now)


def fetch(url: str, throttle: Throttle):
    """페이지 본문 반환. 404 는 None. 반복 실패 시 예외."""
    last = None
    for i in range(MAX_TRIES):
        throttle.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                return None
        except Exception as e:  # 타임아웃 등
            last = e
        time.sleep(min(60, 3 * 2 ** i))
    raise RuntimeError("%s: %s" % (url, str(last)[:80]))


def crawl_day(market: str, d: date, throttle: Throttle):
    page = fetch(MARKETS[market]["url"](d), throttle)
    if page is None:
        return d, "404", []
    rows = MARKETS[market]["parse"](page, d)
    if not rows and market == "UK":
        extra = []
        for sec in UK_FALLBACK_SECTIONS:
            p = fetch(guardian_url(sec, d), throttle)
            if p:
                extra += parse_guardian(p, d)
        rows = sorted(extra, key=lambda r: -r["score"])[:5]
    return d, "ok", rows


# ---------------------------------------------------------------- 진행 기록 / 후보 저장
CAND_FIELDS = ["date", "headline", "source", "section", "url", "summary", "score", "is_close_summary"]


def load_progress(p: Path):
    done = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if len(parts) == 3:
                done[parts[0]] = (parts[1], int(parts[2]))
    return done


def run_collect(market, start, end, workers, min_interval, refetch_empty):
    pth = paths(market)
    pth["candidates"].parent.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    done = load_progress(pth["progress"])
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    todo = [d for d in days
            if d.isoformat() not in done or (refetch_empty and done[d.isoformat()][1] == 0 and done[d.isoformat()][0] != "404")]
    print("[%s] 대상 %d일 중 수집 필요 %d일 (기존 완료 %d일)" % (market, len(days), len(todo), len(days) - len(todo)), flush=True)
    if not todo:
        return True

    new_file = not pth["candidates"].exists()
    throttle = Throttle(min_interval)
    fails = 0
    finished = 0
    t0 = time.time()
    with pth["candidates"].open("a", newline="", encoding="utf-8-sig") as cf, \
            pth["progress"].open("a", encoding="utf-8") as pf, \
            ThreadPoolExecutor(max_workers=workers) as ex:
        w = csv.DictWriter(cf, fieldnames=CAND_FIELDS)
        if new_file:
            w.writeheader()
        futs = {ex.submit(crawl_day, market, d, throttle): d for d in todo}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                _, status, rows = fut.result()
            except Exception as e:
                fails += 1
                print("   실패 %s: %s" % (d, e), flush=True)
                if fails >= ABORT_AFTER:
                    print("[%s] 연속 %d건 실패 — 차단/네트워크 문제로 보고 중단합니다. 잠시 후 같은 명령으로 재개하세요." % (market, fails), flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    return False
                continue
            fails = 0
            for r in rows:
                w.writerow({k: r[k] for k in CAND_FIELDS})
            pf.write("%s,%s,%d\n" % (d.isoformat(), status, len(rows)))
            cf.flush(); pf.flush()
            finished += 1
            if finished % 100 == 0 or finished == len(todo):
                rate = finished / max(time.time() - t0, 1e-9)
                print("   [%s] %d/%d 일 완료  (%.1f일/초, 잔여 약 %.0f분)" % (market, finished, len(todo), rate, (len(todo) - finished) / max(rate, 1e-9) / 60), flush=True)
    return True


# ---------------------------------------------------------------- 선별
def select_day(rows):
    seen, uniq = set(), []
    for r in rows:
        k = norm(r["headline"])
        if k in seen or len(r["headline"]) < 15 or r["headline"].lower() in BOILERPLATE:
            continue
        seen.add(k)
        uniq.append(r)
    good = sorted((r for r in uniq if not r["is_close_summary"] and r["score"] >= THRESH), key=lambda r: -r["score"])
    if good:
        return [(r, 0) for r in good[:MAX_PER_DAY]]
    pool = [r for r in uniq if not r["is_close_summary"]] or uniq
    if not pool:
        return []
    return [(max(pool, key=lambda r: r["score"]), 1)]   # 그날 최고점 1건을 fallback 으로


DAILY_FIELDS = ["year", "month", "date", "headline", "source", "section", "url", "relevance_score", "day_rank", "is_fallback"]


def run_select(market):
    pth = paths(market)
    if not pth["candidates"].exists():
        print("[%s] 후보 파일이 없습니다: %s" % (market, pth["candidates"]))
        return
    by_day = {}
    with pth["candidates"].open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r["score"] = int(r["score"])
            r["is_close_summary"] = int(r["is_close_summary"])
            by_day.setdefault(r["date"], []).append(r)
    out, n_fb, seen_url = [], 0, set()
    for day in sorted(by_day):
        # 같은 URL 이 인접한 날짜 사이트맵에 중복 등장할 수 있어, 가장 이른 날짜에만 남긴다
        fresh = [r for r in by_day[day] if r["url"] not in seen_url] or by_day[day]   # 전부 겹치면 그날은 유지(커버리지 우선)
        seen_url.update(r["url"] for r in fresh)
        for rank, (r, fb) in enumerate(select_day(fresh), 1):
            n_fb += fb
            out.append({"year": day[:4], "month": int(day[5:7]), "date": day, "headline": r["headline"],
                        "source": r["source"], "section": r["section"], "url": r["url"],
                        "relevance_score": r["score"], "day_rank": rank, "is_fallback": fb})
    with pth["daily"].open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=DAILY_FIELDS)
        w.writeheader()
        w.writerows(out)
    print("[%s] 저장: %s  (%d일, %d행, fallback %d일)" % (market, pth["daily"], len({r["date"] for r in out}), len(out), n_fb))


# ---------------------------------------------------------------- 점검 / 검증
def run_smoke_test(markets):
    samples = [date(2014, 1, 1), date(2014, 3, 9), date(2018, 12, 25), date(2020, 3, 16), date(2023, 5, 7), date(2025, 12, 31)]
    ok = True
    for m in markets:
        print("=" * 66 + "\n  %s 스모크 테스트 (파일 미생성)" % m)
        throttle = Throttle(0.5)
        for d in samples:
            try:
                _, status, rows = crawl_day(m, d, throttle)
            except Exception as e:
                print("  %s  실패: %s" % (d, e)); ok = False; continue
            sel = select_day(rows)
            print("  %s  후보 %2d건 → 선택 %d건%s" % (d, len(rows), len(sel), "" if sel else "   <<< 0건"))
            for r, fb in sel[:2]:
                print("      [%2d]%s %s (%s)" % (r["score"], "F" if fb else " ", r["headline"][:95], r["section"]))
            ok &= bool(sel)
    print("\n스모크 테스트 %s" % ("통과" if ok else "실패"))
    return ok


def run_verify(markets):
    expected = [(START + timedelta(n)).isoformat() for n in range((END - START).days + 1)]
    lines = ["# 경제 뉴스 수집 검증 리포트", "",
             "기간 %s ~ %s (%d일). 생성: `collect_economic_news.py --verify`" % (START, END, len(expected)), ""]
    all_pass = True
    rng = random.Random(42)
    for m in markets:
        p = paths(m)["daily"]
        lines += ["## %s — `%s`" % (m, p.relative_to(BASE.parent).as_posix()), ""]
        if not p.exists():
            lines += ["**파일 없음**", ""]; all_pass = False; print("[%s] 파일 없음" % m); continue
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        dates = {r["date"] for r in rows}
        missing = [d for d in expected if d not in dates]
        out_range = [r for r in rows if not (START.isoformat() <= r["date"] <= END.isoformat())]
        keys = [(r["date"], norm(r["headline"])) for r in rows]
        dupes = len(keys) - len(set(keys))
        empty = sum(1 for r in rows if not r["headline"].strip())
        bad_ym = sum(1 for r in rows if r["year"] != r["date"][:4] or int(r["month"]) != int(r["date"][5:7]))
        close_leak = [r for r in rows if r["is_fallback"] == "0" and CLOSE_PATTERNS.search(r["headline"])]
        no_url = sum(1 for r in rows if not r["url"].startswith("http"))
        checks = [
            ("모든 날짜에 ≥1건 (누락 %d일)" % len(missing), not missing),
            ("범위 밖 날짜 행 %d" % len(out_range), not out_range),
            ("날짜+헤드라인 중복 %d" % dupes, dupes == 0),
            ("빈 헤드라인 %d" % empty, empty == 0),
            ("year/month 불일치 %d" % bad_ym, bad_ym == 0),
            ("장 마감 요약 패턴 행(선택본, fallback 제외) %d" % len(close_leak), not close_leak),
            ("URL 형식 오류 %d" % no_url, no_url == 0),
        ]
        all_pass &= all(ok for _, ok in checks)
        print("[%s] 행 %d, 고유 날짜 %d/%d" % (m, len(rows), len(dates), len(expected)))
        for name, ok in checks:
            print("    %s %s" % ("PASS" if ok else "FAIL", name))
        lines += ["- 총 행 %d, 고유 날짜 **%d / %d**, 일평균 %.2f건" % (len(rows), len(dates), len(expected), len(rows) / max(len(dates), 1)), ""]
        lines += ["| 검사 | 결과 |", "|---|---|"] + ["| %s | %s |" % (n, "PASS" if ok else "**FAIL**") for n, ok in checks] + [""]
        if missing:
            lines += ["누락 날짜 (최대 30개): " + ", ".join(missing[:30]), ""]
        # 연도별
        lines += ["| 연도 | 커버 일수 | 행 수 | 일평균 | fallback 일수 | fallback 비율 |", "|---|---|---|---|---|---|"]
        for y in range(START.year, END.year + 1):
            yr = [r for r in rows if r["date"].startswith(str(y))]
            yd = {r["date"] for r in yr}
            n_days = sum(1 for d in expected if d.startswith(str(y)))
            fb = sum(1 for r in yr if r["is_fallback"] == "1")
            lines.append("| %d | %d / %d | %d | %.2f | %d | %.1f%% |" % (y, len(yd), n_days, len(yr), len(yr) / max(len(yd), 1), fb, 100 * fb / max(len(yd), 1)))
        lines.append("")
        weak = sorted((r["date"], r["relevance_score"], r["section"], r["headline"]) for r in rows
                      if r["is_fallback"] == "1" and int(r["relevance_score"]) <= 1)
        lines += ["### 약한 날짜 (fallback 이면서 관련도 점수 ≤ 1) — %d일" % len(weak), "",
                  "해당 매체에 그날 경제 뉴스가 사실상 없어 그날 최고점 기사를 유지한 경우. 필요하면 학습에서 제외하거나 가중치를 낮출 것.", ""]
        lines += ["- %s [%s, score %s] %s" % (d, s, sc, h) for d, sc, s, h in weak[:60]]
        if len(weak) > 60:
            lines.append("- … 외 %d일" % (len(weak) - 60))
        lines.append("")
        print("    (참고) 약한 fallback 날짜 %d일" % len(weak))
        # 표본 20일
        sample_days = sorted(rng.sample(sorted(dates), min(20, len(dates))))
        lines += ["### 무작위 표본 20일 (날짜별 1순위 헤드라인)", ""]
        first = {}
        for r in rows:
            if r["day_rank"] == "1":
                first[r["date"]] = r
        for d in sample_days:
            r = first.get(d)
            if r:
                lines.append("- %s [%s%s] %s" % (d, r["section"], ", fallback" if r["is_fallback"] == "1" else "", r["headline"]))
        lines.append("")
    rep = BASE / "economic_news_collection_report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n리포트 저장: %s\n전체 검증 %s" % (rep, "통과" if all_pass else "실패"))
    return all_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--market", choices=["US", "UK"], help="한 시장만 (기본: 둘 다)")
    ap.add_argument("--start", default=START.isoformat())
    ap.add_argument("--end", default=END.isoformat())
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-interval", type=float, default=0.4, help="요청 간 최소 간격(초)")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--refetch-empty", action="store_true")
    a = ap.parse_args()
    markets = [a.market] if a.market else ["US", "UK"]
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)

    if a.smoke_test:
        sys.exit(0 if run_smoke_test(markets) else 1)
    if a.verify:
        sys.exit(0 if run_verify(markets) else 1)
    if not a.select_only:
        for m in markets:
            print("=" * 66 + "\n  %s 수집 시작 (%s ~ %s)" % (m, start, end), flush=True)
            if not run_collect(m, start, end, a.workers, a.min_interval, a.refetch_empty):
                sys.exit(2)
    for m in markets:
        run_select(m)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
