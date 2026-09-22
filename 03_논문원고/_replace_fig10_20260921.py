# -*- coding: utf-8 -*-
"""<그림 10> 이미지를 커버리지 2줄 비교본(fig10_v15.png)으로 교체 (2026-09-21).

기존 그림은 '실제 뉴스가 수집된 날' 눈금이 한 줄뿐이라, §5.1.1 재검정이 전제로 삼은
커버리지 보강을 보여 주지 못한다. 눈금을 기존/신규 2줄로 대비한 그림으로 바꾼다.
본문 해설(¶466)은 `_apply_reexperiment_20260921.py` 에서 이미 함께 수정하였다.

가로 폭은 그대로 두고 세로만 새 그림의 종횡비에 맞춘다.

실행: python _replace_fig10_20260921.py [--dry-run]
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
DOC = HERE / "(국문) 석사학위논문_임우현.docx"
NEW = (HERE.parent / "ETFpricepredictmodel" / "data" / "2014-2025"
       / "02_성공실험_변동성방향성예측" / "fig10_v15.png")
CAPTION = "<그림 10>"


def png_size(b: bytes):
    """PNG IHDR 에서 (width, height)."""
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "PNG 이 아님"
    w, h = struct.unpack(">II", b[16:24])
    return w, h


def main():
    dry = "--dry-run" in sys.argv
    if not NEW.exists():
        raise SystemExit("새 그림이 없다: %s  (먼저 _fig10_coverage_v15.py 실행)" % NEW)
    blob = NEW.read_bytes()
    nw, nh = png_size(blob)

    document = docx.Document(str(DOC))
    ps = document.paragraphs

    # <그림 10> 캡션 직전의 그림을 찾는다
    # 목차 줄(탭 + 쪽번호)은 제외하고 본문 캡션/해설 중 마지막 것을 쓴다
    cands = [i for i, p in enumerate(ps)
             if p.text.strip().startswith(CAPTION) and "	" not in p.text]
    if not cands:
        raise SystemExit("본문에서 %s 캡션을 찾지 못했다" % CAPTION)
    cap_i = cands[-1]
    # 이 원고는 (해설, 캡션, 이미지) 순서라 뒤쪽을 먼저 보고, 없으면 앞쪽을 본다
    target = None
    for i in list(range(cap_i, min(len(ps), cap_i + 6))) +              list(range(cap_i - 1, max(-1, cap_i - 6), -1)):
        blips = ps[i]._element.findall(".//" + qn("a:blip"))
        if blips:
            target = (i, blips[0])
            break
    if target is None:
        raise SystemExit("%s 부근에서 이미지를 찾지 못했다" % CAPTION)
    pi, blip = target
    rid = blip.get(qn("r:embed"))
    part = document.part.related_parts[rid]
    ow, oh = png_size(part.blob)
    print("대상: ¶%d  %s  %d bytes  %dx%d" % (pi, part.partname, len(part.blob), ow, oh))
    print("새것:      %s  %d bytes  %dx%d" % (NEW.name, len(blob), nw, nh))

    # 표시 크기: 가로 유지, 세로만 새 종횡비로
    shape = next(s for s in document.inline_shapes
                 if s._inline.graphic.graphicData.pic.blipFill.blip.get(qn("r:embed")) == rid)
    # 원고의 그림 10 은 가로 14cm 에 세로를 원본 종횡비보다 늘려 배치해 두었다.
    # 그 배치를 그대로 두고, 늘어난 플롯 영역(눈금 한 줄 추가)만큼만 세로를 키운다.
    cx, cy = shape.width, shape.height
    new_cy = int(round(cy * nh / oh))
    print("표시 크기: %.2f x %.2f cm -> %.2f x %.2f cm  (가로 유지, 세로 x%.3f)"
          % (cx.cm, cy.cm, cx.cm, new_cy / 360000.0, nh / oh))

    if dry:
        print("\n(--dry-run: 저장하지 않음)")
        return
    part._blob = blob
    shape.height = new_cy
    document.save(str(DOC))
    print("\n저장 완료: %s" % DOC.name)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
