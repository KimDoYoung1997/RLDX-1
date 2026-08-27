#!/usr/bin/env python3
"""Build the FFW-SH5 fine-tuning briefing deck.

Run with python-pptx on PYTHONPATH (it is not part of the pixi env, so it is
installed side-by-side to avoid touching an environment that may be running a
training job):

    PYTHONPATH=/NHNHOME/doyoung/.local/pylibs \
      .pixi/envs/rldx-sm100/bin/python docs/make_deck.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent.parent / "FFW-SH5-0826-finetune-briefing.pptx"

NAVY = RGBColor(0x1B, 0x3A, 0x5C)
TEAL = RGBColor(0x1F, 0x7A, 0x7A)
GREY = RGBColor(0x5A, 0x63, 0x6E)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xB4, 0x3A, 0x2E)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
AMBER = RGBColor(0xB4, 0x76, 0x0A)
INK = RGBColor(0x20, 0x26, 0x2E)

FONT = "맑은 고딕"
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.62)
BODY_TOP = Inches(1.52)
BODY_W = W - 2 * MARGIN


def _txt(frame, runs, size=14, color=INK, bold=False, space_after=6, align=PP_ALIGN.LEFT,
         font=FONT, first=False):
    """Append a paragraph made of (text, {overrides}) runs."""
    p = frame.paragraphs[0] if first else frame.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, over in runs:
        r = p.add_run()
        r.text = text
        f = r.font
        f.name = over.get("font", font)
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.color.rgb = over.get("color", color)
    return p


def _box(slide, x, y, w, h, fill=None, line=None, line_w=1.0):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.04
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def slide_base(prs, title, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = slide_shapes_rect(s, 0, 0, W, Inches(1.16), NAVY)
    tf = bar.text_frame
    tf.margin_left, tf.margin_top = Inches(0.62), Inches(0.16)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _txt(tf, [(title, {"size": 25, "bold": True, "color": WHITE})], first=True)
    if kicker:
        _txt(tf, [(kicker, {"size": 12.5, "color": RGBColor(0xB8, 0xCE, 0xE0)})], space_after=0)
    return s


def slide_shapes_rect(slide, x, y, w, h, fill):
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def body_frame(slide, x=None, y=None, w=None, h=None):
    x = MARGIN if x is None else x
    y = BODY_TOP if y is None else y
    w = BODY_W if w is None else w
    h = (H - y - Inches(0.5)) if h is None else h
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def bullets(tf, items, size=15, first=True):
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            text, over = it
        else:
            text, over = it, {}
        lvl = over.pop("level", 0)
        p = _txt(tf, [("• " if lvl == 0 else "– ", {"color": TEAL, "bold": True}),
                      (text, over)],
                 size=size, space_after=over.pop("space_after", 9), first=(first and i == 0))
        p.level = lvl
    return tf


def table(slide, x, y, w, rows, col_w=None, header=True, size=12.5, row_h=Inches(0.34)):
    nrow, ncol = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrow, ncol, x, y, w, row_h * nrow)
    t = shp.table
    if col_w:
        total = sum(col_w)
        for i, c in enumerate(col_w):
            t.columns[i].width = Emu(int(w * c / total))
    for r, row in enumerate(rows):
        t.rows[r].height = row_h
        for c, cell in enumerate(row):
            txt, over = (cell, {}) if isinstance(cell, str) else cell
            cl = t.cell(r, c)
            cl.margin_left = cl.margin_right = Inches(0.08)
            cl.margin_top = cl.margin_bottom = Inches(0.02)
            cl.vertical_anchor = MSO_ANCHOR.MIDDLE
            cl.fill.solid()
            if header and r == 0:
                cl.fill.fore_color.rgb = NAVY
            else:
                cl.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            tf = cl.text_frame
            tf.word_wrap = True
            _txt(tf, [(txt, over)],
                 size=over.get("size", size),
                 color=WHITE if (header and r == 0) else over.get("color", INK),
                 bold=True if (header and r == 0) else over.get("bold", False),
                 space_after=0, align=over.get("align", PP_ALIGN.LEFT), first=True)
    return t


def callout(slide, x, y, w, h, title, body, accent=TEAL, tsize=14, bsize=13):
    _box(slide, x, y, w, h, fill=LIGHT, line=accent, line_w=1.5)
    slide_shapes_rect(slide, x, y, Inches(0.07), h, accent)
    tf = body_frame(slide, x + Inches(0.22), y + Inches(0.10), w - Inches(0.4), h - Inches(0.2))
    _txt(tf, [(title, {"size": tsize, "bold": True, "color": accent})], first=True, space_after=4)
    for line in body:
        _txt(tf, [(line, {"size": bsize})], space_after=3)


def footer(slide, text, n):
    tb = slide.shapes.add_textbox(MARGIN, H - Inches(0.46), BODY_W, Inches(0.3))
    tf = tb.text_frame
    _txt(tf, [(text, {"size": 10, "color": GREY}),
              (f"          {n}", {"size": 10, "color": GREY})], first=True, space_after=0)


# ---------------------------------------------------------------------------

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
n = 0


def nxt():
    global n
    n += 1
    return n


# 1 — title -----------------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6])
slide_shapes_rect(s, 0, 0, W, H, NAVY)
slide_shapes_rect(s, 0, H - Inches(0.5), W, Inches(0.5), TEAL)
tf = body_frame(s, MARGIN, Inches(2.25), BODY_W, Inches(3.4))
_txt(tf, [("RLDX-1 파인튜닝 — FFW-SH5", {"size": 40, "bold": True, "color": WHITE})],
     first=True, space_after=10)
_txt(tf, [("rlwrld_demo_0826_merge · 81 에피소드 / 33,018 프레임 / 10 fps",
           {"size": 19, "color": RGBColor(0x9F, 0xC4, 0xDA)})], space_after=26)
_txt(tf, [("데이터 특성에서 출발한 설계 결정과 학습 구성", {"size": 16, "color": RGBColor(0xC8, 0xDA, 0xE8)})],
     space_after=6)
_txt(tf, [("2026-08-27 · 4× NVIDIA B200", {"size": 13.5, "color": RGBColor(0x88, 0xA8, 0xC0)})])

# 2 — 요약 --------------------------------------------------------------------
s = slide_base(prs, "한 장 요약", "결정 3개와 현재 상태")
rows = [
    ["", "결정", "근거 (데이터/논문)"],
    [("① action", {"bold": True}), "57 → 26차원", "명령이 실제로 들어가는 관절만. 54 중 28개가 상수 0"],
    [("② tactile", {"bold": True}), "쓰지 않음", "PT에 physics stream이 없고, 2지 파지라 신호가 36차원에만 희소"],
    [("③ chunk", {"bold": True}), "16 → 40 (실행 20)", "10 fps에서 2.0초 실행. 논문 ALLEX의 20/40 비율과 동일"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[1.1, 2.0, 5.2], row_h=Inches(0.46))

callout(s, MARGIN, Inches(3.55), Inches(6.1), Inches(1.55), "우리 task는 composite다",
        ["4단계 + 조건 분기(“송장이 위로 오게”). atomic이 아니다.",
         "RoboCasa365에서 RLDX-1도 atomic 67.3% vs composite 19.0%.",
         "→ 1차는 subtask 없이 전체 문장 하나로 end-to-end."], accent=AMBER)
callout(s, MARGIN + Inches(6.4), Inches(3.55), Inches(5.67), Inches(1.55), "지금 학습 중",
        ["batch 128 · lr 1e-4 cosine · 30K step · 4×B200",
         "GPU당 55 GiB, 예상 6~8시간",
         "held-out 9 에피소드로 체크포인트 선택 예정"], accent=GREEN)
callout(s, MARGIN, Inches(5.35), BODY_W, Inches(1.25), "핵심 제약",
        ["RLDX-1-PT는 1단계(사전학습) 산출물이라 motion / memory / physics 텐서가 0개다.",
         "그 기능들은 2단계 mid-training(2,560만 샘플)에서 붙는 것 → 우리 3.2만 샘플로 대체 불가. 세 모듈 모두 OFF."],
        accent=RED)
footer(s, "요약", nxt())

# 3 — 데이터 -------------------------------------------------------------------
s = slide_base(prs, "데이터 — 무엇을 합쳤나", "3개 세션 병합 + 전수 감사")
rows = [
    ["세션", "에피소드", "프레임", "평균 길이", "비중"],
    ["rlwrld_demo_0826_1539", "21", "6,710", "320", "20.3%"],
    ["rlwrld_demo_0826_1652", "30", "11,408", "380", "34.6%"],
    ["rlwrld_demo_0826_1748", "30", "14,900", "497", "45.1%"],
    [("합계", {"bold": True}), ("81", {"bold": True}), ("33,018", {"bold": True}),
     ("378 (중앙값)", {"bold": True}), ("55분", {"bold": True})],
]
table(s, MARGIN, BODY_TOP, Inches(7.4), rows, col_w=[2.6, 1.0, 1.1, 1.2, 0.9])

tf = body_frame(s, MARGIN + Inches(7.75), BODY_TOP - Inches(0.05), Inches(4.95), Inches(2.4))
_txt(tf, [("전수 감사 결과 — 0건", {"size": 15, "bold": True, "color": GREEN})], first=True, space_after=7)
bullets(tf, [
    ("dead episode · NaN/Inf · 영상 desync", {"size": 13}),
    ("인덱스 오류 · 중복 에피소드", {"size": 13}),
    ("task 문자열 플레이스홀더 (0812의 “1611” 문제 없음)", {"size": 13}),
], first=False)

callout(s, MARGIN, Inches(3.55), Inches(6.1), Inches(1.5), "병합에서 실제로 잡은 문제",
        ["pandas가 fixed_size_list<float>[57] → list<float>로 강등하고",
         "huggingface 스키마 메타데이터를 삭제 → Arrow 레벨로 재작성해 해결.",
         "결과: payload 비트 동일, mp4 바이트 동일, 스키마 보존."], accent=TEAL)
callout(s, MARGIN + Inches(6.4), Inches(3.55), Inches(5.67), Inches(1.5), "알고 있어야 할 편향",
        ["세션이 갈수록 시연이 길어진다 (320 → 380 → 497 프레임).",
         "1748 세션이 학습 분포의 45%를 차지.",
         "→ 평가는 반드시 세 세션에 걸쳐서."], accent=AMBER)
callout(s, MARGIN, Inches(5.3), BODY_W, Inches(1.1), "학습/검증 분할",
        ["train 72 ep / 29,661 frames   ·   held-out 9 ep / 3,357 frames (10.2%, 세 세션에서 3개씩)",
         "트레이너에 validation 경로가 없다 (eval_dataset은 항상 None) → held-out + open_loop_eval로 직접 봐야 한다."],
        accent=NAVY)
footer(s, "데이터", nxt())

# 4 — action 26 ---------------------------------------------------------------
s = slide_base(prs, "결정 ① — action을 57에서 26차원으로", "데이터가 말해준 실제 제어 공간")
rows = [
    ["관절", "state", "action", "판정"],
    ["arm_l / arm_r  joint1–7 (14)", "움직임", "움직임", ("제어함", {"color": GREEN, "bold": True})],
    ["finger_*_joint1–13 (26)", "움직임 ±0.02~0.3", ("0.000 고정", {"bold": True}), ("명령 없음", {"color": RED})],
    ["finger_*_joint14,15,16 (6)", "움직임", "움직임 0~1.29", ("제어함", {"color": GREEN, "bold": True})],
    ["finger_*_joint17 (2)", "움직임", ("0.000 고정", {"bold": True}), ("명령 없음", {"color": RED})],
    ["finger_*_joint18,19,20 (6)", "움직임", "움직임", ("제어함", {"color": GREEN, "bold": True})],
]
table(s, MARGIN, BODY_TOP, Inches(7.5), rows, col_w=[2.7, 1.5, 1.5, 1.1])

tf = body_frame(s, MARGIN + Inches(7.85), BODY_TOP - Inches(0.05), Inches(4.85), Inches(2.8))
_txt(tf, [("33,018 프레임 전수 측정", {"size": 15, "bold": True, "color": NAVY})], first=True, space_after=7)
bullets(tf, [
    ("action 54차원 중 28개가 상수 0", {"size": 13.5, "bold": True}),
    ("실질 제어 = 팔 14 + 손 12 = 26차원", {"size": 13.5}),
    ("조이스틱 프리셋 → 4·5번 손가락만 사용 (의도됨)", {"size": 13.5}),
    ("head/lift(54:57)도 명령 없음 → 이미 제외", {"size": 13.5}),
], first=False)

callout(s, MARGIN, Inches(4.05), Inches(6.1), Inches(1.35), "54로 둬도 “틀리지는” 않는다",
        ["q99 정규화가 상수 차원을 0으로 보내고, flow-matching 타깃 −ε는",
         "노이즈 입력에서 정확히 복원 가능 → 학습은 된다.",
         "다만 디코더 폭과 그래디언트를 신호 없는 28차원에 쓴다."], accent=GREY)
callout(s, MARGIN + Inches(6.4), Inches(4.05), Inches(5.67), Inches(1.35), "검증 완료",
        ["action 26차원, dead dim 0개 (모든 그룹 q99−q01 > 0.07)",
         "state는 54 유지 — 관측은 공짜, 접촉 정보를 담을 수 있음",
         "stats.json 재생성 불필요"], accent=GREEN, bsize=12.5)
callout(s, MARGIN, Inches(5.6), BODY_W, Inches(0.95), "⚠ 배포 시 바뀌는 것",
        ["서버 출력 키가 action.left_hand(20) → action.left_hand_j14_16 / _j18_20 (3+3)로 바뀐다.",
         "클라이언트가 26 → 57 매핑을 새 그룹 순서로 다시 조립해야 한다. 아직 실기로 확인 안 됨."],
        accent=AMBER)
footer(s, "결정 ①", nxt())

# 5 — tactile -----------------------------------------------------------------
s = slide_base(prs, "결정 ② — fingertip(tactile)을 쓰지 않는 이유", "설계가 그렇고, 모델도 아직 못 받는다")
rows = [
    ["층위", "무엇이 문제인가", "수치"],
    [("모델", {"bold": True}), "RLDX-1-PT에 physics stream이 존재하지 않음",
     "physics 텐서 0개 (MT는 21개)"],
    [("데이터", {"bold": True}), "90차원 중 대부분이 정규화하면 정보량 0",
     "q01=q99인 차원 79/90"],
    [("운용", {"bold": True}), "2지 파지 설계 — 접촉이 4·5번 손가락에 집중",
     "0이 아닌 (프레임×차원) 비율 0.69%"],
    [("코드", {"bold": True}), "all-zero 샘플은 physics를 통째로 스킵",
     "프레임의 61.6%가 전 차원 0"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[1.0, 4.6, 2.4], row_h=Inches(0.44))

tf = body_frame(s, MARGIN, Inches(3.75), Inches(6.15), Inches(1.9))
_txt(tf, [("접촉이 집중되는 곳 = 손가락 4·5번", {"size": 15, "bold": True, "color": NAVY})], first=True, space_after=7)
bullets(tf, [
    ("접촉 이벤트: 4·5번 2,580건 vs 1~3번 115건", {"size": 13}),
    ("1~3번도 스치면 최대 70~233까지 제대로 읽는다", {"size": 13}),
    ("→ 하드웨어 결함이 아니다. 쓰게 되면 36차원만 넣으면 됨", {"size": 13}),
], first=False)

callout(s, MARGIN + Inches(6.45), Inches(3.75), Inches(5.62), Inches(1.9), "결정 ①과 같은 원인이다",
        ["조이스틱 버튼 하나로 프리셋이 걸려",
         "4·5번 손가락(약지·새끼)만으로 상자를 파지한다.",
         "→ 명령도 접촉도 거기 몰리는 게 당연하다.",
         "   데이터를 다시 모아도 같은 모양이 나온다."], accent=TEAL)
callout(s, MARGIN, Inches(5.85), BODY_W, Inches(1.0), "그래도 tactile이 필요한 순간은 따로 있다",
        ["논문에서 tactile이 효과를 본 건 Plug Insertion(20.8→33.3), Egg PnP(45.8→61.1) — 시각으로 접촉을 알 수 없는 정밀 작업.",
         "상자 운반은 그 범주가 아니다. 나중에 쓰게 되면 파지에 실제로 쓰이는 36차원(4·5번 손가락)만 넣으면 된다."],
        accent=AMBER)
footer(s, "결정 ②", nxt())

# 6 — 모델 위치 ----------------------------------------------------------------
s = slide_base(prs, "RLDX-1의 3단계 학습과 우리 위치", "왜 기능 모듈을 못 쓰는가")
rows = [
    ["단계", "steps × batch", "하드웨어", "무엇을 하는가", "우리"],
    ["Pre-training → RLDX-1-PT", "100K × 8192", "64×H200 195h", "1.5M 에피소드 action prior",
     ("출발점", {"color": GREEN, "bold": True})],
    ["Mid-training → RLDX-1-MT-*", "25K × 1024", "64×H200 15h",
     ("motion / memory / physics 주입", {"bold": True}), ("건너뜀", {"color": RED, "bold": True})],
    ["Post-training → RLDX-1-FT-*", "30K × 64~128", "—", "task별 적응",
     ("우리가 함", {"color": GREEN, "bold": True})],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[2.5, 1.4, 1.5, 3.0, 1.0], row_h=Inches(0.46))

rows2 = [
    ["체크포인트 텐서 조사", "PT", "MT-ALLEX", "MT-DROID"],
    ["motion(STSS)", ("0", {"color": RED, "bold": True}), "28", "28"],
    ["memory", ("0", {"color": RED, "bold": True}), "19", "19"],
    ["physics", ("0", {"color": RED, "bold": True}), "21", "21"],
    ["double / single blocks", "80 / 64", "120 / 128", "120 / 128"],
]
table(s, MARGIN, Inches(3.65), Inches(6.3), rows2, col_w=[2.4, 1.0, 1.3, 1.3], row_h=Inches(0.33))

callout(s, MARGIN + Inches(6.6), Inches(3.65), Inches(5.47), Inches(2.25), "그래서 세 모듈 모두 OFF",
        ["PT 위에서 --use-physics를 켜면 MSAT의 P-stream QKV까지",
         "전부 무작위 초기화된다.",
         "논문은 같은 일을 2,560만 샘플로 했고",
         "우리에겐 3.2만 샘플뿐 — 800배 차이.",
         "MT를 base로 쓰는 것도 답이 아니다. mid-training의 목적이",
         "“embodiment specialization”이라 ALLEX/FR3에 특화돼 있다."],
        accent=RED, bsize=12.5)
callout(s, MARGIN, Inches(5.75), Inches(6.3), Inches(1.1), "PT 모델 카드가 직접 말한다",
        ["“Use it as your starting point for new embodiments and tasks.”",
         "우리는 새 임바디먼트다 → RLDX-1-PT가 맞다."], accent=TEAL)
footer(s, "모델", nxt())

# 7 — 하이퍼파라미터 근거 --------------------------------------------------------
s = slide_base(prs, "하이퍼파라미터 — 논문 실로봇 레시피를 따른다", "§G.2~G.5")
rows = [
    ["플랫폼", "데모 수", "steps", "batch", "lr", "state dropout", "chunk"],
    ["OpenArm 휴머노이드", "348 (4 task)", "30K", "64", "1e-4 cosine", "0.3", "16"],
    ["ALLEX (task별)", "40 / 90 / 72 / 62", "30K", "128", "1e-4 cosine", "0.8", "40"],
    ["FR3 (task별)", "50~100", "30K", "64", "1e-4 cosine", "0.3", "16"],
    [("우리 (FFW-SH5)", {"bold": True}), ("81", {"bold": True}), ("30K", {"bold": True}),
     ("128", {"bold": True}), ("1e-4 cosine", {"bold": True}), ("0.3", {"bold": True}),
     ("40", {"bold": True})],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[2.2, 1.7, 0.8, 0.8, 1.4, 1.3, 0.8], row_h=Inches(0.40))

callout(s, MARGIN, Inches(3.75), Inches(6.1), Inches(1.75), "우리 81개는 논문 범위 안에 있다",
        ["실로봇 task당 데모 40~100개가 논문의 일관된 규모.",
         "프레임 수도 ALLEX Object-in-Box(90 ep ≈ 3.1만)와 비슷한 3.3만.",
         "임바디먼트가 가장 가까운 OpenArm이 chunk 16, ALLEX가 40."], accent=TEAL)
callout(s, MARGIN + Inches(6.4), Inches(3.75), Inches(5.67), Inches(1.75), "backbone을 얼리면 안 된다 (Table 6)",
        ["Full FT(top-4) + Full FT  →  62.67%",
         "Frozen + LoRA r=64        →  36.42%   (−26점)",
         "→ --tune-top-llm-layers 4 유지.",
         "   LoRA는 VRAM 절감 카드인데 B200에는 필요 없다."], accent=NAVY)
callout(s, MARGIN, Inches(5.7), BODY_W, Inches(1.1), "우리 스모크에서 나온 확인",
        ["학습 파라미터 2,375,979,600 = 2,376.0M — 논문 Table 6의 “Full FT(top-4) + Full FT action model = 2,376.0M”과 정확히 일치.",
         "동결 설정이 논문의 full-fine-tuning 구성과 같다는 독립적 검증."], accent=GREEN)
footer(s, "하이퍼파라미터 (1/2)", nxt())

# 8 — 실측 ---------------------------------------------------------------------
s = slide_base(prs, "하이퍼파라미터 — 4×B200 실측", "추측이 아니라 재본 값")
rows = [
    ["global batch", "image_max_area", "vision token", "처리량", "peak VRAM/GPU", "30K step"],
    ["64", "65536", "40", "3.15 it/s (202 sample/s)", "37.4 GiB", "2.6 h"],
    [("128", {"bold": True}), "65536", "40", "2.06 it/s (264 sample/s)", "47.4 GiB", "4.0 h"],
    [("128", {"bold": True}), ("98304", {"bold": True}), ("84", {"bold": True}),
     "1.60 it/s (205 sample/s)", ("54.3 GiB", {"bold": True}), "5.2 h"],
    ["256", "65536", "40", "1.33 it/s (340 sample/s)", "66.9 GiB", "6.3 h"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[1.3, 1.5, 1.2, 2.4, 1.5, 1.0], row_h=Inches(0.42))

callout(s, MARGIN, Inches(3.55), Inches(6.1), Inches(1.6), "VRAM은 병목이 아니다",
        ["가장 무거운 구성이 67 GiB / 183 GB. batch 512도 물리적으론 들어간다.",
         "논문 Table 6의 H200 기준(batch 32에 87 GiB)보다 가벼운 이유는",
         "우리 카메라가 1대이기 때문 (논문 ablation은 3 view × 12프레임).",
         "→ LoRA를 쓸 이유가 없다."], accent=TEAL)
callout(s, MARGIN + Inches(6.4), Inches(3.55), Inches(5.67), Inches(1.6), "배치가 클수록 오히려 빠르다",
        ["샘플 처리량 202 → 264 → 340 sample/s.",
         "작은 배치는 B200을 못 채운다.",
         "batch 256을 안 고른 이유는 속도·메모리가 아니라",
         "30K에서 269 epoch가 되어 과적합이 우려되기 때문."], accent=NAVY)
callout(s, MARGIN, Inches(5.35), BODY_W, Inches(1.25), "⚠ 처리량은 실행마다 ±25% 흔들린다",
        ["같은 구성을 네 번 재보니 1.07 ~ 1.60 it/s. /NHNHOME이 공유 Lustre라 mp4 디코딩이 캐시·타 사용자 부하를 탄다.",
         "→ 계획용 수치는 “30K에 5~8시간”. 5.2h는 가장 좋았을 때 값이지 기대값이 아니다."], accent=AMBER)
footer(s, "하이퍼파라미터 (2/2)", nxt())

# 9 — 이미지 -------------------------------------------------------------------
s = slide_base(prs, "이미지 전처리 — 기본값이 우리 종횡비에서 최악이었다", "cam_head 376×672")
rows = [
    ["--image-max-area", "resize", "crop", "vision token/frame", "가로 손실"],
    [("65536 (기본값)", {"color": RED}), "160×285", ("160×256", {"bold": True}),
     ("40", {"color": RED, "bold": True}), ("10.2%", {"color": RED, "bold": True})],
    ["67584", "192×343", "192×320", "60", "6.7%"],
    [("98304 (채택)", {"color": GREEN, "bold": True}), "224×400", ("224×384", {"bold": True}),
     ("84", {"color": GREEN, "bold": True}), ("4.0%", {"color": GREEN, "bold": True})],
    ["131072", "256×457", "256×448", "112", "2.0%"],
]
table(s, MARGIN, BODY_TOP, Inches(7.2), rows, col_w=[2.0, 1.2, 1.2, 1.7, 1.2])

tf = body_frame(s, MARGIN + Inches(7.55), BODY_TOP - Inches(0.05), Inches(5.15), Inches(2.6))
_txt(tf, [("왜 크롭이 생기나", {"size": 15, "bold": True, "color": NAVY})], first=True, space_after=7)
bullets(tf, [
    ("논문 §B.1은 Qwen3-VL smart_resize를 쓴다고 하지만,", {"size": 13}),
    ("레포 구현은 짧은 변을 맞춘 뒤 긴 변을 잘라낸다.", {"size": 13, "bold": True}),
    ("정사각형 입력에선 무해하지만 가로로 긴 우리 카메라에선 실제로 잘린다.", {"size": 13}),
], first=False)

callout(s, MARGIN, Inches(3.9), BODY_W, Inches(1.5), "화질보다 중요한 이유 — 분기 판단이 이 화면에 달려 있다",
        ["이 작업의 분기(“송장이 위로 오게 방향 맞추기”)를 결정하는 건 상자 위의 흰 송장 라벨이다.",
         "224×384에서도 라벨이 뚜렷한 밝은 사각형으로 남는 것을 프레임 추출로 직접 확인했다 — 글자는 못 읽지만 읽을 필요가 없다.",
         "65536에서는 라벨 덩어리가 뭉개지기 시작하고 가로 화각도 더 잘린다."], accent=TEAL)
callout(s, MARGIN, Inches(5.55), BODY_W, Inches(1.0), "비용",
        ["처리량 −22%, VRAM +7 GiB (2.06 → 1.60 it/s, 47.4 → 54.3 GiB). 30K 기준 4.0h → 5.2h.",
         "B200 VRAM은 여전히 30%도 안 쓴다 → 살 만한 값."], accent=GREY)
footer(s, "전처리", nxt())

# 10 — chunk ------------------------------------------------------------------
s = slide_base(prs, "chunk와 execution horizon — 10 fps에서의 의미", "결정 ③")
callout(s, MARGIN, BODY_TOP, BODY_W, Inches(0.95), "먼저: 코드는 fps에 무관하다",
        ["self.fps는 info.json에서 읽히지만 데이터 파이프라인 어디서도 쓰이지 않는다. delta_indices·video_stride는 전부 프레임 단위.",
         "→ fps는 학습 동작을 바꾸지 않고, 설정값의 “초 단위 의미”만 정한다. 논문은 자체 실로봇 fps를 밝히지 않는다."],
        accent=NAVY)
rows = [
    ["설정", "프레임", "10 fps 기준 초"],
    ["video context {−6,−4,−2,0}", "6", "0.6 s 시각 이력"],
    [("action chunk (채택)", {"bold": True}), ("40", {"bold": True}), ("4.0 s 예측", {"bold": True})],
    [("실행 (execution horizon)", {"bold": True}), ("20", {"bold": True}), ("2.0 s 실행", {"bold": True})],
    ["에피소드 길이", "215~916", "21.5 ~ 91.6 s"],
]
table(s, MARGIN, Inches(2.75), Inches(6.0), rows, col_w=[2.8, 1.0, 2.2], row_h=Inches(0.36))

rows2 = [
    ["논문의 실행 비율", "chunk", "실행", "비율"],
    ["SIMPLER", "16", "2", "1/8"],
    ["LIBERO · GR-1 Tabletop", "16", "8", "1/2"],
    ["RoboCasa Kitchen · 365", "16", "16", "1/1"],
    [("ALLEX (실로봇)", {"bold": True}), ("40", {"bold": True}), ("20", {"bold": True}),
     ("1/2", {"bold": True})],
]
table(s, MARGIN + Inches(6.35), Inches(2.75), Inches(5.7), rows2, col_w=[2.7, 1.0, 1.0, 1.0],
      row_h=Inches(0.36))

callout(s, MARGIN, Inches(4.85), Inches(6.0), Inches(1.75), "왜 50이 아니라 40인가",
        ["목표(2.0초 실행)는 chunk가 아니라 실행 개수가 정한다 → 40도 동일.",
         "40은 논문이 실제로 학습시켜 본 길이(ALLEX)이고,",
         "비율도 20/40 = 0.50으로 일치한다.",
         "50도 실측상 돈다(VRAM +1.6 GiB). 다만 근거가 없고 샘플이 8.6% 준다."],
        accent=TEAL, bsize=12.5)
callout(s, MARGIN + Inches(6.35), Inches(4.85), Inches(5.7), Inches(1.75), "실행 개수는 학습과 무관하다",
        ["학습은 늘 전체 chunk를 예측하도록 한다.",
         "몇 개 실행할지는 배포 시 클라이언트 결정이고",
         "체크포인트에 기록되지 않는다.",
         "→ 같은 체크포인트로 20/10을 바꿔 시험 가능. 재학습 불필요.",
         "(예외: memory_stride, rtc_training_max_delay — 둘 다 미사용)"],
        accent=GREEN, bsize=12.5)
footer(s, "결정 ③", nxt())

# 11 — composite --------------------------------------------------------------
s = slide_base(prs, "우리 task는 atomic인가 composite인가", "— composite다")
tf = body_frame(s, MARGIN, BODY_TOP - Inches(0.05), BODY_W, Inches(0.9))
_txt(tf, [("“Receive the package from the conveyor, ", {"size": 15}),
          ("check the shipping label", {"size": 15, "bold": True, "color": AMBER}),
          (", ", {"size": 15}),
          ("flip the box face up if necessary", {"size": 15, "bold": True, "color": RED}),
          (", and forward it to the next station.”", {"size": 15})], first=True)
_txt(tf, [("→ 4단계 + 조건 분기. 논문 기준 composite task다.", {"size": 14, "color": GREY})])

rows = [
    ["RoboCasa365 · Table 1b", "Atomic", "Composite-S", "Composite-U"],
    [("RLDX-1", {"bold": True}), ("67.3", {"bold": True, "color": GREEN}),
     ("19.0", {"bold": True, "color": AMBER}), ("5.6", {"bold": True, "color": RED})],
    ["GR00T N1.6", "61.1", "12.6", "2.6"],
    ["π₀", "34.6", "6.1", "1.1"],
]
table(s, MARGIN, Inches(2.85), Inches(6.1), rows, col_w=[2.3, 1.2, 1.3, 1.3], row_h=Inches(0.40))

callout(s, MARGIN + Inches(6.45), Inches(2.85), Inches(5.62), Inches(1.9), "왜 위험한가 — 세 가지",
        ["① 오차 누적: 단계별 0.9여도 4단계면 0.66",
         "② 중간 상태 기억: “라벨을 이미 봤는가”가 현재 프레임에 없다",
         "    (논문은 memory module로 푸는데 우리는 못 쓴다)",
         "③ 같은 관측에 정답이 여러 개 → 평균 내면 둘 다 아닌 행동"], accent=AMBER)

callout(s, MARGIN, Inches(4.95), BODY_W, Inches(1.75), "그런데 ③은 우리에게 완화된다 — 분기 신호가 화면에 있다",
        ["ep5: 라벨이 앞쪽 수직면    ep30: 라벨 안 보임(16.0초까지) → 18.0초에 나타남    ep60: 이미 윗면 → 뒤집기 불필요",
         "흰 송장 라벨은 갈색 상자 위의 큰 고대비 사각형이라 376×672 → 224×384로 줄여도 뚜렷하게 남는다.",
         "→ 관측이 서로 다르므로 정책이 원리적으로 분기를 학습할 수 있다. “입력에 정보가 없어서 못 배우는” 상황은 아니다.",
         "남는 위험은 오차 누적과 진행상태 기억이지 관측 부족이 아니다."], accent=GREEN)
footer(s, "task 성격", nxt())

# 12 — subtask ----------------------------------------------------------------
s = slide_base(prs, "subtask로 나눌 것인가 — 이번엔 아니다", "선택지 A / B / C")
rows = [
    ["", "방식", "태스크 성질", "필요한 것", "배포"],
    [("A", {"bold": True, "color": GREEN}), ("전체 문장 1개로 end-to-end", {"bold": True}),
     "composite", ("없음 (지금 그대로)", {"bold": True, "color": GREEN}), "지시문 1개"],
    [("B", {"bold": True}), "sub_task 조건화 단일 정책", "composite",
     "구간 라벨 + 런타임 플래너", "매 시점 subtask 지정"],
    [("C", {"bold": True}), "구간별 정책 분리 + 오케스트레이터", "atomic으로 분해",
     "라벨 + 정책 N개 + 전환 로직", "시스템 재설계"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[0.5, 3.2, 1.4, 3.0, 2.0], row_h=Inches(0.46))

callout(s, MARGIN, Inches(3.55), Inches(6.1), Inches(1.9), "A를 고른 이유",
        ["① 추가 라벨링이 0이다 (B는 81개 영상을 사람이 훑어야 한다)",
         "② 1차 결과 없이는 어디를 쪼개야 할지 모른다",
         "③ B는 배포 계약을 바꾼다 — 실행 중 “지금 어느 subtask인가”를",
         "    정해줄 상위 플래너가 필요해진다",
         "④ 되돌리기 쉽다 — 나중에 해도 지금 학습이 낭비가 안 된다"], accent=GREEN)
callout(s, MARGIN + Inches(6.4), Inches(3.55), Inches(5.67), Inches(1.9), "B는 이미 지원된다 (필요해지면)",
        ["코드에 LANG_KEYS = [\"task\", \"sub_task\"]가 있다.",
         "episodes.jsonl에 sub_tasks 구간을 넣고 config 한 줄만 바꾸면 된다.",
         "라벨은 start−8 프레임부터 적용되고 겹치는 구간은 확률적으로 섞인다.",
         "atomic으로 바뀌는 게 아니라 언어 신호만 잘게 쪼개지는 것."], accent=NAVY)
callout(s, MARGIN, Inches(5.65), BODY_W, Inches(1.0), "1차 실패를 볼 때 이것만 보면 된다",
        ["실패가 “전 구간 어설픔”인가, “특정 전환 지점에서 멈추거나 잘못된 분기를 타는가”.",
         "후자면 그때 subtask를 도입한다. 그때도 전수 라벨링 말고 “초기 송장 위치(front/top/hidden/side)”부터 센다."],
        accent=AMBER)
footer(s, "subtask", nxt())

# 13 — 이번 학습 ---------------------------------------------------------------
s = slide_base(prs, "이번에 거는 학습", "run_scripts/train/examples/finetune_rldx1_ffw_sh5_0826.sh")
rows = [
    ["항목", "값", "근거"],
    ["base model", "RLWRLD/RLDX-1-PT", "1단계 산출물, 새 임바디먼트의 출발점"],
    ["data", "dataset/rlwrld_demo_0826_train (72 ep)", "held-out 9 ep 제외"],
    ["modality", "ffw_sh5_rev1_config_used26_h40.py", "action 26차원, chunk 40"],
    ["batch / steps", "128 × 30,000", "논문 ALLEX·FR3 실로봇 레시피"],
    ["lr / schedule", "1e-4 / cosine + warmup 5%", "논문 post-training 전부 동일"],
    ["state dropout", "0.3", "OpenArm·FR3 값"],
    ["동결", "vision encoder + LLM (top-4 제외)", "Table 6 — 완전 동결은 −26점"],
    ["image", "--image-max-area 98304 (224×384, 84 token)", "가로 화각 확보"],
    ["모듈", "motion / memory / physics 전부 OFF", "PT에 텐서가 없음"],
    ["저장", "10,000 step마다 · limit 5 (183 GB)", "삭제 없이 10K/20K/30K 보존"],
]
table(s, MARGIN, BODY_TOP, Inches(8.6), rows, col_w=[1.5, 3.5, 3.6], row_h=Inches(0.345), size=12)

callout(s, MARGIN + Inches(8.9), BODY_TOP - Inches(0.05), Inches(3.8), Inches(2.3), "예상",
        ["30K step에 6~8시간",
         "GPU당 55 GiB / 183 GB",
         "체크포인트 1개 = 61 GB",
         "저장 1회당 약 40초"], accent=NAVY)
callout(s, MARGIN + Inches(8.9), Inches(4.0), Inches(3.8), Inches(2.5), "학습 후 바로",
        ["10K / 20K / 30K 각각",
         "held-out 9 에피소드로",
         "open_loop_eval MSE 기록",
         "→ 최저점을 배포 후보로",
         "",
         "--traj-ids 0..8 (재번호됨)",
         "--action-horizon 40"], accent=GREEN, bsize=12.5)
footer(s, "이번 학습", nxt())

# 14 — pre-flight -------------------------------------------------------------
s = slide_base(prs, "학습 직전 점검에서 잡은 것", "레포가 문서보다 조용히 다르게 동작하는 지점들")
rows = [
    ["발견", "무엇이 문제였나", "조치"],
    [("save-total-limit 조합", {"bold": True, "color": RED}),
     "2500 저장 × limit 6 → 잔존이 17.5K~30K. 최적 예상 구간(10K~20K)이 전부 삭제됨",
     "10000 / 5로 변경 (삭제 없음)"],
    [("정규화 통계 누수", {"bold": True, "color": RED}),
     "split 시 stats.json을 복사해 학습 통계가 held-out 9개를 포함한 81 ep로 fit돼 있었음",
     "72 ep로 재생성 (30/57 차원 변경)"],
    [("오해를 부르는 로그", {"bold": True, "color": AMBER}),
     "generate_stats는 “Generating stats…”를 찍은 뒤 유효성 검사로 조기 반환한다",
     "로그는 계산의 증거가 아님을 문서화"],
    [("execution_horizon 플래그", {"bold": True, "color": AMBER}),
     "run_rldx_server의 --execution-horizon은 ReplayPolicy 전용. 실제 정책엔 안 먹는다",
     "실행 개수는 클라이언트 결정"],
    [("episode_sampling_rate", {"bold": True, "color": AMBER}),
     "이름과 달리 데이터를 버리지 않는다 — 에피소드를 10개 서브시퀀스로 쪼개 shard에 분산",
     "기본값 유지"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[2.0, 5.9, 2.3], row_h=Inches(0.52), size=12)

callout(s, MARGIN, Inches(5.15), Inches(6.1), Inches(1.45), "검증 완료된 것",
        ["중간 저장 · 회전 (20 저장 → 40·45 남기고 20 삭제)",
         "open_loop_eval 26차원 관통 — gt(64,26) / pred(64,26)",
         "체크포인트 자립성 (modality config·통계를 자체 보유)",
         "학습 파라미터 2,376.0M = 논문 Table 6과 일치"], accent=GREEN, bsize=12.5)
callout(s, MARGIN + Inches(6.4), Inches(5.15), Inches(5.67), Inches(1.45), "아직 확인 못 한 것",
        ["30K 완주 (최장 45 step까지만 돌려봄)",
         "중단 후 resume",
         "실로봇 서빙 — 26차원 체크포인트로 서버를 띄워본 적 없음",
         "chunk 40의 장기 거동"], accent=AMBER, bsize=12.5)
footer(s, "pre-flight", nxt())

# 14b — 1차 학습 결과 -----------------------------------------------------------
s = slide_base(prs, "1차 학습 결과 — checkpoint-10000이 최적", "30K 완주 후 held-out 채점")
rows = [
    ["step", "held-out MSE", "MAE", "최적 대비", "train loss"],
    [("10,000", {"bold": True, "color": GREEN}), ("0.02379", {"bold": True, "color": GREEN}),
     ("0.09208", {"bold": True}), ("— 최적", {"bold": True, "color": GREEN}), "0.0129"],
    ["20,000", "0.02653", "0.09858", ("+11.6%", {"color": AMBER}), "0.0034"],
    ["30,000", "0.02663", "0.09985", ("+11.9%", {"color": RED}), ("0.0031", {"bold": True})],
]
table(s, MARGIN, BODY_TOP, Inches(7.2), rows, col_w=[1.2, 1.6, 1.3, 1.4, 1.4], row_h=Inches(0.44))

tf = body_frame(s, MARGIN + Inches(7.55), BODY_TOP - Inches(0.05), Inches(5.15), Inches(2.2))
_txt(tf, [("train loss는 4배 좋아지는 동안", {"size": 15, "bold": True, "color": NAVY})], first=True, space_after=3)
_txt(tf, [("held-out은 나빠졌다 — 과적합이다.", {"size": 15, "bold": True, "color": RED})], space_after=9)
bullets(tf, [
    ("0.0129 → 0.0031 (train)  vs  0.0238 → 0.0266 (held-out)", {"size": 13}),
    ("그림: outputs/eval/holdout_curve.png", {"size": 13}),
], first=False)

callout(s, MARGIN, Inches(3.7), Inches(6.1), Inches(1.75), "±15% 변동을 어떻게 넘어섰나",
        ["같은 체크포인트도 실행마다 MSE가 ±15% 흔들린다(부록 A.4).",
         "평균 차이 11.6%는 그 폭 안이라 그것만으로는 약하다.",
         "→ 9개 에피소드를 짝지어 보니 step 10,000이 9/9 전부에서 이긴다.",
         "   노이즈는 에피소드마다 독립이므로 우연이 아니다."], accent=GREEN, bsize=12.5)
callout(s, MARGIN + Inches(6.4), Inches(3.7), Inches(5.67), Inches(1.75), "50K 연장은 중단했다",
        ["10K → 30K가 held-out을 하나도 개선하지 못했다.",
         "게다가 resume은 웜 리스타트다 — lr_lambdas가 [{}]로만 저장돼",
         "새 50K 코사인이 적용되고 LR이 ~0 → 3.77e-5로 되올라간다(실측).",
         "과적합 구간에서 LR을 다시 올리는 셈."], accent=AMBER, bsize=12.5)
callout(s, MARGIN, Inches(5.6), BODY_W, Inches(1.05), "그래서 다음 레버는 스텝이 아니라 데이터다",
        ["81 에피소드에서 10,000 스텝(45 epoch)이면 이 데이터로 배울 건 다 배웠다는 뜻이다.",
         "→ 초기 송장 위치 분포를 세고, 부족한 분기를 집중 수집한다 (§6.3)."], accent=NAVY)
footer(s, "1차 결과", nxt())

# 15 — to-do ------------------------------------------------------------------
s = slide_base(prs, "To-do", "지금 / 학습 후 / 다음 데이터")
rows = [
    ["시점", "할 일", "왜"],
    [("학습 도는 동안", {"bold": True}), "배포 클라이언트의 26 → 57 관절 매핑 갱신",
     "서버 출력 키가 바뀐다. 유일하게 한 번도 안 띄워본 경로"],
    ["", "데이터팀 확인: 초기 송장 위치 분포 (front/top/hidden/side)",
     "분기 균형을 모르면 소수 분기가 학습되지 않는다"],
    [("학습 직후", {"bold": True}), "10K / 20K / 30K를 held-out으로 평가 → 최저 MSE 선택",
     "validation이 없어 train loss만으로는 과적합을 못 본다"],
    ["", "실기 롤아웃으로 실패 모드 관찰",
     "“전 구간 어설픔”인가 “전환 지점에서 멈춤”인가"],
    ["", "배포 시 40 중 20만 실행 (2.0초), 여유되면 RTC guided",
     "16개 전부 = 4초 open-loop는 컨베이어에서 길다"],
    [("다음 데이터", {"bold": True}), "부족한 분기를 골라 집중 수집 (논문 §4.3 adaptive collection)",
     "무작정 늘리는 것보다 효율이 높다고 보고됨"],
    ["", "필요해지면 subtask 도입 (B) — 코드는 이미 지원",
     "1차 실패 모드가 “전환 지점”으로 나올 때만"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[1.6, 4.6, 4.0], row_h=Inches(0.42), size=12)

callout(s, MARGIN, Inches(5.6), BODY_W, Inches(1.15), "목표치를 굳이 잡자면",
        ["분기당 40~60개, 총 120~160 에피소드가 논문 레시피와 맞는 규모다. 지금 81개는 “주 동작은 학습되지만 조건 분기는 부족할 수 있다”는 위치.",
         "정밀 접촉 작업을 하게 되면 그때는 10 fps도 재고 대상 (1스텝 = 100 ms)."], accent=NAVY)
footer(s, "to-do", nxt())

# 16 — 열린 질문 ---------------------------------------------------------------
s = slide_base(prs, "열린 질문", "결정을 바꿀 수 있는 것들")
rows = [
    ["질문", "지금 아는 것", "모르면 생기는 일"],
    ["초기 송장 위치의 분포는?",
     "신호가 화면에 있다는 건 확인. 분포는 사람이 라벨링해야 한다",
     "소수 분기가 20% 미만이면 그 분기는 학습되지 않는다"],
    ["세션마다 시연이 길어지는 이유는? (320→497)",
     "분기가 아니라 세션 효과로 보임 (길이·파지 횟수 상관 0.01)",
     "분포 이동이면 평가 설계에 반영해야 한다"],
    ["chunk 40이 16보다 나은가",
     "위치 인코딩은 해석적이라 잘 정의됨. 학습 때 본 적 없는 길이일 뿐",
     "실제 성능 차이는 돌려봐야 안다"],
]
table(s, MARGIN, BODY_TOP, BODY_W, rows, col_w=[2.6, 4.6, 3.0], row_h=Inches(0.62), size=12)

callout(s, MARGIN, Inches(5.5), BODY_W, Inches(1.2), "상세 근거",
        ["모든 수치는 RLDX-1-FFW-SH5.md에 재현 명령과 함께 있다. 논문 인용에는 절(section)·표 번호를 붙였다.",
         "프레임 시트: outputs/frames/  ·  평가 플롯: outputs/eval/  ·  데이터 도구: run_scripts/data/"], accent=GREY)
footer(s, "열린 질문", nxt())

prs.save(OUT)
print(f"saved: {OUT}  ({n} slides)")
