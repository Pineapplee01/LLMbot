from __future__ import annotations

import json
import re
import zipfile
from html import escape
from pathlib import Path


FIG_DIR = Path(__file__).resolve().parent
TEMPLATE = FIG_DIR / "Architecture.vsdx"
DATA_JSON = FIG_DIR / "data" / "casestudy_node11654_evidence.json"
OUT_VSDX = FIG_DIR / "casestudy.vsdx"

REF_W = 1400.0
REF_H = 780.0
PAGE_W = 14.0
PAGE_H = 7.8

BOT = "#C84C4C"
HUMAN = "#4C78A8"
GREEN = "#2F855A"
RED = "#C84C4C"
GRAY = "#667085"
BLACK = "#182230"
PANEL_EDGE = "#9AA5B1"
WHITE = "#FFFFFF"


class ShapeBuilder:
    def __init__(self) -> None:
        self._next_id = 1
        self.parts: list[str] = []

    def _id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    @staticmethod
    def _x(px: float) -> float:
        return PAGE_W * px / REF_W

    @staticmethod
    def _y(py: float) -> float:
        return PAGE_H - (PAGE_H * py / REF_H)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _hex(color: str) -> str:
        return color.lower()

    @staticmethod
    def _pt(size: float) -> float:
        return size / 72.0

    def _character(self, size: float, color: str = BLACK, bold: bool = False) -> str:
        style = 1 if bold else 0
        return (
            "<Section N='Character'><Row IX='0'>"
            "<Cell N='Font' V='Times New Roman' F='FONT(\"Times New Roman\")'/>"
            f"<Cell N='Color' V='{self._hex(color)}'/>"
            f"<Cell N='Style' V='{style}'/>"
            f"<Cell N='Size' V='{self._fmt(self._pt(size))}' U='PT'/>"
            "</Row></Section>"
        )

    def _paragraph(self, align: int = 0) -> str:
        return f"<Section N='Paragraph'><Row IX='0'><Cell N='HorzAlign' V='{align}'/></Row></Section>"

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        text: str = "",
        fill: str = WHITE,
        line: str = PANEL_EDGE,
        line_pt: float = 0.8,
        size: float = 8,
        bold: bool = False,
        align: int = 0,
        text_color: str = BLACK,
        no_fill: bool = False,
        no_line: bool = False,
    ) -> None:
        sid = self._id()
        pinx = self._x(x + w / 2)
        piny = self._y(y + h / 2)
        width = self._x(w)
        height = self._x(h)
        fill_pattern = 0 if no_fill else 1
        line_pattern = 0 if no_line else 1
        text_xml = f"<Text><cp IX='0'/>{escape(text)}</Text>" if text else ""
        self.parts.append(
            f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
            f"<Cell N='PinX' V='{self._fmt(pinx)}'/>"
            f"<Cell N='PinY' V='{self._fmt(piny)}'/>"
            f"<Cell N='Width' V='{self._fmt(width)}'/>"
            f"<Cell N='Height' V='{self._fmt(height)}'/>"
            f"<Cell N='LocPinX' V='{self._fmt(width / 2)}' F='Width*0.5'/>"
            f"<Cell N='LocPinY' V='{self._fmt(height / 2)}' F='Height*0.5'/>"
            f"<Cell N='FillPattern' V='{fill_pattern}'/>"
            f"<Cell N='FillForegnd' V='{self._hex(fill)}'/>"
            f"<Cell N='LinePattern' V='{line_pattern}'/>"
            f"<Cell N='LineColor' V='{self._hex(line)}'/>"
            f"<Cell N='LineWeight' V='{self._fmt(line_pt / 72)}' U='PT'/>"
            f"<Cell N='Rounding' V='{self._fmt(self._x(8))}'/>"
            f"<Cell N='VerticalAlign' V='0'/>"
            "<Cell N='TxtMarginLeft' V='0.03'/>"
            "<Cell N='TxtMarginRight' V='0.03'/>"
            "<Cell N='TxtMarginTop' V='0.03'/>"
            "<Cell N='TxtMarginBottom' V='0.03'/>"
            f"{self._character(size, text_color, bold)}"
            f"{self._paragraph(align)}"
            "<Section N='Geometry' IX='0'>"
            "<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/><Cell N='NoShow' V='0'/><Cell N='NoSnap' V='0'/><Cell N='NoQuickDrag' V='0'/>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            f"<Row T='LineTo' IX='2'><Cell N='X' V='{self._fmt(width)}'/><Cell N='Y' V='0'/></Row>"
            f"<Row T='LineTo' IX='3'><Cell N='X' V='{self._fmt(width)}'/><Cell N='Y' V='{self._fmt(height)}'/></Row>"
            f"<Row T='LineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='{self._fmt(height)}'/></Row>"
            "<Row T='LineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            "</Section>"
            f"{text_xml}"
            "</Shape>"
        )

    def text(self, x: float, y: float, w: float, h: float, text: str, *, size: float = 8, color: str = BLACK, bold: bool = False, align: int = 0) -> None:
        self.rect(x, y, w, h, text=text, fill=WHITE, line=WHITE, size=size, bold=bold, align=align, text_color=color, no_fill=True, no_line=True)

    def oval(self, cx: float, cy: float, r: float, label: str, fill: str, *, sublabel: str = "") -> None:
        sid = self._id()
        width = self._x(2 * r)
        height = self._x(2 * r)
        pinx = self._x(cx)
        piny = self._y(cy)
        text_xml = f"<Text><cp IX='0'/>{escape(label)}</Text>"
        self.parts.append(
            f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
            f"<Cell N='PinX' V='{self._fmt(pinx)}'/>"
            f"<Cell N='PinY' V='{self._fmt(piny)}'/>"
            f"<Cell N='Width' V='{self._fmt(width)}'/>"
            f"<Cell N='Height' V='{self._fmt(height)}'/>"
            f"<Cell N='LocPinX' V='{self._fmt(width / 2)}' F='Width*0.5'/>"
            f"<Cell N='LocPinY' V='{self._fmt(height / 2)}' F='Height*0.5'/>"
            "<Cell N='FillPattern' V='1'/>"
            f"<Cell N='FillForegnd' V='{self._hex(fill)}'/>"
            "<Cell N='LinePattern' V='1'/>"
            "<Cell N='LineColor' V='#ffffff'/>"
            "<Cell N='LineWeight' V='0.012' U='PT'/>"
            "<Cell N='VerticalAlign' V='1'/>"
            f"{self._character(8, WHITE, True)}"
            f"{self._paragraph(1)}"
            "<Section N='Geometry' IX='0'>"
            "<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/><Cell N='NoShow' V='0'/><Cell N='NoSnap' V='0'/><Cell N='NoQuickDrag' V='0'/>"
            f"<Row T='Ellipse' IX='1'><Cell N='X' V='{self._fmt(width / 2)}' F='Width*0.5'/><Cell N='Y' V='{self._fmt(height / 2)}' F='Height*0.5'/>"
            f"<Cell N='A' V='{self._fmt(width)}' U='DL' F='Width*1'/><Cell N='B' V='{self._fmt(height / 2)}' U='DL' F='Height*0.5'/>"
            f"<Cell N='C' V='{self._fmt(width / 2)}' U='DL' F='Width*0.5'/><Cell N='D' V='{self._fmt(height)}' U='DL' F='Height*1'/></Row>"
            "</Section>"
            f"{text_xml}"
            "</Shape>"
        )
        if sublabel:
            self.text(cx - 32, cy + r + 5, 64, 18, sublabel, size=6.5, color=GRAY, align=1)

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = "#344054", line_pt: float = 0.8, dash: bool = False) -> None:
        sid = self._id()
        bx, by = self._x(x1), self._y(y1)
        ex, ey = self._x(x2), self._y(y2)
        width = ex - bx
        height = ey - by
        pinx = (bx + ex) / 2
        piny = (by + ey) / 2
        pattern = 2 if dash else 1
        self.parts.append(
            f"<Shape ID='{sid}' Type='Shape' LineStyle='3' FillStyle='3' TextStyle='3'>"
            f"<Cell N='PinX' V='{self._fmt(pinx)}'/>"
            f"<Cell N='PinY' V='{self._fmt(piny)}'/>"
            f"<Cell N='Width' V='{self._fmt(width)}' F='GUARD(EndX-BeginX)'/>"
            f"<Cell N='Height' V='{self._fmt(height)}' F='GUARD(EndY-BeginY)'/>"
            f"<Cell N='LocPinX' V='{self._fmt(width / 2)}'/>"
            f"<Cell N='LocPinY' V='{self._fmt(height / 2)}'/>"
            f"<Cell N='BeginX' V='{self._fmt(bx)}'/>"
            f"<Cell N='BeginY' V='{self._fmt(by)}'/>"
            f"<Cell N='EndX' V='{self._fmt(ex)}'/>"
            f"<Cell N='EndY' V='{self._fmt(ey)}'/>"
            f"<Cell N='LineColor' V='{self._hex(color)}'/>"
            f"<Cell N='LineWeight' V='{self._fmt(line_pt / 72)}' U='PT'/>"
            f"<Cell N='LinePattern' V='{pattern}'/>"
            "<Cell N='FillPattern' V='0'/>"
            "<Section N='Geometry' IX='0'>"
            "<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>"
            f"<Row T='LineTo' IX='2'><Cell N='X' V='{self._fmt(width)}'/><Cell N='Y' V='{self._fmt(height)}'/></Row>"
            "</Section>"
            "</Shape>"
        )


def label_color(label_name: str) -> str:
    return BOT if label_name == "Bot" else HUMAN


def short(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_page(data: dict) -> str:
    b = ShapeBuilder()

    # Panels.
    b.rect(25, 45, 335, 690, fill="#F7FBFF", line=PANEL_EDGE, line_pt=1.0)
    b.rect(385, 45, 285, 690, fill="#FAFAF7", line=PANEL_EDGE, line_pt=1.0)
    b.rect(695, 45, 280, 690, fill="#F8FBF8", line=PANEL_EDGE, line_pt=1.0)
    b.text(45, 66, 280, 28, "Target Evidence", size=10, bold=True)
    b.text(400, 66, 260, 28, "Local / Support Neighborhood", size=10, bold=True)
    b.text(725, 66, 220, 28, "Branch Evidence", size=10, bold=True)

    target = data["target"]
    case = data["case"]
    header = (
        f"node {case['node_index']} | {target['user_id']} | @{target['username']}\n"
        f"gold: {case['gold_name']}    risk: {case['risk_score']:.3f}    routed"
    )
    b.rect(45, 115, 295, 65, text=header, fill=WHITE, line=BOT, line_pt=1.1, size=8, bold=True)
    b.rect(45, 205, 285, 110, text="Description:\n" + short(target["description"], 120), fill="#FFF5D9", line="#C59F3F", size=6.8, bold=True)
    meta = (
        f"Metadata:\ncreated: {target['created_at'][:16]}\n"
        f"followers/following: {target['followers_count']} / {target['following_count']}\n"
        f"tweets/listed: {target['tweet_count']} / {target['listed_count']}\n"
        f"verified/protected: {target['verified']} / {target['protected']}"
    )
    b.rect(45, 335, 285, 125, text=meta, fill="#FBE7D9", line="#C98962", size=6.9, bold=True)
    tweets = "\n".join(f"{idx + 1}. {short(tweet, 88)}" for idx, tweet in enumerate(target["tweets"][:4]))
    b.rect(45, 485, 285, 220, text="Tweets:\n" + tweets, fill="#DFF1FF", line="#6A9EC3", size=6.5, bold=True)

    # Neighborhood.
    cx, cy = 525.0, 425.0
    rel_pos = [(455, 255), (600, 255)]
    for pos in rel_pos:
        b.line(cx, cy, pos[0], pos[1], color="#344054", line_pt=1.0)
    support_pos = [(430, 535), (465, 595), (515, 625), (575, 605), (620, 545), (610, 465), (555, 492), (475, 492)]
    for pos in support_pos:
        b.line(cx, cy, pos[0], pos[1], color=GRAY, line_pt=0.8, dash=True)

    b.oval(cx, cy, 32, "T", BOT, sublabel=str(case["node_index"]))
    b.text(490, 350, 70, 18, "target", size=7, color=GRAY, align=1)
    for pos, nb in zip(rel_pos, data["relation_neighbors"]):
        b.oval(pos[0], pos[1], 22, nb["label_name"][0], label_color(nb["label_name"]), sublabel=str(nb["node_index"]))
    b.text(435, 190, 190, 20, f"relation bot ratio = {case['relation_bot_ratio']:.3f}", size=7.5, align=1)
    for pos, nb in zip(support_pos, data["support_neighbors"]):
        b.oval(pos[0], pos[1], 18, nb["label_name"][0], label_color(nb["label_name"]), sublabel=str(nb["rank"]))
    b.text(420, 650, 210, 18, f"KNN support bot ratio = {case['support_bot_ratio']:.3f}", size=7.5, align=1)
    b.text(430, 680, 190, 18, "support labels: B B B B B H B B", size=7.2, color=GRAY, align=1)

    # Branch evidence.
    y = 125.0
    for branch in data["branch_evidence"]:
        edge = GREEN if branch["correct"] else RED
        pred = branch["prediction"]
        b.rect(725, y, 225, 88, fill=WHITE, line=edge, line_pt=1.2)
        b.text(737, y + 10, 125, 20, branch["name"], size=7.8, bold=True)
        b.text(867, y + 10, 70, 20, f"conf {branch['true_confidence']:.3f}", size=6.7, color=GRAY, align=2)
        b.text(737, y + 38, 70, 18, "prediction", size=6.6, color=GRAY)
        b.text(817, y + 38, 70, 18, pred, size=7.4, color=label_color(pred), bold=True)
        b.text(867, y + 38, 70, 18, "correct" if branch["correct"] else "wrong", size=7.0, color=edge, bold=True, align=2)
        b.text(737, y + 64, 190, 16, f"margin true-other {branch['margin_true_minus_other']:+.3f}", size=6.5, color=GRAY)
        y += 110
    b.rect(
        725,
        590,
        225,
        90,
        text="Final decision:\nRouted-only residual -> Bot\nLow-order error is corrected.",
        fill=WHITE,
        line=GREEN,
        line_pt=1.2,
        size=6.9,
        bold=True,
    )

    return (
        "<?xml version='1.0' encoding='utf-8' ?>\n"
        "<PageContents xmlns='http://schemas.microsoft.com/office/visio/2012/main' "
        "xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships' "
        "xml:space='preserve'><Shapes>"
        + "".join(b.parts)
        + "</Shapes></PageContents>"
    )


def update_pages_xml(text: str) -> str:
    text = re.sub(r"<Cell N='PageWidth' V='[^']*'/>", f"<Cell N='PageWidth' V='{PAGE_W}'/>", text)
    text = re.sub(r"<Cell N='PageHeight' V='[^']*'/>", f"<Cell N='PageHeight' V='{PAGE_H}'/>", text)
    text = re.sub(r"ViewCenterX='[^']*'", f"ViewCenterX='{PAGE_W / 2}'", text)
    text = re.sub(r"ViewCenterY='[^']*'", f"ViewCenterY='{PAGE_H / 2}'", text)
    return text


def update_windows_xml(text: str) -> str:
    text = re.sub(r"ViewCenterX='[^']*'", f"ViewCenterX='{PAGE_W / 2}'", text)
    text = re.sub(r"ViewCenterY='[^']*'", f"ViewCenterY='{PAGE_H / 2}'", text)
    return text


def main() -> None:
    data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    page_xml = build_page(data)
    tmp_path = OUT_VSDX.with_suffix(".vsdx.tmp")
    with zipfile.ZipFile(TEMPLATE, "r") as src, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            payload = src.read(info.filename)
            if info.filename == "visio/pages/page1.xml":
                payload = page_xml.encode("utf-8")
            elif info.filename == "visio/pages/pages.xml":
                payload = update_pages_xml(payload.decode("utf-8", errors="replace")).encode("utf-8")
            elif info.filename == "visio/windows.xml":
                payload = update_windows_xml(payload.decode("utf-8", errors="replace")).encode("utf-8")
            dst.writestr(info, payload)
    tmp_path.replace(OUT_VSDX)
    print(f"Wrote {OUT_VSDX}")


if __name__ == "__main__":
    main()
