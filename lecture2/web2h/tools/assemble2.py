"""Assemble the 2-hour deck: one slide per beat, with the motion graphics as a video at the back and every
static title, caption, label, card, table, code block and equation as a native, editable PowerPoint object.

    python3 tools/assemble2.py <clipdir> <textdir> <out.pptx>

<clipdir>  what `node tools/record.cjs <clipdir> --split` wrote: manifest.json, sNN_<id>_bB.mp4, _first.jpg
<textdir>  what `node tools/extract.cjs <textdir>` wrote: sNN_<id>_bB.json and the equation PNGs

Text is set in the deck's own fonts (Unbounded, IBM Plex Sans, IBM Plex Mono). Install the TTFs in
fonts-ttf/ before opening the deck, or PowerPoint substitutes them. Equations are pictures whose alt text is
their LaTeX. Everything that appears during a beat fades in on the same delay as in the browser.
"""
import json
import os
import sys
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Emu

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
POSTER = os.environ.get('POSTER', 'first')   # POSTER=last shows the end of each beat (for checking layouts)
NOBG = ' animBg="1"'
PX = 6350                      # EMU per stage pixel (1920 px = 13.333 in)

CHAPTERS = ["Opening", "1 · A line that learns", "2 · Rolling downhill", "3 · Yes or no?", "4 · Stacking neurons",
            "5 · Blame flows backwards", "6 · Watch it learn", "7 · Gradients for engineers", "8 · Wrap-up"]

# (CSS family, weight, italic) -> (PowerPoint typeface, bold, italic), from the name tables of fonts-ttf/*.ttf
FACES = {
    "Unbounded": {600: ("Unbounded SemiBold", 0), 700: ("Unbounded", 1), 800: ("Unbounded ExtraBold", 0)},
    "IBM Plex Sans": {400: ("IBM Plex Sans", 0), 500: ("IBM Plex Sans Medium", 0), 600: ("IBM Plex Sans SemiBold", 0),
                      700: ("IBM Plex Sans", 1)},
    "IBM Plex Mono": {400: ("IBM Plex Mono", 0), 500: ("IBM Plex Mono Medium", 0), 600: ("IBM Plex Mono SemiBold", 0),
                      700: ("IBM Plex Mono", 1)},
}


def face(font: str, weight: int) -> tuple[str, int]:
    table = FACES.get(font)
    if not table:
        return font, int(weight >= 600)
    w = min(table, key=lambda k: abs(k - weight))
    return table[w]


def emu(v: float) -> Emu:
    return Emu(int(round(v * PX)))


def fill_xml(hexc: str, alpha: float) -> str:
    a = f'<a:alpha val="{int(round(max(0, min(1, alpha)) * 100000))}"/>' if alpha < 0.999 else ""
    return f'<a:solidFill><a:srgbClr val="{hexc}">{a}</a:srgbClr></a:solidFill>'


def set_fill(sp, hexc, alpha):
    """Replace a shape's fill with a solid colour (or none)."""
    spPr = sp._element.spPr
    for tag in ("solidFill", "noFill", "gradFill", "pattFill"):
        for e in spPr.findall(f"{{{A}}}{tag}"):
            spPr.remove(e)
    x = fill_xml(hexc, alpha) if hexc else "<a:noFill/>"
    el = etree.fromstring(f'<x xmlns:a="{A}">{x}</x>')[0]
    geom = spPr.find(f"{{{A}}}prstGeom")
    geom.addnext(el)


def set_line(sp, hexc, alpha, width_px):
    spPr = sp._element.spPr
    for e in spPr.findall(f"{{{A}}}ln"):
        spPr.remove(e)
    inner = fill_xml(hexc, alpha) if hexc and width_px > 0 else "<a:noFill/>"
    w = f' w="{int(round(width_px * PX))}"' if hexc and width_px > 0 else ""
    el = etree.fromstring(f'<a:ln xmlns:a="{A}"{w}>{inner}</a:ln>')
    spPr.append(el)


def no_shadow(sp):
    """python-pptx autoshapes pick up the theme's style; strip it so only our fill and line show."""
    style = sp._element.find(f"{{{P}}}style")
    if style is not None:
        sp._element.remove(style)


def text_box(slide, it):
    """A text box at the browser's content box, with the browser's own line breaks."""
    slack = max(12.0, it["w"] * 0.04)
    x, w = it["x"], it["w"] + slack
    if it["align"] == "center":
        x -= slack / 2
    elif it["align"] in ("right", "end"):
        x -= slack
    tb = slide.shapes.add_textbox(emu(x), emu(it["y"]), emu(w), emu(max(it["h"], it["lineH"])))
    body = tb.text_frame._txBody
    bp = body.find(f"{{{A}}}bodyPr")
    for k, v in dict(wrap="square", lIns="0", tIns="0", rIns="0", bIns="0", anchor="t").items():
        bp.set(k, v)
    for e in list(bp):
        bp.remove(e)
    etree.SubElement(bp, f"{{{A}}}noAutofit")
    for p in body.findall(f"{{{A}}}p"):
        body.remove(p)
    algn = {"center": "ctr", "right": "r", "end": "r", "justify": "just"}.get(it["align"], "l")
    lnspc = int(round(it["lineH"] * 0.5 * 100))
    parts = [f'<a:p xmlns:a="{A}"><a:pPr algn="{algn}"><a:lnSpc><a:spcPts val="{lnspc}"/></a:lnSpc>'
             '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:spcAft><a:spcPts val="0"/></a:spcAft><a:buNone/></a:pPr>']
    last = None
    for r in it["runs"]:
        if r.get("br"):
            parts.append(rpr_xml(last, "br") if last else "<a:br/>")
            continue
        s = r["s"]
        last = s
        parts.append(f'<a:r>{rpr_xml(s, "rPr")}<a:t>{escape(r["t"])}</a:t></a:r>')
    end = rpr_xml(last, "endParaRPr") if last else ""
    parts.append(end + "</a:p>")
    body.append(etree.fromstring("".join(parts)))
    tb.name = "Text: " + "".join(r.get("t", " ") for r in it["runs"])[:40]
    return tb


def rpr_xml(s, tag):
    tf, bold = face(s["font"], s["weight"])
    sz = int(round(s["size"] * 0.5 * 100))
    spc = int(round(s["spacing"] * 0.5 * 100))
    attrs = f'lang="en-US" sz="{sz}" b="{bold}" i="{int(s["italic"])}" dirty="0"' + (f' spc="{spc}"' if spc else "")
    inner = (fill_xml(s["color"], s["alpha"]) +
             f'<a:latin typeface="{escape(tf)}"/><a:ea typeface="{escape(tf)}"/><a:cs typeface="{escape(tf)}"/>')
    if tag == "br":
        return f'<a:br><a:rPr {attrs}>{inner}</a:rPr></a:br>'
    return f"<a:{tag} {attrs}>{inner}</a:{tag}>"


def rect(slide, it):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if it["radius"] > 0.5 else MSO_SHAPE.RECTANGLE
    sp = slide.shapes.add_shape(shape, emu(it["x"]), emu(it["y"]), emu(it["w"]), emu(it["h"]))
    if it["radius"] > 0.5:
        sp.adjustments[0] = min(0.5, it["radius"] / max(1.0, min(it["w"], it["h"])))
    no_shadow(sp)
    set_fill(sp, it["fill"], it["fillA"])
    set_line(sp, it["stroke"], it["strokeA"], it["strokeW"])
    sp.name = "Box"
    return sp


def line(slide, it):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, emu(it["x"]), emu(it["y"]), emu(it["x"] + it["w"]), emu(it["y"]))
    st = ln._element.find(f"{{{P}}}style")
    if st is not None:
        ln._element.remove(st)
    spPr = ln._element.spPr
    spPr.append(etree.fromstring(f'<a:ln xmlns:a="{A}" w="{int(round(it["strokeW"] * PX))}">{fill_xml(it["stroke"], it["strokeA"])}</a:ln>'))
    ln.name = "Rule"
    return ln


def math(slide, it, textdir: Path):
    pic = slide.shapes.add_picture(str(textdir / it["file"]), emu(it["x"]), emu(it["y"]), emu(it["w"]), emu(it["h"]))
    pic.name = "Equation"
    pic._element.nvPicPr.cNvPr.set("descr", "LaTeX: " + it["tex"])
    if it.get("alpha", 1) < 0.98:   # dimmed equation: apply the same opacity to the picture
        blip = pic._element.find(f".//{{{A}}}blip")
        etree.SubElement(blip, f"{{{A}}}alphaModFix", amt=str(int(it["alpha"] * 100000)))
    return pic


def timing(video_id: int, dur_ms: int, anims: list[tuple[int, int, bool]]) -> etree._Element:
    """The video starts by itself; each (shape id, delay ms) fades in on its own delay alongside it."""
    nid = [6]

    def n():
        nid[0] += 1
        return nid[0]

    fades = []
    for spid, delay, _ in anims:
        a, b, c = n(), n(), n()
        fades.append(f"""<p:par><p:cTn id="{a}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" grpId="0" nodeType="withEffect">
<p:stCondLst><p:cond delay="{delay}"/></p:stCondLst><p:childTnLst>
<p:set><p:cBhvr><p:cTn id="{b}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>
<p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{c}" dur="500"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect>
</p:childTnLst></p:cTn></p:par>""")
    vid = n()
    bld = "".join(f'<p:bldP spid="{spid}" grpId="0"{"" if txt else NOBG}/>' for spid, _, txt in anims)
    xml = f"""<p:timing xmlns:p="{P}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>
<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>
<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="afterEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="6" dur="{dur_ms}" fill="hold"/><p:tgtEl><p:spTgt spid="{video_id}"/></p:tgtEl></p:cBhvr></p:cmd>
</p:childTnLst></p:cTn></p:par>{"".join(fades)}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>
</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>
<p:video><p:cMediaNode vol="80000" mute="1"><p:cTn id="{vid}" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{video_id}"/></p:tgtEl></p:cMediaNode></p:video>
</p:childTnLst></p:cTn></p:par></p:tnLst>{f"<p:bldLst>{bld}</p:bldLst>" if bld else ""}</p:timing>"""
    return etree.fromstring(xml)


def main(clipdir: str, textdir: str, out: str) -> None:
    d, td = Path(clipdir), Path(textdir)
    scenes = json.loads((d / "manifest.json").read_text())
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(1920 * PX), Emu(1080 * PX)
    blank = prs.slide_layouts[6]
    sections: dict[int, list[int]] = {}
    for s in scenes:
        for b in range(s["beats"]):
            stem = f"s{s['i']:02d}_{s['id']}_b{b}"
            slide = prs.slides.add_slide(blank)
            mov = slide.shapes.add_movie(str(d / f"{stem}.mp4"), 0, 0, prs.slide_width, prs.slide_height,
                                         poster_frame_image=str(d / f"{stem}_{POSTER}.jpg"), mime_type="video/mp4")
            mov.name = "Animation"
            jf = td / f"{stem}.json"
            items = json.loads(jf.read_text()) if jf.exists() else []
            anims = []
            # shapes first (they sit under the text), then equations, then text: the browser's stacking
            order = {"rect": 0, "line": 0, "math": 1, "text": 2}
            for it in sorted(items, key=lambda q: order[q["type"]]):
                if it["type"] == "rect":
                    obj = rect(slide, it)
                elif it["type"] == "line":
                    obj = line(slide, it)
                elif it["type"] == "math":
                    obj = math(slide, it, td)
                else:
                    obj = text_box(slide, it)
                    if it.get("footer"):
                        obj.name = "Footer"
                rv = it.get("reveal") or {}
                if rv.get("b") == b:
                    anims.append((obj.shape_id, int(round(rv.get("d", 0) * 1000)), it["type"] == "text"))
            sld = slide._element
            for t in sld.findall(f"{{{P}}}timing"):
                sld.remove(t)
            tm = timing(mov.shape_id, s["dur"][b], anims)
            ext = sld.find(f"{{{P}}}extLst")
            (ext.addprevious(tm) if ext is not None else sld.append(tm))
            note = s["notes"][b] if b < len(s["notes"]) else ""
            head = f"[{s['title']} · step {b + 1} of {s['beats']}]"
            slide.notes_slide.notes_text_frame.text = f"{head}\n\n{note}".strip()
            sections.setdefault(s["ch"], []).append(slide.slide_id)

    pres = prs.part._element
    ext_lst = pres.find(f"{{{P}}}extLst")
    if ext_lst is None:
        ext_lst = etree.SubElement(pres, f"{{{P}}}extLst")
    ext = etree.SubElement(ext_lst, f"{{{P}}}ext", uri="{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
    sl = etree.SubElement(ext, f"{{{P14}}}sectionLst", nsmap={"p14": P14})
    for ch in sorted(sections):
        sec = etree.SubElement(sl, f"{{{P14}}}section", name=CHAPTERS[ch] if ch < len(CHAPTERS) else f"Chapter {ch}",
                               id="{" + str(uuid.uuid4()).upper() + "}")
        ids = etree.SubElement(sec, f"{{{P14}}}sldIdLst")
        for sid in sections[ch]:
            etree.SubElement(ids, f"{{{P14}}}sldId", id=str(sid))

    cp = prs.core_properties
    cp.title, cp.author, cp.subject = "Lecture 2 - Learning by gradients (2-hour edition)", "Paras Koundal", "From Decision Trees to LLMs"
    prs.save(out)
    print("slides", len(prs.slides), "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
