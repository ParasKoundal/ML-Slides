"""
motion.py - small toolkit for building "video-style" PowerPoint decks with python-pptx.

What it adds on top of python-pptx:
  * drawing helpers (dots, rects, freeform curves, text) that always set a shape *name*
  * Morph transitions: shapes that share a name starting with "!!" on two consecutive
    slides are tweened by PowerPoint (position, size, rotation, colour, geometry)
  * entrance / exit animations (fade, zoom-with-overshoot "pop", wipe, rise, fade-out)
    written straight into the slide's <p:timing> tree, chained automatically
  * auto-advance so a run of slides plays like a clip

Everything is plain OOXML, so the deck opens in PowerPoint 2019 / 365 without add-ins.
Morph needs PowerPoint 2019+, 365 or PowerPoint for the web; older versions fall back to a fade.
"""
from lxml import etree
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _rgb(hexstr):
    return RGBColor.from_string(hexstr)


# --------------------------------------------------------------------------- styling bits
def set_alpha(shape, alpha, which="fill"):
    """alpha in 0..1 (1 = opaque). Adds <a:alpha> to the solid fill or line colour."""
    sp = shape._element.spPr
    if which == "fill":
        clr = sp.find(qn("a:solidFill") + "/" + qn("a:srgbClr"))
    else:
        clr = sp.find(qn("a:ln") + "/" + qn("a:solidFill") + "/" + qn("a:srgbClr"))
    if clr is None:
        return
    for old in clr.findall(qn("a:alpha")):
        clr.remove(old)
    el = etree.SubElement(clr, qn("a:alpha"))
    el.set("val", str(int(alpha * 100000)))


def add_glow(shape, color, radius_pt=10, alpha=0.5):
    sp = shape._element.spPr
    eff = etree.SubElement(sp, qn("a:effectLst"))
    glow = etree.SubElement(eff, qn("a:glow"))
    glow.set("rad", str(int(radius_pt * 12700)))
    c = etree.SubElement(glow, qn("a:srgbClr"))
    c.set("val", color)
    a = etree.SubElement(c, qn("a:alpha"))
    a.set("val", str(int(alpha * 100000)))


def _style(shape, fill=None, line=None, line_w=1.0, alpha=None, line_alpha=None, dash=None):
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = _rgb(fill)
        if alpha is not None:
            set_alpha(shape, alpha)
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = _rgb(line)
        shape.line.width = Pt(line_w)
        if line_alpha is not None:
            set_alpha(shape, line_alpha, "line")
        if dash:
            ln = shape._element.spPr.find(qn("a:ln"))
            d = etree.SubElement(ln, qn("a:prstDash"))
            d.set("val", dash)
    shape.shadow.inherit = False


# --------------------------------------------------------------------------- shapes
def dot(sl, name, cx, cy, r, fill, line=None, line_w=1.0, alpha=None, glow=None):
    s = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - r), Inches(cy - r), Inches(2 * r), Inches(2 * r))
    s.name = name
    _style(s, fill, line, line_w, alpha)
    if glow:
        add_glow(s, glow, 12, 0.55)
    return s


def rect(sl, name, x, y, w, h, fill=None, line=None, line_w=1.0, alpha=None, radius=None, dash=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    s = sl.shapes.add_shape(kind, Inches(x), Inches(y), Inches(max(w, 0.001)), Inches(max(h, 0.001)))
    s.name = name
    if radius:
        s.adjustments[0] = min(0.5, radius / max(min(w, h), 1e-6))
    _style(s, fill, line, line_w, alpha, dash=dash)
    return s


def poly(sl, name, pts, color=None, width=2.0, fill=None, alpha=None, line_alpha=None, closed=False, dash=None):
    """Freeform through pts (inches). Keep the same point count across slides so Morph
    can tween the geometry point-by-point."""
    E = [(int(Inches(x)), int(Inches(y))) for x, y in pts]
    fb = sl.shapes.build_freeform(E[0][0], E[0][1], scale=1.0)
    fb.add_line_segments(E[1:], close=closed)
    s = fb.convert_to_shape()
    s.name = name
    _style(s, fill, color, width, alpha, line_alpha, dash)
    if color:
        ln = s._element.spPr.find(qn("a:ln"))
        ln.set("cap", "rnd")
        j = etree.SubElement(ln, qn("a:round"))
    return s


def seg(sl, name, x1, y1, x2, y2, color, width=2.0, alpha=None, dash=None):
    return poly(sl, name, [(x1, y1), (x2, y2)], color, width, line_alpha=alpha, dash=dash)


def text(sl, name, x, y, w, h, content, size=20, color="F2F0EA", bold=False, font="Calibri",
         align="l", anchor="t", italic=False, spacing=None, line_spacing=None):
    """content: str, or list of runs (text, {opts}); '\n' inside a str starts a new paragraph.
    run opts: color, bold, italic, font, size, sub, sup"""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(content, str):
        content = [(content, {})]
    # split runs into paragraphs on '\n'
    paras = [[]]
    for t, o in content:
        parts = t.split("\n")
        for i, part in enumerate(parts):
            if i > 0:
                paras.append([])
            if part:
                paras[-1].append((part, o))
    for pi, runs in enumerate(paras):
        p = tf.paragraphs[0] if pi == 0 else tf.add_paragraph()
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if line_spacing:
            p.line_spacing = line_spacing
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.name = o.get("font", font)
            f.color.rgb = _rgb(o.get("color", color))
            rPr = r._r.get_or_add_rPr()
            if spacing is not None:
                rPr.set("spc", str(int(spacing * 100)))
            if o.get("sub"):
                rPr.set("baseline", "-25000")
            if o.get("sup"):
                rPr.set("baseline", "30000")
    return tb


def picture(sl, name, path, x, y, w, h):
    p = sl.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    p.name = name
    return p


def background(sl, color):
    f = sl.background.fill
    f.solid()
    f.fore_color.rgb = _rgb(color)


# --------------------------------------------------------------------------- transitions
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
P159 = "http://schemas.microsoft.com/office/powerpoint/2015/09/main"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def set_transition(sl, kind="morph", dur_ms=1200, advance_ms=None):
    """kind: 'morph' | 'fade'. advance_ms: auto-advance after this many ms (also still on click)."""
    adv = f' advTm="{advance_ms}"' if advance_ms is not None else ""
    if kind == "morph":
        xml = (
            f'<mc:AlternateContent xmlns:mc="{MC}" xmlns:p="{NS["p"]}">'
            f'<mc:Choice xmlns:p159="{P159}" Requires="p159">'
            f'<p:transition xmlns:p14="{P14}" spd="slow" p14:dur="{dur_ms}"{adv}><p159:morph option="byObject"/></p:transition>'
            f"</mc:Choice>"
            f'<mc:Fallback><p:transition spd="slow"{adv}><p:fade/></p:transition></mc:Fallback>'
            f"</mc:AlternateContent>"
        )
    else:
        xml = (
            f'<mc:AlternateContent xmlns:mc="{MC}" xmlns:p="{NS["p"]}">'
            f'<mc:Choice xmlns:p14="{P14}" Requires="p14">'
            f'<p:transition spd="slow" p14:dur="{dur_ms}"{adv}><p:fade/></p:transition>'
            f"</mc:Choice>"
            f'<mc:Fallback><p:transition spd="slow"{adv}><p:fade/></p:transition></mc:Fallback>'
            f"</mc:AlternateContent>"
        )
    el = etree.fromstring(xml)
    root = sl._element
    clr = root.find(qn("p:clrMapOvr"))
    clr.addnext(el)
    return el


# --------------------------------------------------------------------------- animations
class Timeline:
    """Collect animations for one slide, then write them with .apply().

    tl.add(shape, effect, at=ms, dur=ms, click=0)
        effect: 'fade' | 'pop' | 'zoom' | 'wipe_l' | 'wipe_r' | 'wipe_u' | 'wipe_d' | 'rise' | 'out'
        at:     start time in ms, measured from the start of its click group
        click:  0 = plays automatically when the slide appears, 1 = on first click, 2 = second...
    """

    def __init__(self, slide):
        self.slide = slide
        self.items = []

    def add(self, shape, effect="fade", at=0, dur=500, click=0):
        self.items.append(dict(shape=shape, effect=effect, at=int(at), dur=int(dur), click=click))
        return self

    def stagger(self, shapes, effect="pop", start=0, step=60, dur=400, click=0):
        for i, s in enumerate(shapes):
            self.add(s, effect, start + i * step, dur, click)
        return start + (len(shapes) - 1) * step + dur

    # -- xml builders
    def _id(self):
        self._n += 1
        return self._n

    def _tgt(self, spid):
        return f'<p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>'

    def _set_vis(self, spid, val="visible", delay=0):
        return (
            f'<p:set><p:cBhvr><p:cTn id="{self._id()}" dur="1" fill="hold"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst></p:cTn>'
            f'{self._tgt(spid)}<p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr>'
            f'<p:to><p:strVal val="{val}"/></p:to></p:set>'
        )

    def _anim_effect(self, spid, filt, dur, transition="in"):
        return (
            f'<p:animEffect transition="{transition}" filter="{filt}"><p:cBhvr><p:cTn id="{self._id()}" dur="{dur}"/>'
            f"{self._tgt(spid)}</p:cBhvr></p:animEffect>"
        )

    def _anim_prop(self, spid, attr, keys, dur, decel=False):
        tav = "".join(
            f'<p:tav tm="{int(t * 100000)}"><p:val><p:strVal val="{v}"/></p:val></p:tav>' for t, v in keys
        )
        dc = ' decel="100000"' if decel else ""
        return (
            f'<p:anim calcmode="lin" valueType="num"><p:cBhvr additive="base"><p:cTn id="{self._id()}" dur="{dur}" fill="hold"{dc}/>'
            f"{self._tgt(spid)}<p:attrNameLst><p:attrName>{attr}</p:attrName></p:attrNameLst></p:cBhvr>"
            f"<p:tavLst>{tav}</p:tavLst></p:anim>"
        )

    def _effect(self, it, node_type, grp):
        spid = it["shape"].shape_id
        d = it["dur"]
        e = it["effect"]
        cls, pid, sub = "entr", 10, 0
        if e == "fade":
            body = self._set_vis(spid) + self._anim_effect(spid, "fade", d)
        elif e in ("pop", "zoom"):
            pid, sub = 53, 16
            if e == "pop":  # overshoot then settle - the "logo sting" snap
                kw = [(0, "0"), (0.7, "#ppt_w*1.12"), (1, "#ppt_w")]
                kh = [(0, "0"), (0.7, "#ppt_h*1.12"), (1, "#ppt_h")]
            else:
                kw = [(0, "#ppt_w*0.6"), (1, "#ppt_w")]
                kh = [(0, "#ppt_h*0.6"), (1, "#ppt_h")]
            body = (
                self._set_vis(spid)
                + self._anim_prop(spid, "ppt_w", kw, d)
                + self._anim_prop(spid, "ppt_h", kh, d)
                + self._anim_effect(spid, "fade", d)
            )
        elif e.startswith("wipe"):
            pid = 22
            direction = {"wipe_l": ("right", 8), "wipe_r": ("left", 2), "wipe_u": ("up", 4), "wipe_d": ("down", 1)}[e]
            sub = direction[1]
            body = self._set_vis(spid) + self._anim_effect(spid, f"wipe({direction[0]})", d)
        elif e == "rise":
            pid, sub = 42, 0
            body = (
                self._set_vis(spid)
                + self._anim_effect(spid, "fade", d)
                + self._anim_prop(spid, "ppt_y", [(0, "#ppt_y+0.04"), (1, "#ppt_y")], d, decel=True)
            )
        elif e == "out":
            cls, pid = "exit", 10
            body = self._anim_effect(spid, "fade", d, "out") + self._set_vis(spid, "hidden", d - 1)
        else:
            raise ValueError(e)
        return (
            f'<p:par><p:cTn id="{self._id()}" presetID="{pid}" presetClass="{cls}" presetSubtype="{sub}" '
            f'fill="hold" grpId="{grp}" nodeType="{node_type}"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
            f"<p:childTnLst>{body}</p:childTnLst></p:cTn></p:par>"
        )

    def apply(self):
        if not self.items:
            return
        self._n = 2
        groups = sorted(set(i["click"] for i in self.items))
        grp_count = {}
        bld = []
        group_xml = []
        for gi, g in enumerate(groups):
            items = sorted([i for i in self.items if i["click"] == g], key=lambda i: i["at"])
            slots = {}
            for it in items:
                slots.setdefault(it["at"], []).append(it)
            gid = self._id()
            slot_xml = []
            for si, t in enumerate(sorted(slots)):
                sid = self._id()
                effs = []
                for k, it in enumerate(slots[t]):
                    if si == 0 and k == 0:
                        nt = "afterEffect" if g == 0 else "clickEffect"
                    elif k == 0:
                        nt = "afterEffect"
                    else:
                        nt = "withEffect"
                    spid = it["shape"].shape_id
                    grp = grp_count.get(spid, 0)
                    grp_count[spid] = grp + 1
                    is_pic = it["shape"]._element.tag == qn("p:pic")
                    if not is_pic:
                        bld.append(f'<p:bldP spid="{spid}" grpId="{grp}" animBg="1"/>')
                    effs.append(self._effect(it, nt, grp))
                slot_xml.append(
                    f'<p:par><p:cTn id="{sid}" fill="hold"><p:stCondLst><p:cond delay="{t}"/></p:stCondLst>'
                    f'<p:childTnLst>{"".join(effs)}</p:childTnLst></p:cTn></p:par>'
                )
            start = '<p:cond delay="indefinite"/>'
            if g == 0:
                start += '<p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>'
            group_xml.append(
                f'<p:par><p:cTn id="{gid}" fill="hold"><p:stCondLst>{start}</p:stCondLst>'
                f'<p:childTnLst>{"".join(slot_xml)}</p:childTnLst></p:cTn></p:par>'
            )
        xml = (
            f'<p:timing xmlns:p="{NS["p"]}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
            f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
            f'{"".join(group_xml)}'
            f"</p:childTnLst></p:cTn>"
            f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
            f'<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
            f"</p:seq></p:childTnLst></p:cTn></p:par></p:tnLst>"
            + (f'<p:bldLst>{"".join(bld)}</p:bldLst>' if bld else "")
            + "</p:timing>"
        )
        el = etree.fromstring(xml)
        root = self.slide._element
        # timing goes after the transition (if any), before extLst
        anchor = root.find("{%s}AlternateContent" % MC)
        if anchor is None:
            anchor = root.find(qn("p:transition"))
        if anchor is None:
            anchor = root.find(qn("p:clrMapOvr"))
        anchor.addnext(el)


# --------------------------------------------------------------------------- plotting frame
class Ax:
    """Map data coordinates to slide inches inside a box (x0, y0, w, h)."""

    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h, self.xl, self.yl = x0, y0, w, h, xlim, ylim

    def X(self, x):
        return self.x0 + (x - self.xl[0]) / (self.xl[1] - self.xl[0]) * self.w

    def Y(self, y):
        return self.y0 + self.h - (y - self.yl[0]) / (self.yl[1] - self.yl[0]) * self.h

    def P(self, x, y):
        return (self.X(x), self.Y(y))

    def sy(self, dy):  # data height -> inches
        return dy / (self.yl[1] - self.yl[0]) * self.h
