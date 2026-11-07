"""Assemble the rendered clips into a PowerPoint deck: one slide per beat, a full-screen video that
starts by itself, the narration in the speaker notes, and one PowerPoint section per chapter.

    python3 tools/assemble.py <clipdir> <out.pptx>

<clipdir> is what tools/record.cjs wrote: manifest.json plus sNN_<id>_bB.mp4 and _first.jpg files.
"""
import json
import sys
import uuid
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.util import Emu

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"

CHAPTERS = ["Opening", "1 · A line that learns", "2 · Rolling downhill", "3 · Yes or no?", "4 · Stacking neurons",
            "5 · Blame flows backwards", "6 · Watch it learn", "7 · Wrap-up"]


def autoplay_timing(spid: int, dur_ms: int) -> etree._Element:
    """Timing tree PowerPoint writes for a video set to 'Start: Automatically'."""
    xml = f"""<p:timing xmlns:p="{P}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>
<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>
<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:par><p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="afterEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>
<p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="6" dur="{dur_ms}" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:cmd>
</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>
</p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>
<p:video><p:cMediaNode vol="80000" mute="1"><p:cTn id="7" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cMediaNode></p:video>
</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>"""
    return etree.fromstring(xml)


def main(clipdir: str, out: str) -> None:
    d = Path(clipdir)
    scenes = json.loads((d / "manifest.json").read_text())
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 16:9
    blank = prs.slide_layouts[6]
    sections: dict[int, list[int]] = {}
    for s in scenes:
        for b in range(s["beats"]):
            stem = f"s{s['i']:02d}_{s['id']}_b{b}"
            slide = prs.slides.add_slide(blank)
            mov = slide.shapes.add_movie(str(d / f"{stem}.mp4"), 0, 0, prs.slide_width, prs.slide_height,
                                         poster_frame_image=str(d / f"{stem}_first.jpg"), mime_type="video/mp4")
            mov.name = stem
            sld = slide._element
            for t in sld.findall(f"{{{P}}}timing"):
                sld.remove(t)
            # p:timing must follow p:clrMapOvr / p:transition and precede p:extLst
            timing = autoplay_timing(mov.shape_id, s["dur"][b])
            ext = sld.find(f"{{{P}}}extLst")
            (ext.addprevious(timing) if ext is not None else sld.append(timing))
            note = s["notes"][b] if b < len(s["notes"]) else ""
            head = f"[{s['title']} · step {b + 1} of {s['beats']}]"
            slide.notes_slide.notes_text_frame.text = f"{head}\n\n{note}".strip()
            sections.setdefault(s["ch"], []).append(slide.slide_id)

    # one section per chapter, so the slide sorter shows the structure
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
    cp.title, cp.author, cp.subject = "Lecture 2 - Learning by gradients", "Paras Koundal", "From Decision Trees to LLMs"
    prs.save(out)
    print("slides", len(prs.slides), "->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
