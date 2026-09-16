# -*- coding: utf-8 -*-
"""Markdown -> DOCX converter tailored for the DeepSeek V4.1 tech report translation."""
import os
import argparse
from pathlib import Path
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement, parse_xml
from mathml2omml import latex_to_omml_xml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report.zh.md"
DST = ROOT / "build/translation.zh.docx"
FIG_DIR = ROOT / "assets/figures"

HEI = u"黑体"
KAI = u"楷体"
SONG = u"宋体"
ARIAL = "Arial"

def set_run_font(run, east_asia=SONG, ascii_font=ARIAL, size=12, bold=False, color=None):
    run.font.name = ascii_font
    run.font.size = Pt(size)
    run.font.bold = bold
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), ascii_font)
    rFonts.set(qn('w:hAnsi'), ascii_font)
    rFonts.set(qn('w:eastAsia'), east_asia)
    if color:
        run.font.color.rgb = color

def add_heading(doc, text, level):
    h = doc.add_heading('', level=level)
    run = h.add_run(text)
    if level == 1:
        set_run_font(run, HEI, ARIAL, 16, True, RGBColor(0,0,0))
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
    elif level == 2:
        set_run_font(run, HEI, ARIAL, 14, True, RGBColor(0,0,0))
        h.paragraph_format.space_before = Pt(11)
        h.paragraph_format.space_after = Pt(5)
    else:
        set_run_font(run, HEI, ARIAL, 12, True, RGBColor(0,0,0))
        h.paragraph_format.space_before = Pt(9)
        h.paragraph_format.space_after = Pt(4)
    return h

def add_para(doc, text, size=12, indent=True, align=None, bold=False, east_asia=SONG):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align:
        pf.alignment = align
    # inline bold **...**
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            r = p.add_run(part[2:-2])
            set_run_font(r, east_asia, ARIAL, size, True)
        else:
            r = p.add_run(part)
            set_run_font(r, east_asia, ARIAL, size, bold)
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.left_indent = Pt(24)
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    # bullet char via numbering-free approach: use "•" prefix with hanging indent
    parts = re.split(r'(\*\*.*?\*\*)', text)
    first = True
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            r = p.add_run(part[2:-2])
            set_run_font(r, SONG, ARIAL, 12, True)
        else:
            r = p.add_run(part)
            set_run_font(r, SONG, ARIAL, 12, False)
    if first:
        r = p.runs[0] if p.runs else p.add_run('')
        r.text = '\u2022 ' + r.text
    pf.first_line_indent = Pt(-12)
    return p

def add_quote(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.left_indent = Pt(24)
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    r = p.add_run(text)
    set_run_font(r, KAI, ARIAL, 12, False, RGBColor(0x40,0x40,0x40))
    return p

def add_code(doc, text):
    for line in text.strip().split('\n'):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.left_indent = Pt(24)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        r = p.add_run(line if line else ' ')
        set_run_font(r, SONG, 'Consolas', 10.5, False)
    return

def set_cell_bg(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)
    tcPr.append(shd)

def add_table(doc, rows):
    # rows: list of list of str; first row header
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j in range(ncols):
            cell = table.cell(i, j)
            txt = row[j] if j < len(row) else ''
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            # alignment: header center; short text center; long text left; numbers right
            if i == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                if len(txt) <= 12:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif re.search(r'\d', txt) and not re.search(r'[A-Za-z\u4e00-\u9fff]', txt.replace(' ','').replace('-','').replace('.','').replace('%','')):
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            parts = re.split(r'(\*\*.*?\*\*)', txt)
            for part in parts:
                if not part:
                    continue
                if part.startswith('**') and part.endswith('**') and len(part) > 4:
                    r = p.add_run(part[2:-2])
                    set_run_font(r, SONG, ARIAL, 10.5, True)
                else:
                    r = p.add_run(part)
                    set_run_font(r, SONG, ARIAL, 10.5, False)
            if i == 0:
                set_cell_bg(cell, 'D9D9D9')
    # widths
    for j in range(ncols):
        for i in range(len(rows)):
            table.cell(i, j).width = Cm(16.0 / ncols)
    return table

def add_toc(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar = OxmlElement('w:fldChar')
    fldChar.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = 'TOC \\o "1-2" \\h \\z \\u'
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t')
    t.text = '（请在 Word 中右键此处选择"更新域"以生成目录）'
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(t)
    run._r.append(fldChar3)

def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fldChar1 = OxmlElement('w:fldChar'); fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText'); instrText.set(qn('xml:space'), 'preserve'); instrText.text = 'PAGE'
    fldChar2 = OxmlElement('w:fldChar'); fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1); run._r.append(instrText); run._r.append(fldChar2)
    set_run_font(run, SONG, ARIAL, 9, False)

def add_formula(doc, latex, annot, num):
    """Insert a Word-native equation (OMML), centered, with optional right-aligned number."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    # tab stops: center at 8cm, right at 16cm
    pf.tab_stops.add_tab_stop(Cm(8), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(16), WD_TAB_ALIGNMENT.RIGHT)
    # leading tab to center
    r0 = p.add_run('\t')
    set_run_font(r0, SONG, ARIAL, 12, False)
    # OMML equation
    try:
        omml = latex_to_omml_xml(latex)
        # wrap in m:oMath if not already (function returns <m:oMath>)
        if not omml.strip().startswith('<m:oMath'):
            omml = '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">' + omml + '</m:oMath>'
        omath_el = parse_xml(omml)
        p._p.append(omath_el)
    except Exception as e:
        # fallback: render latex as plain text
        r = p.add_run(latex.replace('\\', ''))
        set_run_font(r, SONG, 'Consolas', 11, False)
    # annotation text (e.g. （残差更新，对 n 收缩）)
    if annot:
        r = p.add_run(u'  ' + annot)
        set_run_font(r, SONG, ARIAL, 10.5, False)
    # right tab + number
    if num:
        r2 = p.add_run('\t')
        set_run_font(r2, SONG, ARIAL, 12, False)
        r3 = p.add_run(u'（%s）' % num)
        set_run_font(r3, SONG, ARIAL, 12, False)

def parse_md(source=SRC):
    with open(source, encoding='utf-8') as f:
        content = f.read()
    # Keep the equation-number parser while accepting GitHub math fences.
    fence = chr(96) * 3
    content = re.sub(r"^" + fence + r"math[ \t]*\n(.*?)^" + fence + r"[ \t]*\n\s*([^\n]*)",
                     lambda m: "$$" + m[1].strip().replace("\n", " ") + "$$" + m[2],
                     content, flags=re.M | re.S)
    # Figures are inserted by their numbered captions in build().
    content = re.sub(r"^!\[[^\]]*\]\([^\n]+\)\s*$", "", content, flags=re.M)
    lines = content.split("\n")
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip().lstrip('\ufeff')
        if not stripped:
            i += 1
            continue
        if stripped == '---':
            blocks.append(('hr', ''))
            i += 1
            continue
        if stripped.startswith('```'):
            # code block; lines starting with $$ are treated as equations
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith('```'):
                buf.append(lines[i]); i += 1
            i += 1  # skip closing
            # split into code lines and formula lines
            for bl in buf:
                s = bl.strip()
                mf = re.match(r'^\$\$(.+?)\$\$(.*)$', s)
                if mf:
                    latex = mf.group(1).strip()
                    tail = mf.group(2).strip()
                    # tail like （残差更新，对 n 收缩）（3）
                    mnum = re.search(r'（(\d+)）\s*$', tail)
                    num = mnum.group(1) if mnum else ''
                    annot = re.sub(r'（\d+）\s*$', '', tail).strip()
                    blocks.append(('formula', (latex, annot, num)))
            # collect pure code lines (non-formula)
            code_lines = [bl for bl in buf if not re.match(r'^\s*\$\$.+?\$\$', bl.strip())]
            if code_lines:
                blocks.append(('code', '\n'.join(code_lines)))
            continue
        m = re.match(r'^(#{1,4})\s+(.*)$', stripped)
        if m:
            blocks.append(('h' + str(len(m.group(1))), m.group(2).strip()))
            i += 1
            continue
        if stripped.startswith('|'):
            buf = []
            while i < n and lines[i].strip().startswith('|'):
                buf.append(lines[i].strip()); i += 1
            # parse table
            rows = []
            for rline in buf:
                cells = [c.strip() for c in rline.strip().strip('|').split('|')]
                if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
                    continue
                rows.append(cells)
            if rows:
                blocks.append(('table', rows))
            continue
        if stripped.startswith('- '):
            blocks.append(('bullet', stripped[2:]))
            i += 1
            continue
        if stripped.startswith('> '):
            blocks.append(('quote', stripped[2:]))
            i += 1
            continue
        if stripped.startswith('$$'):
            # equation line like $$latex$$（注释）（编号）
            mf = re.match(r'^\$\$(.+?)\$\$(.*)$', stripped)
            if mf:
                latex = mf.group(1).strip()
                tail = mf.group(2).strip()
                mnum = re.search(r'（(\d+)）\s*$', tail)
                num = mnum.group(1) if mnum else ''
                annot = re.sub(r'（\d+）\s*$', '', tail).strip()
                blocks.append(('formula', (latex, annot, num)))
                i += 1
                continue
        # normal paragraph; merge following non-empty non-special lines
        buf = [line]
        i += 1
        while i < n:
            s = lines[i].strip()
            if not s or s == '---' or s.startswith('#') or s.startswith('```') or s.startswith('|') or s.startswith('- ') or s.startswith('> ') or s.startswith('$$'):
                break
            buf.append(lines[i])
            i += 1
        blocks.append(('para', '\n'.join(buf)))
    return blocks

def build(source=SRC, destination=DST, figure_dir=FIG_DIR):
    doc = Document()
    # page setup A4, margins 2.5cm
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    # cover
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('DeepSeek-V4.1-Flash 技术报告')
    set_run_font(r, HEI, ARIAL, 22, True)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('中文翻译与深度解读')
    set_run_font(r, KAI, ARIAL, 16, False)
    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('原文：DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（DeepSeek-AI）')
    set_run_font(r, SONG, ARIAL, 11, False)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('翻译与解读整理：2026年9月')
    set_run_font(r, SONG, ARIAL, 11, False)
    doc.add_page_break()

    # TOC page
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run('目  录')
    set_run_font(r, HEI, ARIAL, 16, True)
    add_toc(doc)
    doc.add_page_break()

    # body
    blocks = parse_md(source)
    first_h1_done = False
    for kind, payload in blocks:
        if kind == 'hr':
            continue
        elif kind.startswith('h'):
            lvl = int(kind[1])
            text = payload.replace('**', '')
            # first H1 that is the document main title -> Title style, not in TOC
            if lvl == 1 and not first_h1_done and ('技术报告' in text or '深度解读' in text):
                first_h1_done = True
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(text)
                set_run_font(r, HEI, ARIAL, 18, True)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(12)
            else:
                add_heading(doc, text, lvl)
        elif kind == 'para':
            text = payload.strip()
            # figure caption -> insert image first, then caption
            mfig = re.match(r'^(\*\*)?图\s*(\d+)(\*\*)?\s*\|', text)
            if mfig:
                fname = os.path.join(figure_dir, 'figure-%02d.png' % int(mfig.group(2)))
                if os.path.exists(fname):
                    fp = doc.add_paragraph()
                    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    fp.paragraph_format.space_before = Pt(8)
                    fp.paragraph_format.space_after = Pt(2)
                    run = fp.add_run()
                    run.add_picture(fname, width=Cm(14.5))
                add_para(doc, text, size=10.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            elif text.startswith('**表') or '表 1 |' in text:
                add_para(doc, text, size=10.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
            else:
                add_para(doc, text)
        elif kind == 'formula':
            latex, annot, num = payload
            add_formula(doc, latex, annot, num)
        elif kind == 'bullet':
            add_bullet(doc, payload)
        elif kind == 'quote':
            add_quote(doc, payload)
        elif kind == 'code':
            add_code(doc, payload)
        elif kind == 'table':
            add_table(doc, payload)

    add_page_number(sec)

    # update fields on open so the TOC refreshes
    settings = doc.settings.element
    updateFields = OxmlElement('w:updateFields')
    updateFields.set(qn('w:val'), 'true')
    settings.append(updateFields)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.save(destination)
    print('saved:', destination)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SRC)
    parser.add_argument('--output', type=Path, default=DST)
    parser.add_argument('--figures', type=Path, default=FIG_DIR)
    args = parser.parse_args()
    build(args.source, args.output, args.figures)
