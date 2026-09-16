# -*- coding: utf-8 -*-
"""LaTeX -> MathML -> OMML converter for Word native equations.

Design: MathML <mrow>/<math> are element sequences; OMML m:oMath, m:e,
m:sub, m:sup etc. also accept element sequences, so no m:d (delimiter)
wrapper is needed unless a real delimiter (parentheses) appears.
"""
from lxml import etree

M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'

def _mk(tag, children=None, text=None):
    el = etree.Element('{%s}%s' % (M_NS, tag), nsmap={'m': M_NS})
    if text is not None:
        el.text = text
    if children:
        for c in children:
            if c is not None:
                el.append(c)
    return el

def _run(text, italic=False, bold=False):
    r = _mk('r')
    rPr = _mk('rPr')
    if italic:
        i = etree.SubElement(rPr, '{%s}i' % M_NS)
        i.set('{%s}val' % M_NS, '1')
    if bold:
        b = etree.SubElement(rPr, '{%s}b' % M_NS)
        b.set('{%s}val' % M_NS, '1')
    r.append(rPr)
    t = _mk('t', text=text)
    r.append(t)
    return r

def _lname(tag):
    return etree.QName(tag).localname

def _children(el):
    return [c for c in el if isinstance(c.tag, str)]

# --- delimiter detection ------------------------------------------------
DELIMS = {
    '(': ')', '[': ']', '{': '}', '|': '|',
    '\u2223': '\u2223',  # ∣
}
OPEN_DELIMS = set(DELIMS.keys())

def _is_open_mo(el):
    if _lname(el.tag) != 'mo':
        return False
    txt = ''.join(el.itertext()).strip()
    return txt in OPEN_DELIMS

def _is_close_mo(el):
    if _lname(el.tag) != 'mo':
        return False
    txt = ''.join(el.itertext()).strip()
    return txt in set(DELIMS.values())

# --- conversion ----------------------------------------------------------
def convert_node(el):
    """Return a LIST of OMML elements."""
    name = _lname(el.tag)
    kids = _children(el)

    if name in ('math', 'mstyle', 'merror', 'mphantom'):
        out = []
        for k in kids:
            out.extend(convert_node(k))
        return out

    if name == 'mrow':
        # if whole mrow is wrapped in delimiters (mfenced style), keep as-is;
        # otherwise just flatten. Latex2mathml emits <mrow><mo>(</mo>...<mo>)</mo></mrow>
        # only when \left( \right) or plain parens? It emits mrow with mo parens,
        # which render fine flattened.
        out = []
        for k in kids:
            out.extend(convert_node(k))
        return out

    if name in ('mi', 'mn', 'mo', 'mtext', 'ms'):
        txt = ''.join(el.itertext())
        if not txt:
            txt = ' '
        if name == 'mi':
            # multi-letter identifiers should be upright (e.g. arg, max, len)
            if len(txt) > 1 and txt.isalpha():
                return [_run(txt, italic=False)]
            return [_run(txt, italic=True)]
        if name == 'mo':
            # operators: bold if function-like names; keep plain otherwise
            return [_run(txt)]
        return [_run(txt)]

    if name == 'msub':
        e = convert_node(kids[0])
        sub = convert_node(kids[1])
        ss = _mk('sSub')
        ee = _mk('e')
        for x in e:
            ee.append(x)
        ss.append(ee)
        s = _mk('sub')
        for x in sub:
            s.append(x)
        ss.append(s)
        return [ss]

    if name == 'msup':
        e = convert_node(kids[0])
        sup = convert_node(kids[1])
        ss = _mk('sSup')
        ee = _mk('e')
        for x in e:
            ee.append(x)
        ss.append(ee)
        s = _mk('sup')
        for x in sup:
            s.append(x)
        ss.append(s)
        return [ss]

    if name == 'msubsup':
        e = convert_node(kids[0])
        sub = convert_node(kids[1])
        sup = convert_node(kids[2])
        ss = _mk('sSubSup')
        ee = _mk('e')
        for x in e:
            ee.append(x)
        ss.append(ee)
        s = _mk('sub')
        for x in sub:
            s.append(x)
        ss.append(s)
        s2 = _mk('sup')
        for x in sup:
            s2.append(x)
        ss.append(s2)
        return [ss]

    if name == 'mfrac':
        num = convert_node(kids[0])
        den = convert_node(kids[1])
        f = _mk('f')
        n = _mk('num')
        for x in num:
            n.append(x)
        f.append(n)
        d = _mk('den')
        for x in den:
            d.append(x)
        f.append(d)
        return [f]

    if name == 'msqrt':
        e = convert_node(kids[0])
        rad = _mk('rad')
        deg = _mk('deg')
        rad.append(deg)
        ee = _mk('e')
        for x in e:
            ee.append(x)
        rad.append(ee)
        return [rad]

    if name == 'mroot':
        e = convert_node(kids[0])
        deg = convert_node(kids[1])
        rad = _mk('rad')
        d = _mk('deg')
        for x in deg:
            d.append(x)
        rad.append(d)
        ee = _mk('e')
        for x in e:
            ee.append(x)
        rad.append(ee)
        return [rad]

    if name == 'munder':
        e = convert_node(kids[0])
        und = convert_node(kids[1])
        lim = _mk('limLow')
        ee = _mk('e')
        for x in e:
            ee.append(x)
        lim.append(ee)
        u = _mk('lim')
        for x in und:
            u.append(x)
        lim.append(u)
        return [lim]

    if name == 'mover':
        e = convert_node(kids[0])
        ov = convert_node(kids[1])
        # \hat{x}: over is a single char like ^ or a symbol; use sSup for hat
        ov_txt = ''.join(el.itertext())
        lim = _mk('limUpp')
        ee = _mk('e')
        for x in e:
            ee.append(x)
        lim.append(ee)
        u = _mk('lim')
        for x in ov:
            u.append(x)
        lim.append(u)
        return [lim]

    if name == 'munderover':
        e = convert_node(kids[0])
        und = convert_node(kids[1])
        ov = convert_node(kids[2])
        ss = _mk('sSubSup')
        ee = _mk('e')
        for x in e:
            ee.append(x)
        ss.append(ee)
        s = _mk('sub')
        for x in und:
            s.append(x)
        ss.append(s)
        s2 = _mk('sup')
        for x in ov:
            s2.append(x)
        ss.append(s2)
        return [ss]

    if name == 'mtable':
        m = _mk('m')
        for row in kids:
            mr = _mk('mr')
            for cell in _children(row):
                e = _mk('e')
                for inner in _children(cell):
                    e.extend(convert_node(inner))
                mr.append(e)
            m.append(mr)
        return [m]

    if name in ('mtr', 'mtd', 'mlabeledtr'):
        out = []
        for k in kids:
            out.extend(convert_node(k))
        return out

    if name == 'mfenced':
        # mfenced: delimiters as attributes or default parens
        out = []
        open_c = el.get('open', '(')
        close_c = el.get('close', ')')
        if open_c:
            out.append(_run(open_c))
        for k in kids:
            out.extend(convert_node(k))
        if close_c:
            out.append(_run(close_c))
        return out

    if name == 'mspace':
        return [_run(' ')]

    # fallback: flatten children
    out = []
    for k in kids:
        out.extend(convert_node(k))
    return out


def convert_mathml(root):
    om = _mk('oMath')
    for ch in _children(root):
        if _lname(ch.tag) == 'semantics':
            inner = None
            for sub in _children(ch):
                if _lname(sub.tag) == 'math':
                    inner = sub
                    break
            if inner is None:
                for sub in _children(ch):
                    if _lname(sub.tag) in ('mrow', 'mi', 'mn', 'mo', 'msub', 'msup', 'msubsup', 'mfrac', 'msqrt'):
                        inner = sub
                        break
            if inner is not None:
                for x in convert_node(inner):
                    om.append(x)
        else:
            for x in convert_node(ch):
                om.append(x)
    return om


def latex_to_omml_xml(latex):
    """Convert LaTeX string to OMML XML string (root <m:oMath>)."""
    import latex2mathml.converter
    mml = latex2mathml.converter.convert(latex)
    root = etree.fromstring(mml.encode('utf-8'))
    om = convert_mathml(root)
    s = etree.tostring(om, encoding='unicode')
    return s


if __name__ == '__main__':
    tests = [
        r'C_l = H_{L/2} W_{KV}^l,\quad Z_l = H_{L/2} W_Z^l,\quad l > L/2',
        r'X_{l+1} = B_l X_l + C_l F_l(A_l X_l),\quad (A_l, B_l, C_l) = H(X_l)',
        r'\Delta_t = \sqrt{n}\, U(K) = \sqrt{n}\, D_r \hat{G}_t D_c,\quad \frac{1}{n}\sum_j (\Delta_t)^2_{ij} \approx 1',
        r'k(b) = k_0 \cdot \exp\left(-(b - b_{\min})/\tau\right),\quad \tau = \lambda \Delta b',
        r'\ell^*_x(b) \in \arg\max_{\ell \ge 0}\left[p_x(\ell) - k(b)\cdot \ell / L_{\mathrm{norm}}\right]',
    ]
    for t in tests:
        try:
            s = latex_to_omml_xml(t)
            print('OK  :', s[:120])
        except Exception as e:
            print('FAIL:', t[:50], '->', e)
