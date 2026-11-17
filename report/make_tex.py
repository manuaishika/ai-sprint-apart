"""Build a LaTeX version of the paper from report.md.

    python report/make_tex.py        # writes report/latex/main.tex (+ figures/) and compiles it

Needs pandoc and a LaTeX installation with pdflatex. The generated main.tex is plain
`article` class, so it can be dropped into Overleaf or arXiv as it is, or pasted into a
venue template.
"""
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "latex")
FIGS = os.path.join(HERE, "..", "results", "figures")

SUP = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
FIG_WIDTH = {"setup.png": "0.95", "rl_main.png": "0.82", "mechanisms.png": "0.82", "mind_vs_hands.png": "0.95"}

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,longtable,array,float,calc}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\urlstyle{same}
\Urlmuskip=0mu plus 1mu
\emergencystretch=2em
\setlist{itemsep=1pt,topsep=3pt,parsep=0pt}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\providecommand{\tightlist}{\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}}
\DeclareUnicodeCharacter{2212}{\ensuremath{-}}
\DeclareUnicodeCharacter{0394}{\ensuremath{\Delta}}
\DeclareUnicodeCharacter{03C0}{\ensuremath{\pi}}
\DeclareUnicodeCharacter{03B3}{\ensuremath{\gamma}}
\DeclareUnicodeCharacter{03B1}{\ensuremath{\alpha}}
\DeclareUnicodeCharacter{03B5}{\ensuremath{\varepsilon}}
\DeclareUnicodeCharacter{03B2}{\ensuremath{\beta}}
\DeclareUnicodeCharacter{03BC}{\ensuremath{\mu}}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{2082}{\ensuremath{_2}}
\renewcommand{\arraystretch}{1.15}
\setlength{\tabcolsep}{4pt}
\setkeys{Gin}{keepaspectratio}
\captionsetup{font=small} % replaced below if caption package is missing
"""


def split_abstract(md):
    a = md.index("## Abstract")
    b = md.index("## 1. Introduction")
    return md[a + len("## Abstract"):b].strip(), md[b:]


def prepare(md):
    # superscript digits, e.g. 10⁻⁵ -> $10^{-5}$
    def sup(m):
        return "$10^{" + "".join(SUP[c] for c in m.group(1)) + "}$"
    md = re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", sup, md)
    md = md.replace("log₂15", r"$\log_2 15$").replace("e^(−βt)", r"$e^{-\beta t}$")
    # subscripted symbols p_M, π_N, Δp_i(t) ...
    md = re.sub(r"(?<![\w$])([pπΔ])_(M|N|dev)\b", r"$\1_{\2}$", md)
    md = md.replace("corr(Δp_i(t), Δp_j(t+1))", r"$\mathrm{corr}(\Delta p_i(t), \Delta p_j(t{+}1))$")
    md = md.replace("I(true; observed)", r"$I(\text{true};\text{observed})$")
    md = md.replace("p_i(t-1)", r"$p_i(t{-}1)$")
    # headings: strip manual numbers, mark unnumbered sections
    md = re.sub(r"^## (\d+)\. ", "## ", md, flags=re.M)
    md = re.sub(r"^### (\d+\.\d+) ", "### ", md, flags=re.M)
    for name in ("Code and Data", "Author Contributions", "References"):
        md = md.replace(f"## {name}\n", f"## {name} {{-}}\n")
    if "## Appendix" in md:
        a = md.index("## Appendix")
        head, appendix = md[:a], md[a:]
        appendix = appendix.replace("## Appendix\n", "\n```{=latex}\n\\appendix\n```\n", 1)
        appendix = re.sub(r"^### ([A-C])\. ", "## ", appendix, flags=re.M)
        md = head + appendix
    # figures: image + italic caption -> pandoc figure with caption (LaTeX numbers them 1..n in order)
    def fig(m):
        path, cap = m.group(1), m.group(2).strip()
        cap = re.sub(r"^Figure \d+\.\s*", "", cap)
        name = os.path.basename(path)
        return f"![{cap}](figures/{name}){{width={float(FIG_WIDTH.get(name, '0.9')) * 100:.0f}%}}\n"
    md = re.sub(r"!\[\]\(([^)]+)\)\s*\n\s*\n\*(Figure \d+\..*?)\*\n", fig, md, flags=re.S)
    # table captions: keep as small italic lines above the table (labels like "Table A4" are manual)
    md = re.sub(r"^(\*Table [A\d]+\d?\..*?\*)$",
                lambda m: "```{=latex}\n\\begingroup\\small\\noindent\n```\n" + m.group(1) + "\n\n```{=latex}\n\\endgroup\n```\n",
                md, flags=re.M)
    return md


def main():
    md = open(os.path.join(HERE, "report.md")).read()
    md = md[md.index("## Abstract"):]
    abstract, body = split_abstract(md)
    body = prepare(body)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for f in FIG_WIDTH:
        shutil.copy(os.path.join(FIGS, f), os.path.join(OUT, "figures", f))
    tmp_md = os.path.join(OUT, "_body.md")
    open(tmp_md, "w").write(body)
    abs_md = os.path.join(OUT, "_abs.md")
    open(abs_md, "w").write(prepare(abstract))

    def pandoc(src, extra=()):
        return subprocess.run(["pandoc", src, "-t", "latex", "--wrap=preserve", "--columns=100",
                               "--shift-heading-level-by=-1", *extra],
                              check=True, capture_output=True, text=True).stdout

    body_tex = pandoc(tmp_md)
    abs_tex = pandoc(abs_md, ("--shift-heading-level-by=0",))
    # pandoc's longtable captions/columns: make tables use the small font
    body_tex = body_tex.replace("\\begin{longtable}", "\\begingroup\\small\n\\begin{longtable}").replace(
        "\\end{longtable}", "\\end{longtable}\n\\endgroup")
    title = open(os.path.join(HERE, "report.md")).read().split("\n")[1].split('"')[1]
    doc = (PREAMBLE.replace("\\captionsetup{font=small} % replaced below if caption package is missing\n",
                            "\\usepackage[font=small,labelfont=bf]{caption}\n")
           + "\n\\title{" + title.replace(":", ":\\\\[2pt]\\large") + "}\n"
           "% TODO: replace with your name and affiliation before submitting\n"
           "\\author{manuaishika\\\\ \\small Independent researcher}\n"
           "\\date{}\n\n\\begin{document}\n\\maketitle\n\n"
           "\\begin{abstract}\n" + abs_tex + "\n\\end{abstract}\n\n" + body_tex + "\n\\end{document}\n")
    with open(os.path.join(OUT, "main.tex"), "w") as f:
        f.write(doc)
    for p in (tmp_md, abs_md):
        os.remove(p)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=OUT,
                           capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("pdflatex failed")
    for ext in ("aux", "log", "out"):
        try:
            os.remove(os.path.join(OUT, f"main.{ext}"))
        except FileNotFoundError:
            pass
    print("built", os.path.join(OUT, "main.pdf"))


if __name__ == "__main__":
    main()
