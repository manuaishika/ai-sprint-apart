"""Build report.pdf (pandoc -> HTML -> Chromium) and report.docx from report.md.

    python report/build.py
"""
import glob
import os
import re
import subprocess

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
WIDTH = {"detection_roc.png": "55%", "setup.png": "92%", "rl_main.png": "88%"}


def chromium():
    hits = glob.glob("/opt/pw-browsers/chromium_headless_shell-*/chrome-linux/headless_shell")
    return hits[0] if hits else None


def main():
    md = open(os.path.join(HERE, "report.md")).read()
    md = re.sub(r"!\[\]\(([^)]+?([^/)]+\.png))\)",
                lambda m: f"![]({m.group(1)}){{width={WIDTH.get(m.group(2), '100%')}}}", md)
    tmp = os.path.join(HERE, "_print.md")
    open(tmp, "w").write(md)
    html = os.path.join(HERE, "_report.html")
    subprocess.run(["pandoc", tmp, "-s", "--embed-resources", f"--resource-path={HERE}:{HERE}/..",
                    "--css", os.path.join(HERE, "print.css"), "-o", html], check=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium())
        pg = b.new_page()
        pg.goto("file://" + html)
        pg.pdf(path=os.path.join(HERE, "report.pdf"), format="A4", prefer_css_page_size=True)
        b.close()
    subprocess.run(["pandoc", os.path.join(HERE, "report.md"), "-o", os.path.join(HERE, "report.docx"),
                    f"--resource-path={HERE}:{HERE}/.."], check=True)
    os.remove(tmp)
    os.remove(html)
    print("built report.pdf and report.docx")


if __name__ == "__main__":
    main()
