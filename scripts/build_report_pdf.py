"""Convert REPORT.md to REPORT.pdf (Markdown -> HTML -> headless Edge/Chrome print).

  python scripts/build_report_pdf.py [--md REPORT.md] [--pdf REPORT.pdf]
"""
import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-right { content: counter(page); font-family: "Times New Roman", Times, serif; font-size: 10pt; color: #555; } }
body { font-family: "Times New Roman", Times, serif; font-size: 11pt; line-height: 1.4;
       color: #111; max-width: 100%; }
h1 { font-size: 18pt; margin: 0 0 10pt 0; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt 0; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; }
h2, h3 { break-after: avoid; }
p, li { margin: 4pt 0; }
table { border-collapse: collapse; margin: 6pt 0 10pt 0; font-size: 9.8pt; break-inside: avoid; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 1px solid #c8c8c8; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.6pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6pt; white-space: pre-wrap; break-inside: avoid; }
p:has(> img) { text-align: center; margin: 8pt 0 2pt 0; break-inside: avoid; break-after: avoid; }
img { max-width: 100%; max-height: 228mm; }
p:has(> em:only-child) { font-size: 9.8pt; color: #444; margin-top: 0; break-before: avoid; }
"""


def find_browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    for name in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("no Edge/Chrome found for printing to PDF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="REPORT.md")
    ap.add_argument("--pdf", default="REPORT.pdf")
    a = ap.parse_args()

    md_path = Path(a.md).resolve()
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    title = md_path.stem
    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{md_path.parent.as_uri()}/"><style>{CSS}</style></head><body>{body}</body></html>')
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(html, encoding="utf-8")
        pdf_path = Path(a.pdf).resolve()
        pdf_path.unlink(missing_ok=True)
        # a separate profile forces a new browser process instead of handing off to a running one
        cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               "--allow-file-access-from-files", f"--user-data-dir={Path(tmp) / 'profile'}",
               "--virtual-time-budget=10000", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        subprocess.run(cmd, capture_output=True, timeout=180)
        for _ in range(120):
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                break
            time.sleep(0.5)
        else:
            raise SystemExit("browser did not produce the PDF")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
