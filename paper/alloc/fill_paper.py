"""Regenerate the allocation-study rows of Table tab:alloc inside main_iclr.tex (between %ALLOC-ROWS-BEGIN/END)."""
import contextlib
import io
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze  # noqa: E402

TEX = pathlib.Path(__file__).resolve().parents[1] / "manuscript" / "iclr2027" / "main_iclr.tex"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    analyze.latex()
s = TEX.read_text()
s = re.sub(r"%ALLOC-ROWS-BEGIN\n.*?%ALLOC-ROWS-END", lambda m: "%ALLOC-ROWS-BEGIN\n" + buf.getvalue() + "%ALLOC-ROWS-END", s,
           flags=re.S)
TEX.write_text(s)
print(buf.getvalue())
