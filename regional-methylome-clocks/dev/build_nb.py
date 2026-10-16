import nbformat, sys, os, re
HERE = os.path.dirname(os.path.abspath(__file__))

def build(body, out, title_md):
    src = open(os.path.join(HERE, body)).read()
    cells = [nbformat.v4.new_markdown_cell(title_md)]
    chunks = re.split(r"^# %%(.*)$", src, flags=re.M)
    # chunks: [pre, marker1, body1, marker2, body2, ...]
    for marker, text in zip(chunks[1::2], chunks[2::2]):
        marker = marker.strip(); text = text.strip("\n")
        if marker.startswith("[markdown]"):
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        elif marker.startswith("WRITEFILE"):
            fname = marker.split()[1]
            code = open(os.path.join(HERE, fname)).read()
            cells.append(nbformat.v4.new_code_cell(f"%%writefile {{CODE}}/{fname}\n" + code))
        elif text.strip():
            cells.append(nbformat.v4.new_code_cell(text))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbformat.write(nb, os.path.join(HERE, out))
    print("written", out, len(cells), "cells")

if __name__ == "__main__":
    body, out, title = sys.argv[1], sys.argv[2], open(os.path.join(HERE, sys.argv[3])).read()
    build(body, out, title)
