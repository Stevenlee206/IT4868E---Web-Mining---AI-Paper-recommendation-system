"""Convert report/paperrec_research_paper.tex to Markdown with LaTeX math.

Handles only the commands used in that file: python scripts/paper_to_markdown.py
"""

import re
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "report"
DIAGRAM = """```mermaid
flowchart LR
    T[Title and abstract] --> E[SPECTER2 + adapter, frozen] --> H[H]
    H --> P((+))
    C[Citation graph A<br/>train papers send] --> AH[AH]
    H --> AH --> P
    P --> L[Linear layer, 768 to 128] --> S[Cosine with user profile]
```
"""


def inline(text):
    text = re.sub(r"\{\\scriptsize\$\\pm\$([\d.]+)\}", r"± \1", text)
    text = re.sub(r"\\(?:textbf)\{((?:[^{}]|\{[^{}]*\})*)\}", r"**\1**", text)
    text = re.sub(r"(?<!\$)\\texttt\{([^{}]*)\}(?!:)", r"`\1`", text)
    text = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", text)
    text = text.replace("$^\\dagger$", "†").replace("$\\dagger$", "†")
    for old, new in (("\\%", "%"), ("\\&", "&"), ("~", " "), ("---", "—"), ("--", "–"),
                     ("``", "“"), ("''", "”"), ("\\linewidth", ""), ("$-$", "−")):  # fmt: skip
        text = text.replace(old, new)
    return text


def table(match, caption, label, number):
    body = match
    body = re.sub(
        r"\\rows\{(\w+)\}", lambda m: (REPORT / "generated" / f"{m[1]}.tex").read_text(), body
    )
    inside = re.search(r"\\toprule(.*)\\bottomrule", body, re.DOTALL)[1]
    head, rows = inside.split(r"\midrule", 1)
    rows = re.sub(r"\\(addlinespace|midrule)\n?", "", rows)
    cells = [[c.strip() for c in r.split("&")] for r in re.split(r"\\\\", rows) if r.strip()]
    header = {
        "tab:paired": "Setting & Comparison & R@20 gap & 95% interval & Folds & NDCG@20 gap"
        " & 95% interval & Folds",
        "tab:link": "Model & No citations & Cited, no link to profile & One link"
        " & Two or more links",
    }.get(label, head.replace(r"\\", ""))
    header = [inline(c.strip()) for c in header.split("&")]
    lines = [f"**Table {number}.** {inline(caption)}", "", "| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    lines += ["| " + " | ".join(inline(c) for c in row) + " |" for row in cells]
    return "\n".join(lines)


def main():
    source = (REPORT / "paperrec_research_paper.tex").read_text()
    title = re.search(r"\\title\{\\textbf\{(.*?)\}\}", source)[1]
    author = re.search(r"\\author\{(.*?)\}", source)[1]
    body = source.split(r"\maketitle")[1].split(r"\end{document}")[0]
    body, bibliography = body.split(r"\begin{thebibliography}{99}\small")
    keys = re.findall(r"\\bibitem\{(\w+)\}", bibliography)
    entries = re.split(r"\\bibitem\{\w+\}", bibliography.split(r"\end{thebibliography}")[0])[1:]
    labels = {}
    for kind, name in (("table", "tab"), ("figure", "fig")):
        for number, found in enumerate(re.findall(rf"\\label\{{({name}:\w+)\}}", body), 1):
            labels[found] = number

    def floating(m):
        text = m[0]
        label = re.search(r"\\label\{(.*?)\}", text)[1]
        caption = re.search(r"\\caption\{(.*)\}\n\\label", text)[1]
        if label.startswith("tab"):
            return "\n" + table(text, caption, label, labels[label]) + "\n"
        picture = DIAGRAM if label == "fig:architecture" else figure_table()
        return f"\n{picture}\n**Figure {labels[label]}.** {inline(caption)}\n"

    body = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", floating, body, flags=re.DOTALL)
    body = body.replace(r"\begin{abstract}", "## Abstract\n").replace(r"\end{abstract}", "")
    body = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\n$$\1$$\n", body, flags=re.DOTALL
    )
    body = re.sub(
        r"\\begin\{align\}(.*?)\\end\{align\}",
        r"\n$$\n\\begin{aligned}\1\\end{aligned}\n$$\n",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"\\citep\{(.*?)\}",
        lambda m: "[" + ", ".join(str(keys.index(k) + 1) for k in m[1].split(",")) + "]",
        body,
    )
    body = re.sub(r"\\ref\{(.*?)\}", lambda m: str(labels[m[1]]), body)
    body = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", body)
    body = body.replace(r"\item ", "- ")
    numbers = [0, 0]

    def section(m):
        if m[1] == "sub":
            numbers[1] += 1
            return f"### {numbers[0]}.{numbers[1]} {m[2]}"
        numbers[0], numbers[1] = numbers[0] + 1, 0
        return f"## {numbers[0]}. {m[2]}"

    body = re.sub(r"\\(sub)?section\{(.*?)\}", section, body)
    body = re.sub(r"\\paragraph\{((?:[^{}]|\{[^{}]*\})*)\}\s*", r"**\1** ", body)
    parts = re.split(r"(\$\$.*?\$\$|```.*?```|^\|[^\n]*$)", body, flags=re.DOTALL | re.MULTILINE)
    body = "".join(p if i % 2 else inline(p) for i, p in enumerate(parts))
    references = [f"{i}. {inline(e.strip())}" for i, e in enumerate(entries, 1)]
    text = f"# {title}\n\n*{author}*\n\n{body.strip()}\n\n## References\n\n" + "\n".join(references)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    left = re.findall(r"\\(?:begin\{(?!aligned)|rows|input|label|cite|ref)\S*", text)
    if left:
        raise ValueError(f"Unconverted LaTeX: {left}")
    (REPORT / "paperrec_research_paper.md").write_text(text + "\n", encoding="utf-8")


def figure_table():
    rows = (REPORT / "generated" / "cold_degree.dat").read_text().splitlines()[1:]
    lines = ["| Citations | No graph | Plain GCN | Residual GCN |", "|---|---|---|---|"]
    for row in rows:
        cells = row.replace("{", "").replace("}", "").replace("--", "–").split()
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
