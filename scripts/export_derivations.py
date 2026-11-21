"""Export notebook mathematics as plain Markdown with a source-file manifest."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ("backprop_derivation.ipynb", "Backpropagation",
     ["forward equations", "chain rule", "hand calculations", "tensor shapes"]),
    ("probability_loss.ipynb", "Probability and loss",
     ["Gaussian MLE to MSE", "Bernoulli MLE to BCE", "categorical MLE to cross-entropy"]),
)


def cell_text(value):
    """Normalize notebook text stored as either a string or a list of lines."""
    return "".join(value) if isinstance(value, list) else value


def github_math(markdown):
    """Make notebook LaTeX survive GitHub's Markdown-before-math rendering.

    GitHub processes Markdown escapes before rendering math, so ``\\{``,
    ``\\;`` and ``\\\\`` lose a backslash, and a ``$$`` block directly under a
    text line is not recognised as math. Display math therefore becomes a
    fenced ``math`` block with blank lines around it, and inline math that
    contains a backslash before punctuation uses GitHub's $`...`$ form.
    """
    def fence(match):
        body = match.group(1).strip()
        return "\n\n```math\n" + body + "\n```\n\n"
    markdown = re.sub(r"\$\$(.+?)\$\$", fence, markdown, flags=re.S)

    def inline(match):
        body = match.group(1)
        if re.search(r"\\[^A-Za-z]", body):
            return "$`" + body + "`$"
        return match.group(0)
    parts = re.split(r"(```math\n.*?\n```)", markdown, flags=re.S)
    parts = [part if part.startswith("```math") else
             re.sub(r"(?<![\\$`])\$([^$\n`]+?)\$(?!`)", inline, part)
             for part in parts]
    return re.sub(r"\n{3,}", "\n\n", "".join(parts))


def export_derivations():
    """Preserve all mathematical Markdown and comparison tables without images.

    The notebooks remain the executable sources. This companion is readable
    using ordinary text/Markdown tools and requires no notebook renderer.
    """
    lines = ["# Mathematical derivations", "",
             "역전파와 손실 함수의 수학적 유도를 노트북과 같은 내용으로 정리한다.",
             "수식의 원본과 실행 코드는 아래 연결된 노트북에서 확인할 수 있다.", ""]
    documents = []
    for filename, title, topics in NOTEBOOKS:
        relative = "notebooks/" + filename
        nb = json.loads((ROOT/relative).read_text(encoding="utf-8"))
        code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        documents.append(dict(
            path=relative, format="ipynb", topics=topics,
            markdown_companion="DERIVATIONS.md",
            code_cells=len(code),
            executed_code_cells=sum(c.get("execution_count") is not None for c in code)))
        lines.extend([f"## {title}", "", f"원본: [{relative}]({relative})", ""])
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                for line in cell_text(cell["source"]).splitlines():
                    # Nest original notebook headings under this file's section.
                    lines.append("##" + line if line.startswith("#") else line)
                lines.append("")
            else:
                for output in cell.get("outputs", []):
                    markdown = output.get("data", {}).get("text/markdown")
                    if markdown:
                        lines.extend([cell_text(markdown), ""])
    lines.extend([
        "## Source files", "",
        "- [역전파 구현](src/backprop.py)",
        "- [확률 분포 및 손실 관련 함수](src/probability.py)",
        "- [고유값의 직접 계산 및 참조 비교](src/linear_algebra.py)",
        "- [전체 수치 검증 실행](scripts/run_all.py)", "",
    ])
    text = github_math("\n".join(lines))
    (ROOT/"DERIVATIONS.md").write_text(text.rstrip()+"\n", encoding="utf-8")
    return documents


if __name__ == "__main__":
    for document in export_derivations():
        print(f'{document["path"]}: {document["executed_code_cells"]}/{document["code_cells"]} code cells executed')
    print("Saved DERIVATIONS.md")
