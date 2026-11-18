from pathlib import Path

TEXT_FORMATS = {".txt", ".md", ".markdown"}


def read_document(path):
    path = Path(path)
    ext = path.suffix.lower()

    if ext in TEXT_FORMATS:
        # errors="replace": a stray non-UTF-8 byte should not abort the run.
        return path.read_text(encoding="utf-8", errors="replace")

    if ext == ".pdf":
        from pypdf import PdfReader

        pages = PdfReader(str(path)).pages
        return "\n\n".join(
            f"[PAGE {i + 1}]\n{page.extract_text() or ''}"
            for i, page in enumerate(pages)
        )

    if ext == ".docx":
        return _read_docx(path)

    raise ValueError("Supported formats: PDF, DOCX, TXT, MD")


def _read_docx(path):
    """Paragraphs and tables in document order (tables often hold approval
    matrices, thresholds and owners)."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks = []
    for element in document.element.body.iterchildren():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(element, document).text.strip()
            if text:
                blocks.append(text)
        elif tag == "tbl":
            rows = []
            for row in Table(element, document).rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("\n".join(rows))
    return "\n\n".join(blocks)
