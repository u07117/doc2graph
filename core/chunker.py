import re


def split_into_chunks(text, max_chars=12000, overlap_chars=800):
    """Split on paragraph boundaries into chunks of about max_chars.

    Each chunk after the first is prefixed with the last overlap_chars of the
    previous chunk, so content at a boundary is seen with its context.
    """
    if max_chars <= overlap_chars:
        raise ValueError("chunk size must exceed overlap")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        # Very long paragraphs (common in PDF text) are cut into pieces.
        # Overlap is added once below, not here as well.
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        candidate = (current + "\n\n" + paragraph).strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return [
        chunk if index == 0
        else chunks[index - 1][-overlap_chars:] + "\n\n" + chunk
        for index, chunk in enumerate(chunks)
    ]


def build_preview(chunks, budget_chars=18000, min_excerpt_chars=600):
    """A preview that spans the WHOLE document, not just its beginning.

    Short documents are returned in full. Longer ones become evenly spaced
    excerpts (the start of chunks across the document), so the classifier and
    planner see every part of a large document within the same budget.
    """
    full = "\n\n".join(chunks)
    if len(full) <= budget_chars:
        return full

    count = len(chunks)
    slots = max(1, min(count, budget_chars // min_excerpt_chars))
    picked = sorted({round(i * (count - 1) / max(1, slots - 1)) for i in range(slots)})
    share = budget_chars // len(picked)
    return "\n\n".join(
        f"[Excerpt from part {i + 1} of {count}]\n{chunks[i][:share]}"
        for i in picked
    )
