"""PDF text extraction and scientific citation sentence identification utilities."""
import re
from typing import Any, Dict, List, Optional, Tuple
import pypdf


def extract_text_from_pdf(pdf_file) -> Tuple[str, Dict[str, Any]]:
    """Extract raw text from a PDF file object or file path.

    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    metadata = {
        "page_count": 0,
        "character_count": 0,
        "word_count": 0,
        "is_scanned": False,
        "error": None,
    }

    try:
        reader = pypdf.PdfReader(pdf_file)
        metadata["page_count"] = len(reader.pages)

        pages_text = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        metadata["character_count"] = len(full_text)
        metadata["word_count"] = len(full_text.split())

        # If minimal extractable text is found across pages, it's likely a scanned image
        if metadata["word_count"] < 30 and metadata["page_count"] > 0:
            metadata["is_scanned"] = True

        return full_text, metadata

    except Exception as e:
        metadata["error"] = str(e)
        return "", metadata


def clean_pdf_text(text: str) -> str:
    """Normalize whitespace and fix line-wrap hyphenation common in academic PDFs."""
    if not text:
        return ""

    # Remove de-hyphenation across linebreaks (e.g. "distilla-\ntion" -> "distillation")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Replace single line breaks with spaces, preserving paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Normalize multiple whitespace characters to single spaces
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


# Regex patterns for detecting scientific citations in academic text
CITATION_PATTERNS = [
    # Bracketed numbers: [1], [1, 2], [3-5], [12, 14, 18]
    re.compile(r'\[(?:\d+(?:[,\s–-]+\d+)*)\]'),
    # Author-Year in parentheses: (Smith et al., 2020), (Devlin et al., 2019; Vaswani et al., 2017)
    re.compile(r'\((?:[A-Z][a-zA-Z\s\.\-]+(?:et al\.)?,?\s+(?:19|20)\d{2}[a-z]?(?:;[^\)]+)?)\)'),
    # In-text named citations: Devlin et al. (2019), Vaswani et al. [14]
    re.compile(r'\b[A-Z][a-zA-Z]+\s+et\s+al\.\s+(?:\((?:19|20)\d{2}\)|\[\d+\])'),
]


def extract_citation_sentences(text: str, max_sentences: int = 100) -> List[Dict[str, Any]]:
    """Segment cleaned text into sentences and identify those containing in-text scientific citations.

    Returns:
        List of dicts with:
            - sentence: the raw sentence string
            - markers: list of detected citation markers
            - word_count: sentence length
    """
    cleaned = clean_pdf_text(text)
    if not cleaned:
        return []

    # Regex sentence boundary splitter preserving common scientific abbreviations
    # Negative lookbehinds for et al., e.g., i.e., Fig., Ref., Dr., vs., etc.
    abbrev_pattern = (
        r'(?<!\bet al)'
        r'(?<!\be\.g)'
        r'(?<!\bi\.e)'
        r'(?<!\bFig)'
        r'(?<!\bRef)'
        r'(?<!\bTab)'
        r'(?<!\bDr)'
        r'(?<!\bProf)'
        r'(?<!\bvs)'
        r'(?<!\bEq)'
        r'\.\s+(?=[A-Z\[\(])'
    )

    raw_sentences = re.split(abbrev_pattern, cleaned)

    citation_sentences = []
    seen = set()

    for idx, s in enumerate(raw_sentences):
        s_clean = s.strip()
        # Enforce reasonable length filters for a scientific sentence
        if len(s_clean) < 35 or len(s_clean) > 600:
            continue

        # Skip headers, references list entries, or copyright footers
        lower = s_clean.lower()
        if lower.startswith("references") or lower.startswith("bibliography") or "all rights reserved" in lower:
            continue
        # Avoid typical reference list lines: e.g., "[1] A. Author, Title, Conference, 2020."
        if re.match(r'^\[\d+\]\s+[A-Z]\.', s_clean):
            continue

        # Detect citation markers
        found_markers = []
        for pat in CITATION_PATTERNS:
            matches = pat.findall(s_clean)
            if matches:
                found_markers.extend(matches)

        if found_markers:
            # Deduplicate similar sentences
            s_key = s_clean[:80]
            if s_key not in seen:
                seen.add(s_key)
                citation_sentences.append({
                    "id": len(citation_sentences) + 1,
                    "sentence": s_clean if s_clean.endswith(".") else s_clean + ".",
                    "markers": list(set(found_markers)),
                    "word_count": len(s_clean.split()),
                })

        if len(citation_sentences) >= max_sentences:
            break

    return citation_sentences
