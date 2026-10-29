"""Unit tests for PDF citation extraction and text cleaning."""
from src.utils.pdf_extractor import clean_pdf_text, extract_citation_sentences


def test_clean_pdf_text():
    raw_sample = "This is a distilla-\ntion framework.\nIt evaluates citation   markers."
    cleaned = clean_pdf_text(raw_sample)
    assert "distillation" in cleaned
    assert "   " not in cleaned


def test_extract_citation_sentences_bracketed():
    sample_text = (
        "Transformer models have revolutionized NLP [12]. "
        "We utilize the Adam optimizer as implemented by Kingma et al. [14] with a learning rate of 2e-5. "
        "This is an un-cited baseline sentence that should not be extracted."
    )
    extracted = extract_citation_sentences(sample_text)
    assert len(extracted) == 2
    assert "[12]" in extracted[0]["markers"] or "[14]" in extracted[0]["markers"]


def test_extract_citation_sentences_author_year():
    sample_text = (
        "Previous approaches relied on static word embeddings (Mikolov et al., 2013). "
        "In contrast, our proposed architecture builds upon recent advances in self-attention (Vaswani et al., 2017). "
        "No citations appear in this unrelated concluding remark."
    )
    extracted = extract_citation_sentences(sample_text)
    assert len(extracted) == 2
    assert any("Mikolov" in m for m in extracted[0]["markers"] + extracted[1]["markers"])
