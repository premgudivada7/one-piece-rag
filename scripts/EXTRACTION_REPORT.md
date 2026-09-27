# Extraction Report

## Team and Subject

- Team: One Piece
- Subject: Law & Regulation — AI Regulation and Risk Management
- Corpus language: English

## Source Documents

| Filename | Document | PDF pages |
|---|---|---:|
| eu_ai_act_2024.pdf | EU Artificial Intelligence Act, Regulation (EU) 2024/1689 | 144 |
| nist_ai_rmf_1_0.pdf | NIST Artificial Intelligence Risk Management Framework 1.0 | 48 |
| Total | | 192 |

The corpus combines EU legislation and NIST risk-management guidance.
These documents serve different purposes and should not be treated as
equivalent legal authorities.

## Environment and Method

- Python 3.12.13
- Environment managed with uv
- Extraction library: PyMuPDF 1.28.2
- Script: scripts/extract.py

The script processes PDFs in filename order and creates one entry per
PDF page. Each entry includes source, page, char_count, and text.

Page numbers refer to the PDF file's physical page sequence, starting
at 1. They may differ from page numbers printed inside the document.

## Text Cleanup

- Collapsed extra spaces and joined wrapped lines within text blocks.
- Preserved text-block boundaries as paragraph breaks.
- Removed known repeated headers, footers, and page labels in margins.
- Retained article numbers, section headings, and footnote content.
- Repaired split words when the complete or hyphenated spelling was
  found elsewhere in the source PDFs.
- Left uncertain word splits unchanged instead of guessing.
- Calculated char_count after cleaning.
- Saved JSON using UTF-8 with accented characters preserved.

## Latest Corpus Statistics

    source files: 2
    total entries: 192
    total characters: 691893
    avg characters/entry: 3603.61
    near-empty entries (<50 chars): 0

No empty or near-empty entries were reported in the latest run.
The script retains these entries if they occur.

## Validation Performed

Before the final word-cleanup revision, automated checks confirmed:

- All 192 pages were present in the expected order.
- Every entry contained the four required fields.
- Every char_count matched len(text).
- sample.json matched the first 10 entries of corpus.json.

Readability spot-checks included EU AI Act PDF pages 1 and 73,
and NIST PDF pages 6, 27, and 28.

These checks identified broken words in NIST paragraphs and a
reading-order limitation in its tables. The word-cleanup logic was
then revised. The latest run still reports 192 entries and no
near-empty entries.

## Known Limitations

- Some uncertain hyphenated word splits may remain.
- Tables are extracted as plain text, without their visual grid.
  On NIST PDF page 27, GOVERN 1 appeared after its subcategories
  in the inspected extraction. Table reordering was not implemented.
- Paragraphs spanning PDF pages remain in separate page entries.
- Image-only content and diagrams are not transcribed.
- OCR was not performed and is outside this assignment's scope.
- Readability checks were spot-checks, not a manual review of every page.
- Header/footer rules are tailored to these two PDFs.
- This script currently supports PDF inputs only.

## Output Files

- data/corpus.json: all 192 page entries
- data/sample.json: the first 10 entries

The data/ directory is excluded from Git. Source documents and JSON
outputs are shared through the team's private Teams channel.