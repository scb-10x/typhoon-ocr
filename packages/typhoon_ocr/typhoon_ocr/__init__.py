"""
Typhoon OCR is a model for extracting structured markdown from images or PDFs.

This package provides utilities for document analysis, layout extraction, and OCR processing.
It focuses on structured text extraction with proper formatting and layout preservation.

Main Functions:
    - prepare_ocr_messages: Generate OCR-ready messages from PDFs or images
    - get_prompt: Access built-in prompt templates for different OCR tasks (default, structure, v1.5)
    - image_to_pdf: Convert image files to PDF format
    - ocr_document: End-to-end OCR processing with Typhoon API or local Hugging Face models
    - ocr_pdf_document: Multi-page hybrid PDF extraction (PyMuPDF vector fast-path + Typhoon VLM)
    - normalize_thai_text: Normalize Thai PUA glyphs (U+F700..U+F71A) and decomposed toned Sara-Am
    - extract_html_tables: Extract HTML <table> blocks (with rowspan/colspan) into 2D rows
"""
from .pdf_utils import pdf_utils_available, pymupdf_available
from .ocr_utils import (
    prepare_ocr_messages,
    get_prompt,
    get_anchor_text,
    image_to_pdf,
    ocr_document,
    ocr_pdf_document,
    ocr_local_transformers,
    unload_local_models,
    normalize_thai_text,
    extract_html_tables,
    html_table_to_markdown,
)

__version__ = "0.4.2"

__all__ = [
    "pdf_utils_available",
    "pymupdf_available",
    "prepare_ocr_messages",
    "get_prompt",
    "get_anchor_text",
    "image_to_pdf",
    "ocr_document",
    "ocr_pdf_document",
    "ocr_local_transformers",
    "unload_local_models",
    "normalize_thai_text",
    "extract_html_tables",
    "html_table_to_markdown",
]
