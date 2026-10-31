"""
Utility functions for Typhoon OCR.

This code is adapted from https://github.com/allenai/olmocr
Under the Apache 2.0 license.
Edited by Typhoon OCR Contributors & Upgraded with xocr Hybrid Engine enhancements.
"""
from dataclasses import dataclass
import gc
import html.parser
import json
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
import os
import re
import io
import tempfile
from PIL import Image
import subprocess
import base64
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import random
import ftfy
from pypdf.generic import RectangleObject
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Thai Glyph PUA & Decomposed Sara-Am Normalizer (Bug #3 Fix)
# ---------------------------------------------------------------------------
THAI_PUA_MAP = str.maketrans({
    "\uf700": "\u0e10",  # ฐ (lowless)
    "\uf701": "\u0e34",  # ิ (shifted left)
    "\uf702": "\u0e35",  # ี (shifted left)
    "\uf703": "\u0e36",  # ึ (shifted left)
    "\uf704": "\u0e37",  # ื (shifted left)
    "\uf705": "\u0e48",  # ่ (low-left)
    "\uf706": "\u0e49",  # ้ (low-left)
    "\uf707": "\u0e4a",  # ๊ (low-left)
    "\uf708": "\u0e4b",  # ๋ (low-left)
    "\uf709": "\u0e4c",  # ์ (low-left)
    "\uf70a": "\u0e48",  # ่ (low)
    "\uf70b": "\u0e49",  # ้ (low)
    "\uf70c": "\u0e4a",  # ๊ (low)
    "\uf70d": "\u0e4b",  # ๋ (low)
    "\uf70e": "\u0e4c",  # ์ (low)
    "\uf70f": "\u0e0d",  # ญ (lowless)
    "\uf710": "\u0e31",  # ั (shifted left)
    "\uf711": "\u0e4d",  # ํ (nikhahit shifted)
    "\uf712": "\u0e47",  # ็ (maitaikhu shifted)
    "\uf713": "\u0e48",  # ่ (high-left)
    "\uf714": "\u0e49",  # ้ (high-left)
    "\uf715": "\u0e4a",  # ๊ (high-left)
    "\uf716": "\u0e4b",  # ๋ (high-left)
    "\uf717": "\u0e4c",  # ์ (high-left)
    "\uf718": "\u0e38",  # ุ (low)
    "\uf719": "\u0e39",  # ู (low)
    "\uf71a": "\u0e3a",  # ฺ (phinthu low)
    "\u200b": "",        # Zero-width space
    "\ufeff": "",        # BOM
})

# Decomposed Sara-Am: nikhahit (\u0e4d) + optional tone mark + sara aa (\u0e32) -> tone mark + \u0e33
SARA_AM_PATTERN = re.compile(r"\u0e4d([\u0e48-\u0e4b]?)\u0e32")


def normalize_thai_text(text: str) -> str:
    """Normalize Thai Private Use Area (PUA) glyphs and decomposed toned Sara-Am."""
    if not text:
        return text
    text = text.translate(THAI_PUA_MAP)
    text = SARA_AM_PATTERN.sub(r"\g<1>" + "\u0e33", text)
    return text


# ---------------------------------------------------------------------------
# HTML <table> (rowspan/colspan) -> 2D Grid / Markdown Extractor
# ---------------------------------------------------------------------------
class _HTMLTableParser(html.parser.HTMLParser):
    """Parse HTML <table>...</table> blocks with full rowspan/colspan expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_cell_parts: List[str] = []
        self._current_rowspan = 1
        self._current_colspan = 1
        self._raw_rows: List[List[Tuple[str, int, int]]] = []
        self._current_raw_row: List[Tuple[str, int, int]] = []

    def _flush_cell(self) -> None:
        if self._in_cell:
            self._in_cell = False
            cell_text = re.sub(r"\s+", " ", "".join(self._current_cell_parts)).strip()
            cell_text = normalize_thai_text(cell_text)
            self._current_raw_row.append((cell_text, self._current_rowspan, self._current_colspan))
            self._current_cell_parts = []

    def _flush_row(self) -> None:
        self._flush_cell()
        if self._in_row:
            self._in_row = False
            if self._current_raw_row:
                self._raw_rows.append(self._current_raw_row)
            self._current_raw_row = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t == "table":
            if self._in_table:
                self._flush_row()
                grid = self._expand_grid(self._raw_rows)
                if grid:
                    self.tables.append(grid)
            self._in_table = True
            self._raw_rows = []
        elif self._in_table and t == "tr":
            self._flush_row()
            self._in_row = True
            self._current_raw_row = []
        elif self._in_row and t in ("td", "th"):
            self._flush_cell()
            self._in_cell = True
            self._current_cell_parts = []
            attr_dict = {k.lower(): (v or "1") for k, v in attrs}
            try:
                self._current_rowspan = max(1, int(attr_dict.get("rowspan", "1")))
            except ValueError:
                self._current_rowspan = 1
            try:
                self._current_colspan = max(1, int(attr_dict.get("colspan", "1")))
            except ValueError:
                self._current_colspan = 1
        elif self._in_cell and t == "br":
            self._current_cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._flush_cell()
        elif t == "tr" and self._in_row:
            self._flush_row()
        elif t == "table" and self._in_table:
            self._flush_row()
            self._in_table = False
            grid = self._expand_grid(self._raw_rows)
            if grid:
                self.tables.append(grid)

    @staticmethod
    def _expand_grid(raw_rows: List[List[Tuple[str, int, int]]]) -> List[List[str]]:
        if not raw_rows:
            return []
        occupied: Dict[Tuple[int, int], str] = {}
        max_col = 0
        num_rows = len(raw_rows)
        for r_idx, row in enumerate(raw_rows):
            c_idx = 0
            for text, rspan, cspan in row:
                while (r_idx, c_idx) in occupied:
                    c_idx += 1
                for dr in range(rspan):
                    for dc in range(cspan):
                        nr = r_idx + dr
                        nc = c_idx + dc
                        num_rows = max(num_rows, nr + 1)
                        max_col = max(max_col, nc + 1)
                        occupied[(nr, nc)] = text if (dr == 0 and dc == 0) else ""
                c_idx += cspan
        grid: List[List[str]] = []
        for r in range(num_rows):
            row_vals = [occupied.get((r, c), "") for c in range(max_col)]
            if any(v.strip() for v in row_vals):
                grid.append(row_vals)
        return grid


def extract_html_tables(text: str) -> List[List[List[str]]]:
    """Extract all HTML <table>...</table> blocks from OCR text into 2D row lists."""
    if not text or "<table" not in text.lower():
        return []
    parser = _HTMLTableParser()
    parser.feed(text)
    return parser.tables


def html_table_to_markdown(rows: List[List[str]]) -> str:
    """Convert a 2D table grid into a GitHub-Flavored Markdown (GFM) table."""
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    padded = [r + [""] * (col_count - len(r)) for r in rows]
    escaped = [[c.replace("|", "\\|").replace("\n", " ").strip() for c in r] for r in padded]
    header = "| " + " | ".join(escaped[0]) + " |"
    sep = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(r) + " |" for r in escaped[1:]]
    return "\n".join([header, sep] + body)


@dataclass(frozen=True)
class Element:
    pass


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @staticmethod
    def from_rectangle(rect: RectangleObject) -> "BoundingBox":
        return BoundingBox(rect[0], rect[1], rect[2], rect[3])


@dataclass(frozen=True)
class TextElement(Element):
    text: str
    x: float
    y: float


@dataclass(frozen=True)
class ImageElement(Element):
    name: str
    bbox: BoundingBox


@dataclass(frozen=True)
class PageReport:
    mediabox: BoundingBox
    text_elements: List[TextElement]
    image_elements: List[ImageElement]


def image_to_pdf(image_path):
    try:
        # Open the image file.
        img = Image.open(image_path)
        # Create a temporary file to store the PDF.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            filename = tmp.name
        # Convert image to RGB if necessary and save as PDF.
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(filename, "PDF")
        return filename
    except Exception:
        return None


def _can_use_pymupdf_for_path(
    local_pdf_path: str, pdf_backend: Literal["auto", "pymupdf", "poppler"] = "auto"
) -> bool:
    """Use PyMuPDF (fitz) when explicitly requested or when Poppler is unavailable on a real file."""
    from .pdf_utils import pdf_utils_available, pymupdf_available
    if pdf_backend == "poppler":
        return False
    if pdf_backend == "pymupdf":
        return pymupdf_available
    # In "auto" mode: if Poppler is missing, fall back to PyMuPDF when local_pdf_path exists on disk
    # (preserving ImportError on fake paths like '/test/file.pdf' in unit tests).
    if not pdf_utils_available and pymupdf_available and isinstance(local_pdf_path, str) and os.path.isfile(local_pdf_path):
        return True
    return False


def get_pdf_media_box_width_height(
    local_pdf_path: str,
    page_num: int,
    pdf_backend: Literal["auto", "pymupdf", "poppler"] = "auto",
) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using PyMuPDF (fitz) or pdfinfo.
    """
    if _can_use_pymupdf_for_path(local_pdf_path, pdf_backend=pdf_backend):
        import fitz
        with fitz.open(local_pdf_path) as doc:
            if page_num < 1 or page_num > len(doc):
                raise IndexError(f"Page number {page_num} out of range (1..{len(doc)})")
            rect = doc[page_num - 1].rect
            return float(abs(rect.width)), float(abs(rect.height))

    from .pdf_utils import pdf_utils_available
    if not pdf_utils_available:
        raise ImportError(
            "PDF utilities are not available. "
            "Installation instructions for Poppler utilities:\n"
            "- macOS: Run 'brew install poppler'\n"
            "- Ubuntu/Debian: Run 'apt-get install poppler-utils'\n"
            "- Windows: Install from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH"
        )

    # Construct the pdfinfo command to extract info for the specific page
    command = ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", "-enc", "UTF-8", local_pdf_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")

    output = result.stdout
    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")


def render_pdf_to_base64png(
    local_pdf_path: str,
    page_num: int,
    target_longest_image_dim: int = 2048,
    pdf_backend: Literal["auto", "pymupdf", "poppler"] = "auto",
) -> str:
    if _can_use_pymupdf_for_path(local_pdf_path, pdf_backend=pdf_backend):
        import fitz
        with fitz.open(local_pdf_path) as doc:
            if page_num < 1 or page_num > len(doc):
                raise IndexError(f"Page number {page_num} out of range (1..{len(doc)})")
            page = doc[page_num - 1]
            longest_dim = max(float(abs(page.rect.width)), float(abs(page.rect.height)), 1.0)
            scale = float(target_longest_image_dim) / longest_dim
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            png_bytes = pix.tobytes("png")
            return base64.b64encode(png_bytes).decode("utf-8")

    from .pdf_utils import pdf_utils_available
    if not pdf_utils_available:
        raise ImportError(
            "PDF utilities are not available. "
            "Installation instructions for Poppler utilities:\n"
            "- macOS: Run 'brew install poppler'\n"
            "- Ubuntu/Debian: Run 'apt-get install poppler-utils'\n"
            "- Windows: Install from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH"
        )

    longest_dim = max(get_pdf_media_box_width_height(local_pdf_path, page_num))

    # Convert PDF page to PNG using pdftoppm
    pdftoppm_result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f",
            str(page_num),
            "-l",
            str(page_num),
            "-r",
            str(target_longest_image_dim * 72 / longest_dim),  # 72 pixels per point is the conversion factor
            local_pdf_path,
        ],
        timeout=120,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pdftoppm_result.returncode == 0, pdftoppm_result.stderr
    return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")


def _linearize_pdf_report(report: PageReport, max_length: int = 4000) -> str:
    result = ""
    result += f"Page dimensions: {report.mediabox.x1:.1f}x{report.mediabox.y1:.1f}\n"

    if max_length < 20:
        return result

    images = _merge_image_elements(report.image_elements)

    # Process image elements
    image_strings = []
    for element in images:
        image_str = f"[Image {element.bbox.x0:.0f}x{element.bbox.y0:.0f} to {element.bbox.x1:.0f}x{element.bbox.y1:.0f}]\n"
        image_strings.append((element, image_str))

    # Process text elements
    text_strings = []
    for element in report.text_elements:  # type: ignore
        if len(element.text.strip()) == 0:  # type: ignore
            continue

        element_text = _cleanup_element_text(element.text)  # type: ignore
        text_str = f"[{element.x:.0f}x{element.y:.0f}]{element_text}\n"  # type: ignore
        text_strings.append((element, text_str))

    # Combine all elements with their positions for sorting
    all_elements: list[tuple[str, ImageElement, str, tuple[float, float]]] = []
    for elem, s in image_strings:
        position = (elem.bbox.x0, elem.bbox.y0)
        all_elements.append(("image", elem, s, position))
    for elem, s in text_strings:
        position = (elem.x, elem.y)  # type: ignore
        all_elements.append(("text", elem, s, position))

    # Calculate total length
    total_length = len(result) + sum(len(s) for _, _, s, _ in all_elements)

    if total_length <= max_length:
        for _, _, s, _ in all_elements:
            result += s
        return result

    # Identify elements with min/max coordinates
    edge_elements = set()

    if images:
        min_x0_image = min(images, key=lambda e: e.bbox.x0)
        max_x1_image = max(images, key=lambda e: e.bbox.x1)
        min_y0_image = min(images, key=lambda e: e.bbox.y0)
        max_y1_image = max(images, key=lambda e: e.bbox.y1)
        edge_elements.update([min_x0_image, max_x1_image, min_y0_image, max_y1_image])

    if report.text_elements:
        text_elements = [e for e in report.text_elements if len(e.text.strip()) > 0]
        if text_elements:
            min_x_text = min(text_elements, key=lambda e: e.x)
            max_x_text = max(text_elements, key=lambda e: e.x)
            min_y_text = min(text_elements, key=lambda e: e.y)
            max_y_text = max(text_elements, key=lambda e: e.y)
            edge_elements.update([min_x_text, max_x_text, min_y_text, max_y_text])  # type: ignore

    selected_element_ids = set()
    selected_elements = []

    for elem_type, elem, s, position in all_elements:
        if elem in edge_elements and id(elem) not in selected_element_ids:
            selected_elements.append((elem_type, elem, s, position))
            selected_element_ids.add(id(elem))

    current_length = len(result) + sum(len(s) for _, _, s, _ in selected_elements)
    _remaining_length = max_length - current_length

    remaining_elements = [(elem_type, elem, s, position) for elem_type, elem, s, position in all_elements if id(elem) not in selected_element_ids]

    random.shuffle(remaining_elements)

    for elem_type, elem, s, position in remaining_elements:
        if current_length + len(s) > max_length:
            break
        selected_elements.append((elem_type, elem, s, position))
        selected_element_ids.add(id(elem))
        current_length += len(s)

    selected_elements.sort(key=lambda x: (x[3][0], x[3][1]))

    for _, _, s, _ in selected_elements:
        result += s

    return result


def _cap_split_string(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    head_length = max_length // 2 - 3
    tail_length = head_length

    head = text[:head_length].rsplit(" ", 1)[0] or text[:head_length]
    tail = text[-tail_length:].split(" ", 1)[-1] or text[-tail_length:]

    return f"{head} ... {tail}"


def _cleanup_element_text(element_text: str) -> str:
    MAX_TEXT_ELEMENT_LENGTH = 250
    TEXT_REPLACEMENTS = {"[": "\\[", "]": "\\]", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
    text_replacement_pattern = re.compile("|".join(re.escape(key) for key in TEXT_REPLACEMENTS.keys()))

    element_text = ftfy.fix_text(element_text).strip()
    element_text = normalize_thai_text(element_text)

    # Replace square brackets with escaped brackets and other escaped chars
    element_text = text_replacement_pattern.sub(lambda match: TEXT_REPLACEMENTS[match.group(0)], element_text)

    return _cap_split_string(element_text, MAX_TEXT_ELEMENT_LENGTH)


def _merge_image_elements(images: List[ImageElement], tolerance: float = 0.5) -> List[ImageElement]:
    n = len(images)
    parent = list(range(n))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            parent_i = parent[i]
            parent[i] = root
            i = parent_i
        return root

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def bboxes_overlap(b1: BoundingBox, b2: BoundingBox, tolerance: float) -> bool:
        h_dist = max(0, max(b1.x0, b2.x0) - min(b1.x1, b2.x1))
        v_dist = max(0, max(b1.y0, b2.y0) - min(b1.y1, b2.y1))
        return h_dist <= tolerance and v_dist <= tolerance

    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(images[i].bbox, images[j].bbox, tolerance):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged_images = []
    for indices in groups.values():
        merged_bbox = images[indices[0]].bbox
        merged_name = images[indices[0]].name

        for idx in indices[1:]:
            bbox = images[idx].bbox
            merged_bbox = BoundingBox(
                x0=min(merged_bbox.x0, bbox.x0),
                y0=min(merged_bbox.y0, bbox.y0),
                x1=max(merged_bbox.x1, bbox.x1),
                y1=max(merged_bbox.y1, bbox.y1),
            )
            merged_name += f"+{images[idx].name}"

        merged_images.append(ImageElement(name=merged_name, bbox=merged_bbox))

    return merged_images


def _transform_point(x, y, m):
    x_new = m[0] * x + m[2] * y + m[4]
    y_new = m[1] * x + m[3] * y + m[5]
    return x_new, y_new


def _mult(m: List[float], n: List[float]) -> List[float]:
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


def _pdf_report(local_pdf_path: str, page_num: int, reader: Optional[PdfReader] = None) -> PageReport:
    if reader is None:
        reader = PdfReader(local_pdf_path)
    page = reader.pages[page_num - 1]
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    text_elements, image_elements = [], []

    def visitor_body(text, cm, tm, font_dict, font_size):
        txt2user = _mult(tm, cm)
        text_elements.append(TextElement(text, txt2user[4], txt2user[5]))

    def visitor_op(op, args, cm, tm):
        if op == b"Do":
            xobject_name = args[0]
            xobject = xobjects.get(xobject_name)
            if xobject and xobject["/Subtype"] == "/Image":
                _width = xobject.get("/Width")
                _height = xobject.get("/Height")
                x0, y0 = _transform_point(0, 0, cm)
                x1, y1 = _transform_point(1, 1, cm)
                image_elements.append(ImageElement(xobject_name, BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))

    page.extract_text(visitor_text=visitor_body, visitor_operand_before=visitor_op)

    return PageReport(
        mediabox=BoundingBox.from_rectangle(page.mediabox),
        text_elements=text_elements,
        image_elements=image_elements,
    )


def get_anchor_text(
    local_pdf_path: str,
    page: int,
    pdf_engine: Literal["pdftotext", "pdfium", "pypdf", "topcoherency", "pdfreport"],
    target_length: int = 4000,
    reader: Optional[PdfReader] = None,
) -> str:
    assert page > 0, "Pages are 1-indexed in pdf-land"

    if pdf_engine == "pdfreport":
        return _linearize_pdf_report(_pdf_report(local_pdf_path, page, reader=reader), max_length=target_length)
    else:
        raise NotImplementedError("Unknown engine")


PROMPTS_SYS = {
    "default": lambda base_text: (
        f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "structure": lambda base_text: (
        f"Below is an image of a document page, along with its dimensions and possibly some raw textual content previously extracted from it. "
        f"Note that the text extraction may be incomplete or partially missing. Carefully consider both the layout and any available text to reconstruct the document accurately.\n"
        f"Your task is to return the markdown representation of this document, presenting tables in HTML format as they naturally appear.\n"
        f"If the document contains images or figures, analyze them and include the tag <figure>IMAGE_ANALYSIS</figure> in the appropriate location.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    ),
    "v1.5": lambda base_text=None, figure_language="Thai": f"""Extract all text from the image.


Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.


Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:


<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in {figure_language}.
</figure>


- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.
    """,
}


def get_prompt(prompt_name: str) -> Callable[[str], str]:
    """Get a prompt template function for the specified prompt type."""
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")


def resize_if_needed(img: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    Downscale image only when width or height exceeds max_size.
    Fixes Bug #1 where images between 301px and max_size were erroneously upscaled to max_size.
    """
    width, height = img.size
    if width <= 0 or height <= 0 or max_size <= 0:
        return img
    if width > max_size or height > max_size:
        if width >= height:
            scale = max_size / float(width)
            new_size = (max_size, max(1, int(height * scale)))
        else:
            scale = max_size / float(height)
            new_size = (max(1, int(width * scale)), max_size)
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def image_to_base64png(img: Image.Image, image_format: str = "PNG") -> str:
    """
    Encode PIL Image into lossless PNG base64 string matching the data:image/png;base64 header.
    Fixes Bug #2 where JPEG (quality=75, 4:2:0 subsampling) was saved under a PNG MIME header.
    """
    buffered = io.BytesIO()
    img = img.convert("RGB")
    img.save(buffered, format=image_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_anchor_text_from_image(img: Image.Image):
    width = float(img.width)
    height = float(img.height)
    text = f"""Page dimensions: {width:.1f}x{height:.1f}\n[Image 0x0 to {width:.0f}x{height:.0f}]\n"""
    return text


def prepare_ocr_messages(
    pdf_or_image_path: str,
    task_type: str = "v1.5",
    target_image_dim: int = 1800,
    target_text_length: int = 8000,
    page_num: int = 1,
    figure_language: str = "Thai",
    pdf_backend: Literal["auto", "pymupdf", "poppler"] = "auto",
) -> List[Dict[str, Any]]:
    """
    Prepare messages for OCR processing from a PDF or image file.
    """
    ext = os.path.splitext(pdf_or_image_path)[1].lower()
    is_image = ext not in [".pdf"]
    filename = pdf_or_image_path

    try:
        if is_image:
            page_num = 1
            img = Image.open(pdf_or_image_path)
            if task_type == "v1.5":
                img = resize_if_needed(img, max_size=target_image_dim)
            image_base64 = image_to_base64png(img)
            if task_type != "v1.5":
                anchor_text = get_anchor_text_from_image(img)
        else:
            if page_num < 1:
                page_num = 1
            else:
                page_num = int(page_num)
            if pdf_backend == "auto":
                image_base64 = render_pdf_to_base64png(
                    filename, page_num, target_longest_image_dim=target_image_dim
                )
            else:
                image_base64 = render_pdf_to_base64png(
                    filename, page_num, target_longest_image_dim=target_image_dim, pdf_backend=pdf_backend
                )
            if task_type != "v1.5":
                anchor_text = get_anchor_text(
                    filename,
                    page_num,
                    pdf_engine="pdfreport",
                    target_length=target_text_length,
                )

        prompt_fn = get_prompt(task_type)
        if task_type == "v1.5":
            assert figure_language in ["Thai", "English"], "figure_language must be 'Thai' or 'English' for v1.5"
            prompt_text = prompt_fn(figure_language=figure_language)
        else:
            prompt_text = prompt_fn(anchor_text)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]
        return messages
    except IndexError:
        raise ValueError(f"Page number {page_num} is out of range for the document {pdf_or_image_path}")
    except Exception as e:
        raise ValueError(f"Error processing document: {str(e)}")


def is_base64_string(input_string: str) -> bool:
    try:
        return base64.b64encode(base64.b64decode(input_string))[:10] == input_string.encode()[:10]
    except Exception:
        return False


def ensure_image_in_path(input_string: str) -> str:
    """
    Detect whether the input is a base64-encoded image or a file path.
    """
    if input_string.endswith(".png") or input_string.endswith(".jpg") or input_string.endswith(".jpeg") or input_string.endswith(".pdf"):
        return input_string
    elif is_base64_string(input_string):
        try:
            image_data = base64.b64decode(input_string)
            image = Image.open(io.BytesIO(image_data))
            image_format = image.format.lower()
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format}")
            temp_file.close()
            image.save(temp_file.name, format=image_format)
            return temp_file.name
        except Exception:
            return input_string
    return input_string


def _extract_natural_text_json(text_output: str) -> str:
    """
    Robustly parse JSON containing `natural_text`, stripping Markdown ```json ... ``` fences if present.
    Fixes Bug #5.
    """
    if not text_output:
        return ""
    stripped = text_output.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "natural_text" in parsed:
            return str(parsed["natural_text"])
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "natural_text" in parsed:
                return str(parsed["natural_text"])
        raise
    return json.loads(text_output)["natural_text"]


# ---------------------------------------------------------------------------
# Local Hugging Face Transformers Backend (100% Offline, Sequential VRAM Safe)
# ---------------------------------------------------------------------------
_LOCAL_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def unload_local_models() -> None:
    """Explicitly release cached local VLM models and flush CUDA VRAM."""
    _LOCAL_MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def ocr_local_transformers(
    messages: List[Dict[str, Any]],
    model_id: str = "typhoon-ai/typhoon-ocr1.5-2b",
    max_new_tokens: int = 8192,
    repetition_penalty: float = 1.1,
    load_in_4bit: bool = False,
    keep_in_memory: bool = True,
) -> str:
    """
    Execute Typhoon-OCR v1.5 / v1 locally via Hugging Face `transformers` (`Qwen3VL` / `Qwen2_5_VL`)
    without requiring any API key or cloud connection.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cache_key = f"{model_id}|4bit={load_in_4bit}"
    if cache_key not in _LOCAL_MODEL_CACHE:
        unload_local_models()
        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "attn_implementation": "sdpa",
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["dtype"] = torch.bfloat16
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        _LOCAL_MODEL_CACHE[cache_key] = (model, processor)
    else:
        model, processor = _LOCAL_MODEL_CACHE[cache_key]

    try:
        # Convert OpenAI-style messages into Qwen-VL processor format with decoded PIL Image
        hf_messages = []
        for msg in messages:
            hf_content = []
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    hf_content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    b64_payload = url.split(",", 1)[1] if "," in url else url
                    pil_img = Image.open(io.BytesIO(base64.b64decode(b64_payload))).convert("RGB")
                    hf_content.append({"type": "image", "image": pil_img})
            hf_messages.append({"role": msg.get("role", "user"), "content": hf_content})

        inputs = processor.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
            )
        in_len = inputs["input_ids"].shape[1]
        out_ids = [out[in_len:] for out in generated_ids]
        decoded = processor.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return normalize_thai_text(decoded)
    finally:
        if not keep_in_memory:
            unload_local_models()


def ocr_document(
    pdf_or_image_path: str,
    task_type: str = "v1.5",
    target_image_dim: int = 1800,
    target_text_length: int = 8000,
    page_num: int = 1,
    base_url: str = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1"),
    api_key: str = None,
    model: str = "typhoon-ocr",
    figure_language: str = "Thai",
    backend: Literal["openai", "transformers", "local", "ollama"] = "openai",
    load_in_4bit: bool = False,
) -> str:
    """
    OCR a PDF or image file via OpenAI-compatible API, local Ollama/vLLM, or local Hugging Face transformers.
    Automatically cleans up temporary files created from base64 image inputs.
    """
    if "typhoon-ocr-preview" in model:
        assert task_type in ["default", "structure"], "task_type must be 'default' or 'structure' for typhoon-ocr-preview models"

    resolved_path = ensure_image_in_path(pdf_or_image_path)
    temp_created = (
        resolved_path != pdf_or_image_path
        and isinstance(resolved_path, str)
        and os.path.exists(resolved_path)
    )

    try:
        if backend in ("transformers", "local"):
            messages = prepare_ocr_messages(
                pdf_or_image_path=resolved_path,
                task_type=task_type,
                target_image_dim=target_image_dim,
                target_text_length=target_text_length,
                page_num=page_num if page_num else 1,
                figure_language=figure_language,
            )
            hf_model = (
                "typhoon-ai/typhoon-ocr1.5-2b"
                if model in ("typhoon-ocr", "typhoon-ocr1.5-2b", "scb10x/typhoon-ocr1.5-2b")
                else model
            )
            text_output = ocr_local_transformers(
                messages=messages,
                model_id=hf_model,
                repetition_penalty=1.1 if task_type == "v1.5" else 1.2,
                load_in_4bit=load_in_4bit,
            )
        else:
            effective_base_url = "http://localhost:11434/v1" if backend == "ollama" else base_url
            effective_api_key = (
                api_key
                or os.getenv("TYPHOON_OCR_API_KEY")
                or os.getenv("TYPHOON_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or ("ollama" if backend == "ollama" else None)
            )
            openai = OpenAI(base_url=effective_base_url, api_key=effective_api_key)
            messages = prepare_ocr_messages(
                pdf_or_image_path=resolved_path,
                task_type=task_type,
                target_image_dim=target_image_dim,
                target_text_length=target_text_length,
                page_num=page_num if page_num else 1,
                figure_language=figure_language,
            )
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=16384,
                extra_body={
                    "repetition_penalty": 1.1 if task_type == "v1.5" else 1.2,
                    "temperature": 0.1,
                    "top_p": 0.6,
                },
            )
            text_output = response.choices[0].message.content

        if task_type == "v1.5":
            return normalize_thai_text(text_output)
        else:
            return normalize_thai_text(_extract_natural_text_json(text_output))
    finally:
        if temp_created and os.path.exists(resolved_path):
            try:
                os.remove(resolved_path)
            except OSError:
                pass


def ocr_pdf_document(
    pdf_path: str,
    pages: Optional[List[int]] = None,
    hybrid_vector_fastpath: bool = True,
    min_vector_chars: int = 80,
    task_type: str = "v1.5",
    target_image_dim: int = 1800,
    figure_language: str = "Thai",
    backend: Literal["openai", "transformers", "local", "ollama"] = "openai",
    model: str = "typhoon-ocr",
    base_url: str = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1"),
    api_key: str = None,
) -> Dict[str, Any]:
    """
    Multi-page PDF processor with Hybrid Vector/Table Fast-Path (xocr Tier 1) + Typhoon-OCR VLM fallback.
    Eliminates the 'Blind VLM Tax' on digital PDF pages while routing scanned/complex pages to Typhoon-OCR.
    """
    import fitz

    results: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        page_numbers = pages or list(range(1, len(doc) + 1))
        for pno in page_numbers:
            if pno < 1 or pno > len(doc):
                continue
            page = doc[pno - 1]
            raw_text = normalize_thai_text(ftfy.fix_text(page.get_text("text") or "").strip())
            images_info = page.get_images(full=True)
            has_large_scan = len(images_info) > 0 and len(raw_text) < min_vector_chars

            if hybrid_vector_fastpath and len(raw_text) >= min_vector_chars and not has_large_scan:
                tables_md = []
                try:
                    found_tables = page.find_tables()
                    for t in found_tables.tables:
                        extracted = t.extract()
                        if extracted:
                            cleaned_rows = [
                                [normalize_thai_text(str(cell or "").strip()) for cell in row]
                                for row in extracted
                            ]
                            md_tbl = html_table_to_markdown(cleaned_rows)
                            if md_tbl:
                                tables_md.append(md_tbl)
                except Exception:
                    pass
                combined_md = raw_text
                if tables_md:
                    combined_md += "\n\n" + "\n\n".join(tables_md)
                results.append({
                    "page": pno,
                    "tier": "pymupdf_vector",
                    "markdown": combined_md,
                    "tables_count": len(tables_md),
                })
            else:
                vlm_md = ocr_document(
                    pdf_or_image_path=pdf_path,
                    task_type=task_type,
                    target_image_dim=target_image_dim,
                    page_num=pno,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    figure_language=figure_language,
                    backend=backend,
                )
                html_tables = extract_html_tables(vlm_md)
                results.append({
                    "page": pno,
                    "tier": f"typhoon_vlm_{backend}",
                    "markdown": vlm_md,
                    "tables_count": len(html_tables),
                    "extracted_tables": html_tables,
                })

    full_markdown = "\n\n---\n\n".join(r["markdown"] for r in results)
    return {
        "pdf_path": pdf_path,
        "total_pages": len(results),
        "pages": results,
        "markdown": full_markdown,
    }
