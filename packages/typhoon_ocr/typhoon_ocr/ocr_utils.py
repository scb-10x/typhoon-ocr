"""
Utility functions for Typhoon OCR.

This code is adapted from https://github.com/allenai/olmocr
Under the Apache 2.0 license.
Edited by Typhoon OCR Contributors.
"""
from dataclasses import dataclass
import json
from openai import OpenAI
import os
import re
import io
import tempfile
from PIL import Image
import subprocess
import base64
from typing import Any, Callable, Dict, List, Literal
import random
import ftfy
from pypdf.generic import RectangleObject
from pypdf import PdfReader


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
            temp_pdf_created = True
        # Convert image to RGB if necessary and save as PDF.
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(filename, "PDF")
        return filename
    except Exception as conv_err:
        return None

def get_pdf_media_box_width_height(local_pdf_path: str, page_num: int) -> tuple[float, float]:
    """
    Get the MediaBox dimensions for a specific page in a PDF file using the pdfinfo command.

    :param pdf_file: Path to the PDF file
    :param page_num: The page number for which to extract MediaBox dimensions
    :return: A dictionary containing MediaBox dimensions or None if not found
    """
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
    # Run the command using subprocess
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if there is any error in executing the command
    if result.returncode != 0:
        raise ValueError(f"Error running pdfinfo: {result.stderr}")

    # Parse the output to find MediaBox
    output = result.stdout

    for line in output.splitlines():
        if "MediaBox" in line:
            media_box_str: List[str] = line.split(":")[1].strip().split()
            media_box: List[float] = [float(x) for x in media_box_str]
            return abs(media_box[0] - media_box[2]), abs(media_box[3] - media_box[1])

    raise ValueError("MediaBox not found in the PDF info.")
    
def render_pdf_to_base64png(local_pdf_path: str, page_num: int, target_longest_image_dim: int = 2048) -> str:
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
        # Use element's unique identifier (e.g., id or position) for comparison
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
        # Include all elements
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

    # Keep track of element IDs to prevent duplication
    selected_element_ids = set()
    selected_elements = []

    # Include edge elements first
    for elem_type, elem, s, position in all_elements:
        if elem in edge_elements and id(elem) not in selected_element_ids:
            selected_elements.append((elem_type, elem, s, position))
            selected_element_ids.add(id(elem))

    # Calculate remaining length
    current_length = len(result) + sum(len(s) for _, _, s, _ in selected_elements)
    _remaining_length = max_length - current_length

    # Exclude edge elements from the pool
    remaining_elements = [(elem_type, elem, s, position) for elem_type, elem, s, position in all_elements if id(elem) not in selected_element_ids]

    # Sort remaining elements by their positions (e.g., x-coordinate and then y-coordinate)
    # remaining_elements.sort(key=lambda x: (x[3][0], x[3][1]))

    # Shuffle remaining elements randomly
    random.shuffle(remaining_elements)

    # Add elements until reaching max_length
    for elem_type, elem, s, position in remaining_elements:
        if current_length + len(s) > max_length:
            break
        selected_elements.append((elem_type, elem, s, position))
        selected_element_ids.add(id(elem))
        current_length += len(s)

    # Sort selected elements by their positions to maintain logical order
    selected_elements.sort(key=lambda x: (x[3][0], x[3][1]))

    # Build the final result
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

    # Replace square brackets with escaped brackets and other escaped chars
    element_text = text_replacement_pattern.sub(lambda match: TEXT_REPLACEMENTS[match.group(0)], element_text)

    return _cap_split_string(element_text, MAX_TEXT_ELEMENT_LENGTH)

def _merge_image_elements(images: List[ImageElement], tolerance: float = 0.5) -> List[ImageElement]:
    n = len(images)
    parent = list(range(n))  # Initialize Union-Find parent pointers

    def find(i):
        # Find with path compression
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            parent_i = parent[i]
            parent[i] = root
            i = parent_i
        return root

    def union(i, j):
        # Union by attaching root of one tree to another
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def bboxes_overlap(b1: BoundingBox, b2: BoundingBox, tolerance: float) -> bool:
        # Compute horizontal and vertical distances between boxes
        h_dist = max(0, max(b1.x0, b2.x0) - min(b1.x1, b2.x1))
        v_dist = max(0, max(b1.y0, b2.y0) - min(b1.y1, b2.y1))
        # Check if distances are within tolerance
        return h_dist <= tolerance and v_dist <= tolerance

    # Union overlapping images
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(images[i].bbox, images[j].bbox, tolerance):
                union(i, j)

    # Group images by their root parent
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Merge images in the same group
    merged_images = []
    for indices in groups.values():
        # Initialize merged bounding box
        merged_bbox = images[indices[0]].bbox
        merged_name = images[indices[0]].name

        for idx in indices[1:]:
            bbox = images[idx].bbox
            # Expand merged_bbox to include the current bbox
            merged_bbox = BoundingBox(
                x0=min(merged_bbox.x0, bbox.x0),
                y0=min(merged_bbox.y0, bbox.y0),
                x1=max(merged_bbox.x1, bbox.x1),
                y1=max(merged_bbox.y1, bbox.y1),
            )
            # Optionally, update the name
            merged_name += f"+{images[idx].name}"

        merged_images.append(ImageElement(name=merged_name, bbox=merged_bbox))

    # Return the merged images along with other elements
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
    
def _pdf_report(local_pdf_path: str, page_num: int) -> PageReport:
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
                # Compute image bbox
                # The image is placed according to the CTM
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
    local_pdf_path: str, page: int, pdf_engine: Literal["pdftotext", "pdfium", "pypdf", "topcoherency", "pdfreport"], target_length: int = 4000
) -> str:
    assert page > 0, "Pages are 1-indexed in pdf-land"

    
    if pdf_engine == "pdfreport":
        return _linearize_pdf_report(_pdf_report(local_pdf_path, page), max_length=target_length)
    else:
        raise NotImplementedError("Unknown engine")

PROMPTS_SYS = {
    "default": lambda base_text: (f"Below is an image of a document page along with its dimensions. "
        f"Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        f"If the document contains images, use a placeholder like dummy.png for each image.\n"
        f"Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"),
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
    """
    Get a prompt template function for the specified prompt type.
    
    This function returns a callable that generates a prompt template based on the provided prompt name.
    The returned function takes extracted text as input and returns a formatted prompt string
    that can be used with OCR/vision models.
    
    Available prompt types:
    - "default": Creates a prompt for extracting text with tables in markdown format.
    - "structure": Creates a prompt for extracting text with tables in HTML format and image analysis.
    - "v1.5": OCR v1.5 prompt that doesn't require anchor text, uses clean Markdown with HTML tables and Thai figure descriptions.
    
    Args:
        prompt_name (str): The identifier for the desired prompt template ("default", "structure", or "v1.5").
        
    Returns:
        Callable[[str], str]: A function that takes extracted text and returns a formatted prompt.
        
    Examples:
        >>> prompt_fn = get_prompt("default")
        >>> formatted_prompt = prompt_fn("Sample extracted text")
        >>> print(formatted_prompt[:50])  # Print first 50 chars
        Below is an image of a document page along with its
    """
    return PROMPTS_SYS.get(prompt_name, lambda x: "Invalid PROMPT_NAME provided.")

def resize_if_needed(img: Image.Image, max_size: int = 2048) -> Image.Image:
    """
    Downscale image only when width or height exceeds max_size.
    Preserves aspect ratio and prevents unwanted upscaling of smaller images.

    Args:
        img: PIL Image to resize
        max_size: Maximum size for the longest dimension

    Returns:
        Resized image if dimension exceeds max_size, otherwise original image
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

        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def image_to_base64png(img: Image.Image, image_format: str = "PNG") -> str:
    buffered = io.BytesIO()
    if img.mode != "RGB":
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
) -> List[Dict[str, Any]]:
    """
    Prepare messages for OCR processing from a PDF or image file.
    
    This function provides an end-to-end workflow that combines multiple processing steps
    into a single call, creating messages ready for OCR processing with language models.
    It handles both image and PDF inputs, with appropriate page selection for PDFs.
    
    Processing Steps:
    1. Convert image to PDF if necessary (images are always treated as single pages)
    2. Render the selected PDF page to base64 PNG
    3. Extract anchor text from the page with position information (not needed for v1.5)
    4. Apply appropriate prompt template based on task type
    5. Create a messages structure ready for LLM API submission
    
    Args:
        pdf_or_image_path (str): Path to a PDF or image file to process
        task_type (str): Type of OCR task - "default" for standard markdown extraction,
                         "structure" for enhanced layout analysis with HTML tables,
                         "v1.5" for OCR v1.5 with clean Markdown and Thai figure descriptions
        target_image_dim (int): Target longest dimension for the rendered image in pixels
        target_text_length (int): Maximum length of extracted text to include (not used for v1.5)
        page_num (int): Page number to process (default=1, for images always 1)
        figure_language (str): Language for figure descriptions in v1.5 (default: "Thai")
        
    Returns:
        List[Dict[str, Any]]: Messages structure ready for OCR processing with an LLM API,
                             containing both text prompt and image data
    
    Raises:
        ValueError: If image conversion fails, page number is out of range, or other processing errors occur
        
    Examples:
        >>> # Process the first page of a PDF
        >>> messages = prepare_ocr_messages("document.pdf")
        >>> 
        >>> # Process page 5 of a PDF with structure analysis
        >>> messages = prepare_ocr_messages(
        ...     pdf_or_image_path="multipage.pdf",
        ...     task_type="structure",
        ...     page_num=5
        ... )
        >>> 
        >>> # Process an image file (always page 1)
        >>> messages = prepare_ocr_messages("scan.jpg")
    """
    # Check for required PDF utilities
    ext = os.path.splitext(pdf_or_image_path)[1].lower()
    is_image = ext not in [".pdf"]
    
    # Determine if the file is a PDF or image
    filename = pdf_or_image_path
    
    try:
        if is_image:
            page_num = 1
            img = Image.open(pdf_or_image_path)
            # For v1.5, use different resize logic
            if task_type == "v1.5":
                img = resize_if_needed(img, max_size=target_image_dim)
            # Render the image to base64 PNG
            image_base64 = image_to_base64png(img)
            # Get anchor text from the image (not needed for v1.5)
            if task_type != "v1.5":
                anchor_text = get_anchor_text_from_image(img)
        else:
            if page_num < 1:
                page_num = 1
            else:
                page_num = int(page_num)  # cast to int
            # Render the selected page to base64 PNG
            image_base64 = render_pdf_to_base64png(
                filename, page_num, target_longest_image_dim=target_image_dim
            )
            # Extract anchor text from the selected PDF page (not needed for v1.5)
            if task_type != "v1.5":
                anchor_text = get_anchor_text(
                    filename,
                    page_num,
                    pdf_engine="pdfreport",
                    target_length=target_text_length,
                )
        
        
        # Get the prompt template function for the specified task type
        prompt_fn = get_prompt(task_type)
        
        # Apply the prompt template to the extracted anchor text
        # For v1.5, no anchor text is needed but figure_language is passed
        if task_type == "v1.5":
            assert figure_language in ["Thai", "English"], "figure_language must be 'Thai' or 'English' for v1.5"
            prompt_text = prompt_fn(figure_language=figure_language)
        else:
            prompt_text = prompt_fn(anchor_text)
        
        # Create messages structure
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
        # Try to decode and re-encode to check validity
        return base64.b64encode(base64.b64decode(input_string))[:10] == input_string.encode()[:10]
    except Exception:
        return False

def ensure_image_in_path(input_string: str) -> str:
    """
    Detect whether the input is a base64-encoded image or a file path.

    - If it's base64, decode and save it as a temporary image file.
    - If it's a valid image format (e.g. JPEG, PNG), preserve the format.
    - If it's not base64, return the input as-is (assumed to be a path).

    Returns:
        str: A file path (either the original or a temp file path if base64).
    """
    if input_string.endswith(".png") or input_string.endswith(".jpg") or input_string.endswith(".jpeg") or input_string.endswith(".pdf"):
        return input_string
    elif is_base64_string(input_string):
        try:
            image_data = base64.b64decode(input_string)
            image = Image.open(io.BytesIO(image_data))
            image_format = image.format.lower()  # e.g. 'jpeg', 'png'
            # Save image to a temporary file with correct extension
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{image_format}")
            image.save(temp_file.name, format=image_format)
            return temp_file.name
        except Exception:
            return input_string
    return input_string

def ocr_document(pdf_or_image_path: str, task_type: str = "v1.5", target_image_dim: int = 1800, target_text_length: int = 8000, page_num: int = 1, base_url: str = os.getenv("TYPHOON_BASE_URL", 'https://api.opentyphoon.ai/v1'), api_key: str = None, model: str = "typhoon-ocr", figure_language: str = "Thai") -> str:
    """
    OCR a PDF or image file.
    
    This function provides an end-to-end workflow that combines multiple processing steps
    into a single call, creating messages ready for OCR processing with language models.
    It handles both image and PDF inputs, with appropriate page selection for PDFs.
    
    Args:
        pdf_or_image_path (str): Path to a PDF or image file to process
        task_type (str): Type of OCR task - "default" for standard markdown extraction,
                         "structure" for enhanced layout analysis with HTML tables,
                         "v1.5" for OCR v1.5 with clean Markdown and Thai figure descriptions
        target_image_dim (int): Target longest dimension for the rendered image in pixels
        target_text_length (int): Maximum length of extracted text to include (not used for v1.5)
        page_num (int): Page number to process (default=1, for images always 1)
        base_url (str): API base URL
        api_key (str): API key for authentication (will also check environment variables if None)
        model (str): Model identifier to use for OCR
        figure_language (str): Language instruction for figure descriptions in v1.5 (default: "Thai" | "English")
        
    Returns:
        str: Extracted text content in the specified format
        
    Raises:
        ValueError: If image conversion fails, page number is out of range, or other processing errors occur
    """
    if 'typhoon-ocr-preview' in model:
        assert task_type in ['default', 'structure'], "task_type must be 'default' or 'structure' for typhoon-ocr-preview models"
    pdf_or_image_path = ensure_image_in_path(pdf_or_image_path)
    
    openai = OpenAI(base_url=base_url, api_key=api_key or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv('TYPHOON_API_KEY') or os.getenv("OPENAI_API_KEY"))
    messages = prepare_ocr_messages(
        pdf_or_image_path=pdf_or_image_path,
        task_type=task_type,
        target_image_dim=target_image_dim,
        target_text_length=target_text_length,
        page_num=page_num if page_num else 1,
        figure_language=figure_language
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
    # For v1.5, text is returned directly without JSON wrapping
    if task_type == "v1.5":
        return text_output
    else:
        text = json.loads(text_output)['natural_text']
        return text