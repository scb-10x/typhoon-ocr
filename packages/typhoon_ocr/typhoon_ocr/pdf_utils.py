import importlib.util
import shutil
import warnings


def check_pymupdf_available() -> bool:
    """Check if PyMuPDF (fitz) is installed for Poppler-free in-memory PDF processing."""
    return importlib.util.find_spec("fitz") is not None


pymupdf_available = check_pymupdf_available()


def check_pdf_utilities():
    """
    Check if the required Poppler utilities (pdfinfo and pdftoppm) are installed.
    
    This function verifies if the necessary PDF utilities are available on the system
    and provides helpful instructions if they are missing.
    
    Returns:
        bool: True if all required utilities are available, False otherwise.
    """
    missing_utils = []
    
    # Check for pdfinfo
    if shutil.which("pdfinfo") is None:
        missing_utils.append("pdfinfo")
    
    # Check for pdftoppm
    if shutil.which("pdftoppm") is None:
        missing_utils.append("pdftoppm")
    
    if missing_utils:
        warning_message = (
            f"WARNING: The following required Poppler utilities are missing: {', '.join(missing_utils)}.\n"
            "These utilities are required for PDF processing in Typhoon OCR.\n\n"
            "Installation instructions:\n"
            "- macOS: Run 'brew install poppler'\n"
            "- Ubuntu/Debian: Run 'apt-get install poppler-utils'\n"
            "- Windows: Install from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH\n"
        )
        warnings.warn(warning_message, ImportWarning)
        return False
    
    return True


# Check PDF utilities availability (will be checked at runtime when needed)
pdf_utils_available = check_pdf_utilities()
