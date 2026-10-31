"""
Regression & upgrade tests for scb-10x/typhoon-ocr fixes and hybrid enhancements.

Verifies:
1. Bug #1 Fix: resize_if_needed() no longer upscales images smaller than max_size (e.g. 600x400 with max_size=1800).
2. Bug #2 Fix: image_to_base64png() encodes lossless PNG bytes matching data:image/png;base64 header instead of lossy JPEG.
3. Bug #3 Fix: Thai PUA (U+F700..U+F71A) and decomposed toned Sara-Am normalization in _cleanup_element_text() and normalize_thai_text().
4. Bug #4 Fix: Poppler-free in-memory PyMuPDF (fitz) fallback for get_pdf_media_box_width_height(), render_pdf_to_base64png(), and prepare_ocr_messages() on real PDFs when Poppler binaries are absent.
5. Bug #5 Fix: Markdown code-fence JSON unwrapping in ocr_document() and automatic cleanup of base64 temporary files.
6. Feature #6: HTML <table> (rowspan/colspan) to 2D rows/Markdown extraction and hybrid multi-page PDF routing.
"""
import base64
import io
import os
import tempfile
from unittest.mock import MagicMock, patch
import fitz
from PIL import Image
import pytest

from typhoon_ocr.ocr_utils import (
    _cleanup_element_text,
    ensure_image_in_path,
    extract_html_tables,
    get_pdf_media_box_width_height,
    html_table_to_markdown,
    image_to_base64png,
    ocr_document,
    ocr_pdf_document,
    prepare_ocr_messages,
    render_pdf_to_base64png,
    resize_if_needed,
)


class TestBug1ResizeNoUpscaling:
    """Prove Bug #1: images between 301px and max_size must not be upscaled."""

    def test_medium_image_not_upscaled_to_max_size(self):
        img = Image.new("RGB", (600, 400), color="white")
        result = resize_if_needed(img, max_size=1800)
        assert result.size == (600, 400), (
            f"Expected (600, 400) without upscaling, got {result.size}"
        )
        assert result is img

    def test_image_at_exact_max_size_not_resized(self):
        img = Image.new("RGB", (1800, 1200), color="white")
        result = resize_if_needed(img, max_size=1800)
        assert result.size == (1800, 1200)
        assert result is img


class TestBug2TrueLosslessPng:
    """Prove Bug #2: image_to_base64png outputs real PNG bytes, not lossy JPEG."""

    def test_base64_payload_has_png_magic_bytes(self):
        img = Image.new("RGB", (120, 80), color=(10, 20, 30))
        b64_str = image_to_base64png(img)
        raw_bytes = base64.b64decode(b64_str)
        assert raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"), (
            f"Expected PNG magic header, got {raw_bytes[:8]!r}"
        )


class TestBug3ThaiPuaAndSaraAmNormalization:
    """Prove Bug #3: Thai PUA (U+F700..U+F71A) and decomposed Sara-Am are normalized."""

    def test_cleanup_element_text_normalizes_thai_pua_and_sara_am(self):
        corrupted = "สัญญาจ้าง น้ำ\uf70a\uf701 น\u0e4d\u0e49\u0e32ประปา ก\u0e4d\u0e32ลังไฟฟ้า\u200b\ufeff"
        cleaned = _cleanup_element_text(corrupted)
        expected = "สัญญาจ้าง น้ำ\u0e48\u0e34 น้ำประปา กำลังไฟฟ้า"
        assert cleaned == expected, f"Mismatch: {cleaned!r} != {expected!r}"


class TestBug4PopplerFreePyMuPdfFallback:
    """Prove Bug #4: Real PDFs work via PyMuPDF (fitz) even when Poppler is missing."""

    def test_real_pdf_mediabox_and_render_without_poppler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "sample.pdf")
            doc = fitz.open()
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 100), "SCBX Typhoon OCR PyMuPDF Fast-Path Test")
            doc.save(pdf_path)
            doc.close()

            with patch("typhoon_ocr.pdf_utils.pdf_utils_available", False):
                w, h = get_pdf_media_box_width_height(pdf_path, 1)
                assert abs(w - 595.0) < 1.0 and abs(h - 842.0) < 1.0

                b64_png = render_pdf_to_base64png(
                    pdf_path, 1, target_longest_image_dim=1800
                )
                raw = base64.b64decode(b64_png)
                assert raw.startswith(b"\x89PNG\r\n\x1a\n")
                rendered_img = Image.open(io.BytesIO(raw))
                assert max(rendered_img.size) == 1800

                msgs = prepare_ocr_messages(
                    pdf_path, task_type="v1.5", target_image_dim=1800, page_num=1
                )
                assert len(msgs) == 1
                assert msgs[0]["content"][1]["image_url"]["url"].startswith(
                    "data:image/png;base64,"
                )


class TestBug5JsonFenceAndTempFileCleanup:
    """Prove Bug #5: Markdown code fences in JSON responses and temp file cleanup."""

    @patch("typhoon_ocr.ocr_utils.OpenAI")
    @patch("typhoon_ocr.ocr_utils.prepare_ocr_messages")
    def test_markdown_fenced_json_response_parsed(self, mock_prepare, mock_openai):
        mock_prepare.return_value = [{"role": "user", "content": []}]
        fenced_json = '```json\n{"natural_text": "สวัสดี SCBX 2025"}\n```'
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=fenced_json))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai.return_value = mock_client

        result = ocr_document("dummy.png", task_type="default", api_key="test")
        assert result == "สวัสดี SCBX 2025"

    @patch("typhoon_ocr.ocr_utils.OpenAI")
    def test_base64_temp_file_is_cleaned_up_after_ocr_document(self, mock_openai):
        img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_input = base64.b64encode(buf.getvalue()).decode("utf-8")

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Clean output"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai.return_value = mock_client

        created_temps = []
        orig_ensure = ensure_image_in_path

        def spy_ensure(inp):
            res = orig_ensure(inp)
            if res != inp:
                created_temps.append(res)
            return res

        with patch(
            "typhoon_ocr.ocr_utils.ensure_image_in_path", side_effect=spy_ensure
        ):
            out = ocr_document(b64_input, task_type="v1.5", api_key="test")
            assert out == "Clean output"

        assert len(created_temps) == 1
        assert not os.path.exists(created_temps[0]), (
            f"Temp file leaked on disk: {created_temps[0]}"
        )


class TestFeature6HtmlTableAndHybridPdfRouter:
    """Prove Feature #6: HTML <table> rowspan/colspan expansion and hybrid multi-page PDF routing."""

    def test_html_table_rowspan_colspan_expansion_and_markdown(self):
        html_text = """
        # รายงานงบประมาณ
        <table>
          <tr><th rowspan="2">หมวด</th><th colspan="2">งบประมาณ (ล้านบาท)</th></tr>
          <tr><th>ปี 2567</th><th>ปี 2568</th></tr>
          <tr><td>ก\u0e4d\u0e32ลังไฟฟ้า</td><td>120.5</td><td>145.0</td></tr>
        </table>
        """
        tables = extract_html_tables(html_text)
        assert len(tables) == 1
        grid = tables[0]
        assert grid[0] == ["หมวด", "งบประมาณ (ล้านบาท)", ""]
        assert grid[1] == ["", "ปี 2567", "ปี 2568"]
        assert grid[2] == ["กำลังไฟฟ้า", "120.5", "145.0"]
        md = html_table_to_markdown(grid)
        assert "| กำลังไฟฟ้า | 120.5 | 145.0 |" in md

    def test_html_table_unclosed_th_td_auto_flush(self):
        """Prove unclosed <th>/<td> tags emitted by VLM are auto-flushed without losing columns."""
        html_text = (
            "<table><tr><th rowspan=\"2\">ปีงบประมาณ<th colspan=\"2\">เงินงบประมาณ"
            "<th colspan=\"2\">เงินนอกงบประมาณ<th rowspan=\"2\">รวมทั้งสิ้น</th></th></th></th></tr>"
            "<tr><th>จำนวนเงิน</th><th>ร้อยละ</th><th>จำนวนเงิน</th><th>ร้อยละ</th></tr>"
            "<tr><td>ปี 2566<td>169.3<td>7.91<td>0.0<td>0.0<td>169.3</td></tr></table>"
        )
        tables = extract_html_tables(html_text)
        assert len(tables) == 1
        grid = tables[0]
        assert grid[0] == ["ปีงบประมาณ", "เงินงบประมาณ", "", "เงินนอกงบประมาณ", "", "รวมทั้งสิ้น"]
        assert grid[1] == ["", "จำนวนเงิน", "ร้อยละ", "จำนวนเงิน", "ร้อยละ", ""]
        assert grid[2] == ["ปี 2566", "169.3", "7.91", "0.0", "0.0", "169.3"]

    @patch("typhoon_ocr.ocr_utils.ocr_document")
    def test_hybrid_pdf_router_vector_vs_vlm(self, mock_ocr_doc):
        mock_ocr_doc.return_value = "Scanned Page 2 VLM Result <table><tr><td>A</td><td>B</td></tr></table>"
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "hybrid.pdf")
            doc = fitz.open()
            # Page 1: Born-digital page with > 80 vector chars
            p1 = doc.new_page(width=595, height=842)
            p1.insert_text(
                (72, 100),
                "Digital Vector Page 1: This page contains sufficient born-digital vector text "
                "to trigger the PyMuPDF Tier-1 vector fast-path without invoking VLM GPU inference.",
            )
            # Page 2: Scanned/empty vector page (< 80 chars)
            p2 = doc.new_page(width=595, height=842)
            p2.insert_text((72, 100), "Scan")
            doc.save(pdf_path)
            doc.close()

            res = ocr_pdf_document(pdf_path, hybrid_vector_fastpath=True, min_vector_chars=80)
            assert res["total_pages"] == 2
            assert res["pages"][0]["tier"] == "pymupdf_vector"
            assert "Digital Vector Page 1" in res["pages"][0]["markdown"]
            assert res["pages"][1]["tier"] == "typhoon_vlm_openai"
            assert res["pages"][1]["tables_count"] == 1
            mock_ocr_doc.assert_called_once()
