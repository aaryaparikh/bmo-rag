from __future__ import annotations

from typing import Any


def build_docling_converter(
    *,
    device: str = "auto",
    num_threads: int = 4,
    do_ocr: bool = False,
    force_backend_text: bool = True,
    do_table_structure: bool = True,
) -> Any:
    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling is not installed. Install project dependencies with "
            '`pip install -e ".[dev]"` before running ingestion.'
        ) from exc

    pdf_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device, num_threads=num_threads),
        do_table_structure=do_table_structure,
        do_ocr=do_ocr,
        force_backend_text=force_backend_text,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )


def build_hierarchical_chunker() -> Any:
    try:
        from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    except ImportError as exc:
        raise RuntimeError(
            "docling-core is not installed. Install project dependencies with "
            '`pip install -e ".[dev]"` before running ingestion.'
        ) from exc

    return HierarchicalChunker()


def export_docling_document(document: Any) -> dict[str, Any]:
    if hasattr(document, "export_to_dict"):
        try:
            return document.export_to_dict(confid_precision=4)
        except TypeError:
            return document.export_to_dict()
    if hasattr(document, "model_dump"):
        return document.model_dump(mode="json", by_alias=True, exclude_none=True)
    raise TypeError("Unsupported Docling document object")


def serialize_chunk(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(chunk, "dict"):
        return chunk.dict()
    if isinstance(chunk, dict):
        return chunk
    return {"text": str(chunk)}


def status_name(status: Any) -> str:
    name = getattr(status, "name", None) or getattr(status, "value", None) or str(status)
    return str(name).lower()
