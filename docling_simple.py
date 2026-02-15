import json
import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(r"./sample")
    output_dir = Path("./sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline options (done once)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )
    pipeline_options.ocr_options.lang = ["fr"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Process all PDF files in folder
    pdf_files = list(data_folder.glob("*.pdf"))

    if not pdf_files:
        _log.warning("No PDF files found in data folder.")
        return

    _log.info(f"Found {len(pdf_files)} PDF file(s) to process.")

    for input_doc_path in pdf_files:
        try:
            _log.info(f"Processing: {input_doc_path.name}")

            start_time = time.time()
            conv_result = doc_converter.convert(input_doc_path)
            duration = time.time() - start_time

            _log.info(
                f"{input_doc_path.name} converted in {duration:.2f} seconds."
            )

            doc_filename = conv_result.input.file.stem

            # Export DocTags
            with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
                fp.write(conv_result.document.export_to_doctags())

        except Exception as e:
            _log.error(f"Failed to process {input_doc_path.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
