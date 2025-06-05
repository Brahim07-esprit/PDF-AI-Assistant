import json
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List

import pypdf
import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .llm_context_generation import LLMContextGenerator


class ImageContextInfo:
    """Helper class to store context and path together."""

    def __init__(self, context: str, path: str):
        self.context = context
        self.path = path


LoadedContextsType = Dict[str, Dict[int, List[ImageContextInfo]]]
DATA_DIR = Path("processed_data")


class PDFProcessor:
    """Handles PDF parsing, image extraction, context generation, and vector store creation."""

    def __init__(
            self,
            llm_context_generator: LLMContextGenerator,
            embedding_function: Embeddings):
        self.llm_context_generator = llm_context_generator
        self.embedding_function = embedding_function

    @staticmethod
    def get_pdf_id(pdf_file_path: Path) -> str:
        hasher = hashlib.sha256()
        try:
            with open(pdf_file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]
        except OSError as e:
            print(
                f"ERROR: Could not read PDF file {pdf_file_path} to generate ID: {e}")
            raise ValueError(
                f"Could not read PDF file {pdf_file_path} to generate ID.") from e

    def _extract_and_contextualize_images(
            self,
            pdf_path: Path,
            img_output_base_dir: Path,
            context_json_path: Path) -> LoadedContextsType:
        doc = None
        page_image_contexts: Dict[int, List[ImageContextInfo]] = {}
        try:
            doc = fitz.open(pdf_path)
            for page_num_idx, page in enumerate(doc):
                page_num_one_based = page_num_idx + 1
                page_image_contexts[page_num_one_based] = []
                img_idx = 0
                try:
                    image_info_list = page.get_image_info(hashes=True)
                    for info in image_info_list:
                        img_idx += 1
                        rect = fitz.Rect(info["bbox"])
                        if rect.is_empty or rect.width <= 1 or rect.height <= 1:
                            continue

                        pix = page.get_pixmap(clip=rect, dpi=150)
                        img_filename_only = f"p{page_num_one_based:03d}_img{img_idx}.png"
                        img_abs_path = img_output_base_dir / img_filename_only

                        img_abs_path.parent.mkdir(parents=True, exist_ok=True)
                        pix.save(str(img_abs_path))
                        pix = None

                        context_margin = 30
                        page_rect = page.rect
                        context_rect = fitz.Rect(max(0,
                                                     rect.x0 - context_margin),
                                                 max(0,
                                                     rect.y0 - context_margin),
                                                 min(page_rect.width,
                                                     rect.x1 + context_margin),
                                                 min(page_rect.height,
                                                     rect.y1 + context_margin))
                        context_rect.normalize()
                        context_text = ""
                        if not context_rect.is_empty and context_rect.width > 0 and context_rect.height > 0:
                            context_text = page.get_text(
                                "text", clip=context_rect).strip()

                        llm_img_context = self.llm_context_generator.get_image_context(
                            context_text, page_num_one_based, img_idx)
                        if llm_img_context != "Context unclear from surrounding text.":
                            page_image_contexts[page_num_one_based].append(
                                ImageContextInfo(context=llm_img_context, path=img_filename_only)
                            )
                except Exception as e:
                    print(
                        f"WARNING: Error processing images on page {page_num_one_based} of {pdf_path.name}: {e}")
        except Exception as e:
            print(
                f"ERROR: Failed to open or process PDF for image extraction: {pdf_path.name} - {e}")
            raise
        finally:
            if doc:
                doc.close()

        try:
            with open(context_json_path, 'w', encoding='utf-8') as f:
                json.dump(page_image_contexts, f, indent=4)
        except IOError as e:
            print(
                f"ERROR: Could not write image contexts to {context_json_path}: {e}")

        return {pdf_path.name: page_image_contexts}

    def _load_image_contexts_from_file(
            self, context_json_path: Path) -> Dict[int, List[ImageContextInfo]]:
        if not context_json_path.exists():
            print(
                f"WARNING: Image context file not found: {context_json_path}")
            return {}
        try:
            with open(context_json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            page_contexts: Dict[int, List[ImageContextInfo]] = {}
            for page_num_str, infos_raw in loaded_data.items():
                try:
                    page_num_int = int(page_num_str)
                    page_contexts[page_num_int] = [
                        ImageContextInfo(
                            context=info['context'],
                            path=info['path']) for info in infos_raw]
                except (ValueError, KeyError, TypeError) as conversion_err:
                    print(
                        f"WARNING: Skipping malformed context entry for page '{page_num_str}' in {context_json_path}: {conversion_err}")
            return page_contexts
        except Exception as e:
            print(
                f"ERROR: Failed to load or parse image contexts from {context_json_path}: {e}")
            return {}

    def _process_pdf_text(self,
                          pdf_path: Path,
                          loaded_page_contexts: Dict[int,
                                                     List[ImageContextInfo]]) -> List[Document]:
        docs = []
        pdf_filename = pdf_path.name

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    page_num = i + 1
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        metadata = {"source": pdf_filename, "page": page_num}
                        docs.append(
                            Document(
                                page_content=page_text,
                                metadata=metadata))
        except Exception as e:
            print(f"ERROR: Error processing text from {pdf_filename}: {e}")
        return docs

    def _create_vector_store(
            self,
            pdf_path: Path,
            vs_persist_path: Path,
            loaded_full_context_data: LoadedContextsType) -> Chroma:
        pdf_filename = pdf_path.name
        page_specific_contexts = loaded_full_context_data.get(pdf_filename)
        if page_specific_contexts is None:
            print(
                f"WARNING: No context data found for {pdf_filename} in _create_vector_store. Proceeding with empty contexts.")
            page_specific_contexts = {}

        docs = self._process_pdf_text(pdf_path, page_specific_contexts)
        if not docs:
            raise RuntimeError(
                f"Failed to process PDF text for vector store creation: {pdf_filename}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs_splits = text_splitter.split_documents(docs)

        if vs_persist_path.exists():
            shutil.rmtree(vs_persist_path)
        vs_persist_path.parent.mkdir(parents=True, exist_ok=True)

        vector_store = Chroma.from_documents(
            documents=docs_splits,
            embedding=self.embedding_function,
            persist_directory=str(vs_persist_path)
        )
        return vector_store
