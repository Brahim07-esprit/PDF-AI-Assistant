from pathlib import Path
import fitz
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv


def extract_images(
    pdf_file: str | Path,
    out_dir: str | Path,
    llm_client: 'LLMContextGenerator',
    dpi: int = 300,
    deduplicate: bool = True,
    context_margin: int = 50
) -> list[dict]:
    pdf_file, out_dir = Path(pdf_file), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_out_dir = out_dir.resolve()
    results = []
    doc = None

    try:
        doc = fitz.open(pdf_file)
    except Exception as e:
        print(f"Error opening PDF {pdf_file}: {e}")
        return results

    seen_digests = set()
    print(f"Processing {len(doc)} pages...")

    for page in tqdm(doc, desc="Pages"):
        page_num = page.number + 1
        img_index = 0
        try:
            image_info_list = page.get_image_info(hashes=True)
            if not image_info_list:
                continue

            for info in image_info_list:
                digest = info.get("digest")
                if deduplicate and digest and digest in seen_digests:
                    continue
                if digest:
                    seen_digests.add(digest)

                try:
                    rect = fitz.Rect(info["bbox"])

                    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                        print(
                            f"WARN: Skipping invalid/empty image bbox on page {page_num}, index {img_index}: {rect}")
                        continue

                    pix = page.get_pixmap(clip=rect, dpi=dpi)
                    ref_id = info.get(
                        'xref', f'idx{img_index}')
                    img_filename = f"p{page_num:03d}_ref{ref_id}.png"
                    img_filepath = out_dir / img_filename
                    try:
                        pix.save(img_filepath)
                    except Exception as save_err:
                        print(
                            f"Error saving image {img_filename} on page {page_num}: {save_err}")
                        continue
                    finally:
                        pix = None

                    page_rect = page.rect
                    context_rect = fitz.Rect(
                        max(0, rect.x0 - context_margin),
                        max(0, rect.y0 - context_margin),
                        min(page_rect.width, rect.x1 + context_margin),
                        min(page_rect.height, rect.y1 + context_margin)
                    )
                    context_rect.normalize()
                    if context_rect.is_empty or context_rect.width <= 0 or context_rect.height <= 0:
                        context_text = ""
                    else:
                        context_text = page.get_text(
                            "text", clip=context_rect).strip()

                    image_context = llm_client.get_image_context(
                        context_text, page_num, img_index)

                    if image_context == "Context unclear from surrounding text.":
                        print(
                            f"INFO: Skipping image {img_index} on page {page_num} (xref: {info.get('xref', 'N/A')}) due to unclear context.")
                        continue

                    results.append({
                        "image_path": str(img_filepath.relative_to(abs_out_dir)),
                        "page": page_num,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "surrounding_text_snippet": context_text[:200] + ("..." if len(context_text) > 200 else ""),
                        "llm_context": image_context
                    })

                except Exception as img_err:
                    print(
                        f"Error processing image {img_index} on page {page_num} (xref: {info.get('xref', 'N/A')}): {img_err}")
                finally:
                    img_index += 1

        except Exception as page_err:
            print(f"Error processing page {page_num}: {page_err}")

    if doc:
        doc.close()
    return results


if __name__ == "__main__":
    load_dotenv()

    try:
        from llm_context_generation import LLMContextGenerator, MockLLMContextGenerator, GeminiContextGenerator
    except ImportError:
        print("ERROR: Could not import LLM context generator classes from llm_context_generation.py. ")
        print("       Make sure llm_context_generation.py is in the same directory or accessible in PYTHONPATH.")
        print("       This script cannot function without these classes.")
        exit(1)

    pdf_input_path = "books/The-Dark-Zebra-user-guide.pdf"
    output_directory = Path("books/images_with_context")
    results_file = Path("image_contexts.json")

    if not Path(pdf_input_path).is_absolute():
        pdf_input_path = Path.cwd() / pdf_input_path
    if not output_directory.is_absolute():
        output_directory = Path.cwd() / output_directory

    print("Initializing LLM context generator...")
    llm_client = None
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("GOOGLE_API_KEY not found in environment.")
            llm_client = MockLLMContextGenerator()
        else:
            llm_client = GeminiContextGenerator(api_key=api_key)

    except ImportError:
        print("Required libraries for Gemini not found (google-generativeai or python-dotenv).")
        llm_client = MockLLMContextGenerator()
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        llm_client = MockLLMContextGenerator()

    if llm_client is None:
        print("LLM client initialization failed unexpectedly. Using Mock client.")
        llm_client = MockLLMContextGenerator()

    print(
        f"Extracting images and context from '{pdf_input_path}' to '{output_directory}'")
    try:
        image_data = extract_images(
            pdf_file=pdf_input_path,
            out_dir=output_directory,
            llm_client=llm_client,
        )

        print(
            f"Extraction complete. Found {len(image_data)} images with context.")
        results_filepath = Path.cwd() / results_file
        print(f"Saving results to '{results_filepath}'")
        try:
            with open(results_filepath, 'w', encoding='utf-8') as f:
                json.dump(image_data, f, indent=4, ensure_ascii=False)
            print("Results saved successfully.")
        except IOError as e:
            print(f"Error saving results to {results_filepath}: {e}")

        print("Done.")

    except Exception as e:
        print(f"\n--- An error occurred during the main process ---: {e}")
        import traceback
        traceback.print_exc()
