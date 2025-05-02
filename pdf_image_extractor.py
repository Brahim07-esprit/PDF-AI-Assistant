# tested with Python 3.11.7 - Windows 11 & Ubuntu 22.04
from pathlib import Path
import fitz  # PyMuPDF >= 1.25
from tqdm import tqdm
import json
import os
import abc
from dotenv import load_dotenv
# Langchain imports for Gemini
from langchain_core.messages import HumanMessage

# --- LLM Context Generation ---

class LLMContextGenerator(abc.ABC):
    """Abstract base class for LLM context generators."""
    @abc.abstractmethod
    def get_image_context(self, text_around_image: str, page_num: int, img_idx: int) -> str:
        """Generates context description for an image based on surrounding text."""
        pass

class MockLLMContextGenerator(LLMContextGenerator):
    """Mock generator for testing without API calls."""
    def __init__(self):
        print("WARNING: Using Mock LLM Context Generator. No real context will be generated.")
        print("         Install 'google-generativeai', 'python-dotenv' and set your 'GOOGLE_API_KEY' in .env to use Gemini.")

    def get_image_context(self, text_around_image: str, page_num: int, img_idx: int) -> str:
        if text_around_image.strip():
            return f"[Mock Context] Based on surrounding text on page {page_num}, this image likely relates to: '{text_around_image[:100].strip()}...'"
        else:
            return f"[Mock Context] No text found near the image on page {page_num} to determine context."

class GeminiContextGenerator(LLMContextGenerator):
    """Generates context using Google Gemini via Langchain."""
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest"):
        try:
            # Import within the method to ensure library exists if called
            from langchain_google_genai import ChatGoogleGenerativeAI

            print(f"Initializing Langchain Gemini client with model: {model_name}")
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.5, # Adjust temperature as needed
                # convert_system_message_to_human=True # May be needed depending on model/prompt
            )
            print("Langchain Gemini client initialized successfully.")
        except ImportError:
            raise ImportError("langchain-google-genai or langchain-core not found. Please install them: pip install langchain-google-genai langchain-core")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Langchain Gemini client: {e}")

    def get_image_context(self, text_around_image: str, page_num: int, img_idx: int) -> str:
        prompt_text = f"""\
        Analyze the following text extracted from page {page_num} of a PDF, near an image (image index {img_idx}).
        Based *only* on this text, describe the likely purpose or context of the image in a concise sentence.
        If the text is insufficient to determine the context, state 'Context unclear from surrounding text.'.

        Extracted Text:
        ---
        {text_around_image}
        ---

        Context Description:"""
        try:
            message = HumanMessage(content=prompt_text)
            response = self.model.invoke([message])
            # Accessing the content from the AIMessage response object
            if isinstance(response.content, str):
                return response.content.strip()
            else:
                 # Handle cases where content might not be a simple string (e.g., streaming, errors)
                 print(f"WARN: Unexpected response type from Langchain Gemini for image {img_idx} on page {page_num}: {type(response.content)}")
                 return "Context generation failed (unexpected response format)."
        except Exception as e:
            print(f"Error calling Langchain Gemini API for image {img_idx} on page {page_num}: {e}")
            return "Error generating context via Langchain Gemini API."

# --- PDF Processing ---

def extract_images(
    pdf_file: str | Path,
    out_dir: str | Path,
    llm_client: LLMContextGenerator, # Use the base class type hint
    dpi: int = 300,
    deduplicate: bool = True,
    context_margin: int = 50 # Pixels around image bbox to check for text
) -> list[dict]:
    """\
    Extracts displayed images, saves them, extracts surrounding text,
    uses an LLM to determine context, and returns a list of image paths and contexts.
    """
    pdf_file, out_dir = Path(pdf_file), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ensure out_dir is absolute for reliable relative path calculation later
    abs_out_dir = out_dir.resolve()
    results = []
    doc = None # Initialize doc to None

    try:
        doc = fitz.open(pdf_file)
    except Exception as e:
        print(f"Error opening PDF {pdf_file}: {e}")
        return results # Return empty list if PDF can't be opened

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
                # Deduplication logic
                if deduplicate and digest and digest in seen_digests:
                    continue
                if digest: # Add only if deduplicating and digest exists
                    seen_digests.add(digest)

                try:
                    rect = fitz.Rect(info["bbox"]) # Image BBox

                    # --- Image Saving ---
                    # Check for empty or invalid rect
                    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                         print(f"WARN: Skipping invalid/empty image bbox on page {page_num}, index {img_index}: {rect}")
                         continue

                    pix = page.get_pixmap(clip=rect, dpi=dpi)
                    ref_id = info.get('xref', f'idx{img_index}') # Use xref or index
                    img_filename = f"p{page_num:03d}_ref{ref_id}.png"
                    img_filepath = out_dir / img_filename
                    try:
                        pix.save(img_filepath)
                    except Exception as save_err:
                        print(f"Error saving image {img_filename} on page {page_num}: {save_err}")
                        continue # Skip this image if saving fails
                    finally:
                        pix = None # Release pixmap memory

                    # --- Context Extraction ---
                    page_rect = page.rect
                    # Define context box slightly larger than image bbox
                    context_rect = fitz.Rect(
                        max(0, rect.x0 - context_margin),
                        max(0, rect.y0 - context_margin),
                        min(page_rect.width, rect.x1 + context_margin),
                        min(page_rect.height, rect.y1 + context_margin)
                    )
                    # Ensure context_rect is valid
                    context_rect.normalize()
                    if context_rect.is_empty or context_rect.width <= 0 or context_rect.height <= 0:
                         context_text = "" # No valid area to extract text from
                    else:
                         context_text = page.get_text("text", clip=context_rect).strip()

                    # --- LLM Context Generation ---
                    image_context = llm_client.get_image_context(context_text, page_num, img_index)

                    # --- Filtering --- 
                    if image_context == "Context unclear from surrounding text.":
                        print(f"INFO: Skipping image {img_index} on page {page_num} (xref: {info.get('xref', 'N/A')}) due to unclear context.")
                        continue # Skip appending this result

                    results.append({
                        # Use absolute output dir for relative path calculation
                        "image_path": str(img_filepath.relative_to(abs_out_dir)),
                        "page": page_num,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "surrounding_text_snippet": context_text[:200] + ("..." if len(context_text) > 200 else ""), # Store snippet
                        "llm_context": image_context
                    })

                except Exception as img_err:
                    print(f"Error processing image {img_index} on page {page_num} (xref: {info.get('xref', 'N/A')}): {img_err}")
                finally:
                     img_index += 1 # Increment even if an error occurred for this specific image

        except Exception as page_err:
             print(f"Error processing page {page_num}: {page_err}")
             # Continue to the next page

    if doc:
        doc.close()
    return results

# --- Main Execution ---

if __name__ == "__main__":
    load_dotenv() # Load variables from .env file

    pdf_input_path = "books/The-Dark-Zebra-user-guide.pdf"
    # Define output relative to script location or use absolute path
    output_directory = Path("books/images_with_context") 
    results_file = Path("image_contexts.json") # Store results in CWD

    # Ensure paths are handled correctly (e.g., make them absolute if needed)
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
            # Attempt to use the Gemini client
            llm_client = GeminiContextGenerator(api_key=api_key) # Uses gemini-1.5-flash-latest by default

    except ImportError:
         print("Required libraries for Gemini not found (google-generativeai or python-dotenv).")
         llm_client = MockLLMContextGenerator()
    except Exception as e:
         print(f"Error initializing Gemini client: {e}")
         llm_client = MockLLMContextGenerator()

    # Ensure we always have a client (Mock if real one failed)
    if llm_client is None:
         print("LLM client initialization failed unexpectedly. Using Mock client.")
         llm_client = MockLLMContextGenerator()


    print(f"Extracting images and context from '{pdf_input_path}' to '{output_directory}'")
    try:
        image_data = extract_images(
            pdf_file=pdf_input_path,
            out_dir=output_directory,
            llm_client=llm_client,
        )

        print(f"Extraction complete. Found {len(image_data)} images with context.")
        # Ensure results file path is correct (e.g., save in CWD)
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