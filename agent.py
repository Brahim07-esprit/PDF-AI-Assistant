# Standard Library Imports
import os
import glob
import json # Added for loading image contexts
from typing import List, Optional, Dict, Tuple # Added Dict and Tuple
from pathlib import Path
import abc # Added for Generator base class
import shutil # Added for removing directory
import hashlib

# Third-Party Imports
import pypdf
from dotenv import load_dotenv
import fitz # Added for image extraction logic (if moved here later, but needed for context gen now)
from tqdm import tqdm # Added for progress bars (if context gen moved here later)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings # Keep for embeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END

# Load environment variables
load_dotenv()

# --- LLM Context Generation (Moved from pdf_image_extractor) ---
class LLMContextGenerator(abc.ABC):
    """Abstract base class for LLM context generators."""
    @abc.abstractmethod
    def get_image_context(self, text_around_image: str, page_num: int, img_idx: int) -> str:
        pass

class MockLLMContextGenerator(LLMContextGenerator):
    """Mock generator for testing without API calls."""
    def __init__(self):
        print("INFO: Using Mock LLM Context Generator.")
        # --- PDF Processing Pipeline --- #

    def get_pdf_id(self, pdf_file_path: Path) -> str:
         """Generates a unique ID based on PDF content hash."""
         hasher = hashlib.sha256()
         try:
             with open(pdf_file_path, 'rb') as f:
                 while chunk := f.read(8192):
                     hasher.update(chunk)
             # Use first 16 chars of hash for directory names
             return hasher.hexdigest()[:16]
         except OSError as e:
             print(f"Error reading PDF file for hashing: {e}")
             raise ValueError(f"Could not read PDF file {pdf_file_path} to generate ID.") from e

    def setup_pdf_environment(self, pdf_file_path: str) -> str:
        """Processes an uploaded PDF: extracts images/context, builds vector store. Returns pdf_id."""
        pdf_path = Path(pdf_file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        pdf_id = self.get_pdf_id(pdf_path)
        pdf_filename = pdf_path.name
        print(f"Processing PDF: {pdf_filename} (ID: {pdf_id})")

        # Define dynamic paths relative to the agent's execution directory
        script_dir = Path(__file__).parent
        pdf_data_dir = script_dir / DATA_DIR / pdf_id
        vs_path = pdf_data_dir / "vectorstore"
        img_dir = pdf_data_dir / "images"
        context_file_path = pdf_data_dir / "image_contexts.json"

        # Check if already processed and loaded in memory
        if pdf_id in self.vector_stores:
            print(f"PDF {pdf_id} already processed and loaded in memory.")
            # Quick check if context data is also loaded
            if pdf_id not in self.context_data:
                 print(f"Warning: VS for {pdf_id} loaded, but context data missing. Attempting reload.")
                 try:
                     # We need pdf_filename for the structure {pdf_id:{pdf_filename:{page:[Info]}}}
                     # But _load_image_contexts_from_file only returns the page dict {page:[Info]}
                     loaded_page_contexts = self._load_image_contexts_from_file(context_file_path)
                     self.context_data[pdf_id] = {pdf_filename: loaded_page_contexts} # Store correctly
                 except Exception as load_err:
                      print(f"Error reloading context data for {pdf_id}: {load_err}. Proceeding without image context.")
                      self.context_data[pdf_id] = {pdf_filename: {}} # Ensure key exists even if empty
            return pdf_id

        # Check if processed on disk but not loaded in memory
        if vs_path.exists() and context_file_path.exists() and any(vs_path.iterdir()):
             print(f"Found existing processed data for PDF {pdf_id} on disk. Loading...")
             try:
                 # Load vector store
                 self.vector_stores[pdf_id] = Chroma(persist_directory=str(vs_path), embedding_function=self.embedding_function)
                 # Load contexts directly into the main structure
                 loaded_page_contexts = self._load_image_contexts_from_file(context_file_path)
                 # Store under pdf_id -> pdf_filename -> page_num
                 self.context_data[pdf_id] = {pdf_filename: loaded_page_contexts}
                 print(f"Successfully loaded environment for PDF {pdf_id} from disk.")
                 return pdf_id
             except Exception as e:
                 print(f"Error loading existing data for {pdf_id} from disk: {e}. Reprocessing...")
                 if pdf_data_dir.exists(): shutil.rmtree(pdf_data_dir) # Clean up before reprocessing
        else:
             print(f"No complete pre-processed data found for PDF {pdf_id}. Starting full pipeline...")


        # --- Perform Full Processing Pipeline --- #
        # Ensure directories exist
        pdf_data_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        print(f"Starting new processing pipeline for PDF {pdf_id}...")
        try:
            # 1. Extract Images and Generate Context
            # This saves the JSON and returns the structured data {pdf_filename: {page: [Info]}}
            generated_context_data = self._extract_and_contextualize_images(pdf_path, img_dir, context_file_path)
            # Store structured data under pdf_id
            self.context_data[pdf_id] = generated_context_data

            # 2. Create Vector Store with Text and Context Metadata
            # Pass the context data we just generated/loaded
            vs = self._create_vector_store(pdf_path, vs_path, generated_context_data)
            self.vector_stores[pdf_id] = vs

            print(f"Successfully processed and loaded environment for PDF {pdf_id}.")
            return pdf_id

        except Exception as e:
            print(f"FATAL ERROR during PDF processing pipeline for {pdf_id}: {e}")
            # Clean up potentially corrupted directories if processing failed
            if pdf_data_dir.exists():
                print(f"Cleaning up failed processing directory: {pdf_data_dir}")
                shutil.rmtree(pdf_data_dir)
            # Remove from memory if partially loaded
            self.vector_stores.pop(pdf_id, None)
            self.context_data.pop(pdf_id, None)
            import traceback
            traceback.print_exc() # Show full error
            raise # Re-raise the exception to signal failure to Streamlit

    # --- Existing Helper Methods (_extract_and_contextualize_images, etc.) --- #
    # ... Make sure these methods are correctly defined below ...
    def get_image_context(self, text_around_image: str, page_num: int, img_idx: int) -> str:
        if text_around_image.strip(): return f"[Mock Context for image {img_idx} on page {page_num}]"
        else: return "Context unclear from surrounding text."

class GeminiContextGenerator(LLMContextGenerator):
    """Generates context using Google Gemini via Langchain."""
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.0):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print(f"Initializing Image Context Gemini client with model: {model_name} (Temp: {temperature})")
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature
            )
            print("Image Context Gemini client initialized successfully.")
        except ImportError:
            raise ImportError("langchain-google-genai/core missing.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Image Context Gemini client: {e}")

    def get_image_context(self, text_around_image: str, page_num: int, img_idx: int) -> str:
        prompt_text = f"""
Analyze text near an image (index {img_idx}) on PDF page {page_num}. Describe the image's likely purpose/context concisely. If unclear, state 'Context unclear from surrounding text.'.

Text:
---
{text_around_image}
---

Context Description:"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt_text)])
            return response.content.strip() if isinstance(response.content, str) else "Context generation failed."
        except Exception as e:
            print(f"ERROR: Gemini API call failed for image {img_idx}, page {page_num}: {e}")
            return "Error generating context via API."

# --- Helper Data Structure --- #
class ImageContextInfo:
    """Helper class to store context and path together."""
    def __init__(self, context: str, path: str):
        self.context = context
        self.path = path

# Type hint for the loaded contexts dictionary
# Structure: { "filename.pdf": { page_num: [ImageContextInfo(...), ...], ... } }
LoadedContextsType = Dict[str, Dict[int, List[ImageContextInfo]]]

# --- Dynamic Directory Structure Base --- #
DATA_DIR = Path("processed_data") # Base directory for dynamic pdf data

# --- LangGraph State Definition --- #
class CustomState(MessagesState):
    """Custom state for the LangGraph graph, inheriting from MessagesState."""
    retrieved_chunks: Optional[List[Document]] = None # Store full docs to access metadata
    user_query: Optional[str] = None
    relevant_image_paths: Optional[List[str]] = None
    active_pdf_id: Optional[str] = None # To know which PDF environment to use

# --- Zebra Agent Class --- #
class ZebraAgent:
    """Agent specialized in Zebra U-HE using PDF text and pre-generated image contexts."""

    DEFAULT_PERSIST_DIR = "pdf_docs_vectorstore"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_LLM_MODEL = "gemini-1.5-flash-latest"
    DEFAULT_IMAGE_CONTEXT_FILE = "image_contexts.json"
    DEFAULT_IMG_CONTEXT_LLM_MODEL = "gemini-2.0-flash"

    def __init__(self):
        """Initializes the ZebraAgent, loading pre-generated image contexts and setting up components."""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY") # Needed for embeddings

        if not self.google_api_key or not self.openai_api_key:
            raise ValueError("Missing GOOGLE_API_KEY or OPENAI_API_KEY in environment.")

        print("Initializing core agent components...")
        self.embedding_function = OpenAIEmbeddings(model=self.DEFAULT_EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(
            model=self.DEFAULT_LLM_MODEL,
            temperature=0,
            google_api_key=self.google_api_key
        )

        # Separate LLM instance for image context generation
        try:
             self.image_context_llm: LLMContextGenerator = GeminiContextGenerator(
                 api_key=self.google_api_key,
                 model_name=self.DEFAULT_IMG_CONTEXT_LLM_MODEL,
                 temperature=0.0
             )
        except Exception as e:
             print(f"Warning: Failed to initialize Gemini for image context ({e}). Using Mock.")
             self.image_context_llm = MockLLMContextGenerator()

        # Storage for PDF-specific data (keyed by pdf_id)
        self.vector_stores: Dict[str, Chroma] = {}
        self.context_data: Dict[str, LoadedContextsType] = {}

        self._compile_graph()
        print("Agent core initialized.")

    def _load_image_contexts(self) -> LoadedContextsType:
        """Loads image context data AND paths from the pre-generated JSON file."""
        contexts_and_paths_by_page: LoadedContextsType = {}
        try:
            context_path = Path(self.DEFAULT_IMAGE_CONTEXT_FILE)
            if not context_path.is_absolute():
                # Assume JSON file is relative to the agent script's directory if not absolute
                script_dir = Path(__file__).parent
                context_path = script_dir / context_path

            if not context_path.exists():
                print(f"Error: Image context file not found at '{context_path}'. "
                      "Please run the pre-processing script first.")
                # Stop initialization if the essential context file is missing
                raise FileNotFoundError(f"Required image context file not found: {context_path}")

            print(f"Loading image contexts from '{context_path}'")
            with open(context_path, 'r', encoding='utf-8') as f:
                image_data = json.load(f)

            # Get the expected source PDF filename based on the pdf_folder_path
            # Assumes only one PDF is processed by this agent instance for simplicity
            pdf_files = glob.glob(os.path.join(self.pdf_folder_path, "*.pdf"))
            if not pdf_files:
                print("Error: No PDF files found in the specified folder.")
                raise FileNotFoundError(f"No PDF files found in {self.pdf_folder_path}")
            source_pdf_filename = os.path.basename(pdf_files[0])
            print(f"Associating loaded image contexts with source file: {source_pdf_filename}")
            contexts_and_paths_by_page[source_pdf_filename] = {}

            # Structure data: { "filename.pdf": { page_num: [ImageContextInfo(), ...], ... } }
            loaded_context_count = 0
            for item in image_data:
                try:
                    page_num = item['page']
                    context = item.get('llm_context', '')
                    img_path = item.get('image_path')

                    if img_path is None:
                        print(f"Warning: Skipping entry, missing 'image_path': {item}")
                        continue

                    if page_num not in contexts_and_paths_by_page[source_pdf_filename]:
                        contexts_and_paths_by_page[source_pdf_filename][page_num] = []

                    # Store if context is meaningful
                    if context and context != "Context unclear from surrounding text.":
                        contexts_and_paths_by_page[source_pdf_filename][page_num].append(
                            ImageContextInfo(context=context, path=img_path)
                        )
                        loaded_context_count += 1
                    # else: Optionally store entries with unclear context if needed later

                except (KeyError, TypeError) as e:
                    print(f"Warning: Skipping image context entry due to parsing error: {item}. Error: {e}")

            loaded_page_count = len(contexts_and_paths_by_page[source_pdf_filename])
            print(f"Successfully structured {loaded_context_count} contexts/paths across {loaded_page_count} pages for {source_pdf_filename}.")
            return contexts_and_paths_by_page

        except Exception as e:
            print(f"Fatal Error loading image context file '{self.DEFAULT_IMAGE_CONTEXT_FILE}': {e}")
            raise # Re-raise critical error

    def _process_pdf_files(self) -> List[Document]:
        """Reads text from PDF files page by page, adding pre-loaded image *context descriptions* to metadata."""
        docs = []
        pdf_pattern = os.path.join(self.pdf_folder_path, "*.pdf")
        print(f"Searching for PDFs in: {pdf_pattern}")
        pdf_files = glob.glob(pdf_pattern)
        print(f"Found {len(pdf_files)} PDF files.")

        for pdf_file_path in pdf_files:
            pdf_filename = os.path.basename(pdf_file_path)
            print(f"Processing PDF text for: {pdf_filename}")
            try:
                with open(pdf_file_path, "rb") as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for i, page in enumerate(pdf_reader.pages):
                        page_num = i + 1 # 1-based index
                        page_text = page.extract_text()

                        if page_text and page_text.strip():
                            # Find corresponding ImageContextInfo objects using the pre-loaded dictionary
                            page_context_infos = self.context_data.get(pdf_filename, {}).get(page_num, [])

                            # Combine only the context strings for metadata storage
                            combined_contexts = "\n".join(f"- {info.context}" for info in page_context_infos)

                            metadata = {
                                "source": pdf_filename,
                                "page": page_num,
                                # Store combined context descriptions for potential LLM use
                                "image_contexts_on_page": combined_contexts or "No relevant image contexts found on this page."
                            }
                            doc = Document(page_content=page_text, metadata=metadata)
                        docs.append(doc)
                        # else: Skip pages with no text

            except Exception as e:
                print(f"Error processing {pdf_filename}: {e}")
        print(f"Created {len(docs)} page-level documents with text and metadata.")
        return docs

    def _initialize_vectorstore(self) -> None:
        """Initializes or loads the Chroma vector store. Rebuilds if necessary based on context file presence/changes (simple check)."""
        # Basic check: if context file exists but store doesn't, we must build.
        # A more robust check might involve checksums or timestamps.
        needs_rebuild = False
        context_file_exists = Path(self.DEFAULT_IMAGE_CONTEXT_FILE).exists() # Check actual path used in _load

        if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
             if not context_file_exists:
                  print("Error: Vector store does not exist and image context file is also missing. Cannot proceed.")
                  raise FileNotFoundError("Vector store and image context file missing.")
             print(f"No existing vector store found at {self.persist_directory}. Creating new one...")
             needs_rebuild = True
        # Optional: Add logic here to force rebuild if context file is newer than store, etc.
        # else: Check if metadata is as expected? For now, load if exists.

        if needs_rebuild:
            self._create_new_vectorstore()
        else:
            print(f"Loading existing vector store from {self.persist_directory}")
            try:
                self.pdf_docs_vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function
                )
                print("Vector store loaded successfully.")
                # Verify essential component loaded
                if self.pdf_docs_vectorstore is None:
                     raise RuntimeError("Chroma failed to load the vector store object.")
            except Exception as e:
                 print(f"Error loading vector store from {self.persist_directory}: {e}")
                 print("Attempting to rebuild the vector store as fallback.")
                 self._create_new_vectorstore()

        # Final check after load/create attempt
        if self.pdf_docs_vectorstore is None:
             raise RuntimeError("Fatal Error: Vector store could not be initialized or loaded.")


    def _create_new_vectorstore(self) -> None:
        """Processes PDFs page-by-page and creates a new vector store with image context metadata."""
        # Ensure the directory is clean before creating
        if os.path.exists(self.persist_directory):
             print(f"Note: Removing existing directory {self.persist_directory} before creating new store.")
             shutil.rmtree(self.persist_directory)

        # Process PDF text and get page-level docs with pre-loaded image context metadata
        docs = self._process_pdf_files()
        if not docs:
            print("Error: No PDF documents processed. Cannot create vector store.")
            self.pdf_docs_vectorstore = None # Set to None before raising
            raise RuntimeError("Failed to process PDF documents for vector store creation.")

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_splits = text_splitter.split_documents(docs)
        print(f"Split {len(docs)} page documents into {len(docs_splits)} chunks.")

        # Create and persist vector store
        try:
            print("Creating Chroma vector store with page text and image context metadata...")
            self.pdf_docs_vectorstore = Chroma.from_documents(
                documents=docs_splits, # These splits inherit the page metadata
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            print(f"Vector store created and persisted at {self.persist_directory}")
            if self.pdf_docs_vectorstore is None:
                 raise RuntimeError("Chroma.from_documents returned None, vector store creation failed.")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.pdf_docs_vectorstore = None # Set to None before raising
            raise RuntimeError(f"Failed to create vector store: {e}")

    # --- Graph Nodes --- #
    def _router_node(self, state: CustomState) -> dict:
        """Determines whether the query is general or Zebra U-HE related."""
        print("--- Router Node ---")
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            print("Router Error: Last message is not a HumanMessage")
            return {"next": "general_response"}

        router_prompt = f"""
You are an intelligent assistant that determines how to route user queries about the Zebra U-HE synthesizer plugin.

        IMPORTANT ROUTING RULES:
- If the query is clearly about Zebra U-HE (features, sounds, patches, usage, errors, etc.) -> Respond with "ZEBRA_QUERY"
- For all other general questions (math, history, coding, other software, etc.) -> Respond with "GENERAL_QUERY"

        User message: "{last_message.content}"

Respond with only "ZEBRA_QUERY" or "GENERAL_QUERY".
        """

        try:
            classification = self.llm.invoke([HumanMessage(content=router_prompt)])
            classification_text = classification.content.strip()
            print(f"Router classification: {classification_text}")

            if "ZEBRA_QUERY" in classification_text:
                return {"next": "vector_store", "user_query": last_message.content}
            else:
                return {"next": "general_response"}
        except Exception as e:
            print(f"Router LLM call failed: {e}")
            return {"next": "general_response"}

    def _general_response_node(self, state: CustomState) -> dict:
        """Handles general, non-Zebra related queries."""
        print("--- General Response Node ---")
        messages = state["messages"]
        general_system_prompt = """
You are a helpful AI assistant. Answer the user's query concisely.
If the user asks specifically about the Zebra U-HE synthesizer here, politely state that another part of the system handles those questions.
        """
        llm_messages = [SystemMessage(content=general_system_prompt)]
        # Include only the last human message for context
        if messages and isinstance(messages[-1], HumanMessage):
            llm_messages.append(messages[-1])
        else:
             # Handle cases where there might not be a human message (shouldn't happen with entry point)
             return {"messages": [AIMessage(content="I seem to have lost the context. Could you please repeat your question?")], "next": END}

        try:
            response = self.llm.invoke(llm_messages)
            return {"messages": [AIMessage(content=response.content)], "next": END}
        except Exception as e:
            print(f"General Response LLM call failed: {e}")
            return {"messages": [AIMessage(content="Sorry, I encountered an error trying to respond.")], "next": END}

    def _vector_store_node(self, state: CustomState) -> dict:
        """Retrieves relevant chunks from the active PDF's vector store."""
        print("--- Vector Store Node ---")
        vector_store, _ = self._get_current_resources(state)
        user_query = state.get("user_query")

        # Check if required resources are available
        if not vector_store:
            error_msg = "Vector store not available for the active PDF."
            print(f"Error in vector store node: {error_msg}")
            return {"messages": [AIMessage(content=f"Error: {error_msg}")], "relevant_image_paths": [], "next": END}
        if not user_query:
            error_msg = "User query missing from state."
            print(f"Error in vector store node: {error_msg}")
            return {"messages": [AIMessage(content=f"Error: {error_msg}")], "relevant_image_paths": [], "next": END}

        print(f"Searching vector store for query: '{user_query[:50]}...'")
        retrieved_docs = []
        try:
            # Use the specific vector_store instance for the active PDF
            results_with_scores = vector_store.similarity_search_with_relevance_scores(query=user_query, k=5)
            retrieved_docs = [doc for doc, score in results_with_scores if score > 0.7]
            print(f"Retrieved {len(retrieved_docs)} relevant document chunks (score > 0.7).")
            # Fallback logic
            if not retrieved_docs and results_with_scores:
                 print("Using top result as fallback.")
                 retrieved_docs = [results_with_scores[0][0]]
            elif not retrieved_docs:
                 print("No relevant chunks found.")
        except Exception as e:
            print(f"Error during vector store search: {e}")
            return {"messages": [AIMessage(content="Error searching document database.")], "relevant_image_paths": [], "next": END}

        # Update state with retrieved documents (including metadata)
        return { "retrieved_chunks": retrieved_docs, "next": "pdf_questions" }

    def _pdf_questions_node(self, state: CustomState) -> dict:
        """Generates a response using query, text chunks, and looks up associated image paths from pre-loaded data."""
        print("--- PDF Questions Node ---")
        vector_store, full_context_data = self._get_current_resources(state)
        retrieved_docs: List[Document] = state.get("retrieved_chunks", [])
        user_query = state.get("user_query", "Unknown query")
        pdf_id = state.get("active_pdf_id")

        if not retrieved_docs:
            print("No retrieved documents to provide context.")
            no_context_response = "I couldn't find specific information related to your query in the Zebra UHE documentation." # Simpler message
            return {"messages": [AIMessage(content=no_context_response)], "relevant_image_paths": [], "next": END}

        if not full_context_data or not pdf_id:
             print(f"Error: Context data or pdf_id missing in PDF Questions Node.")
             # Fallback: Respond with text only?
             return {"messages": [AIMessage(content="Error: Image context data missing.")], "relevant_image_paths": [], "next": END}

        # Extract text chunks, identify relevant pages, and look up image paths/contexts
        text_chunks = []
        image_contexts_for_prompt = set()
        image_paths_for_state = set()
        relevant_pages = set() # Store tuples of (source_filename, page_number)

        for doc in retrieved_docs:
            text_chunks.append(doc.page_content)
            source_file = doc.metadata.get("source")
            page_num = doc.metadata.get("page")
            if source_file and page_num:
                relevant_pages.add((source_file, page_num))
            # Extract pre-combined context string directly from metadata for prompt
            ctx_string = doc.metadata.get("image_contexts_on_page", "")
            if ctx_string and ctx_string != "No relevant image contexts found on this page.":
                 # Split the combined string back into individual contexts for the set
                 for line in ctx_string.split('\n'):
                     if line.startswith("- "):
                          image_contexts_for_prompt.add(line[2:])

        # Look up the actual image paths for the identified relevant pages
        for source, page in relevant_pages:
             if source in full_context_data and page in full_context_data[source]:
                 page_infos = full_context_data[source][page]
                 for info in page_infos:
                     image_paths_for_state.add(info.path) # Add the path

        # --- Format prompt --- #
        chunks_str = "\n\n---\n\n".join(text_chunks)
        # Use the unique contexts extracted from metadata for the prompt string
        image_contexts_str = "\n".join(f"- {ctx}" for ctx in sorted(list(image_contexts_for_prompt))) \
                              if image_contexts_for_prompt else "No relevant image descriptions found for the retrieved text."

        zebra_system_prompt = """ # Reverted to a simpler, robust prompt
You are ZebraSynthExpert, an AI assistant specialized in the Zebra U-HE synthesizer.
Base your answer *only* on the USER_QUERY and the relevant DOCUMENTATION SNIPPETS and IMAGE DESCRIPTIONS provided below.
Synthesize the information from both text and image descriptions to answer the query comprehensively.
If the results do not contain the answer, state that clearly.
Use clear, technical language appropriate for Zebra users.
Maintain a professional and helpful tone.
Do not invent features or functionality not mentioned in the provided results.
        """
        human_message_content = (
            f"USER_QUERY: {user_query}\n\n"
            "RELEVANT DOCUMENTATION SNIPPETS:\n"
            "---\n"
            f"{chunks_str}\n"
            "---\n\n"
            "RELEVANT IMAGE DESCRIPTIONS FROM THE SAME PAGES AS SNIPPETS:\n"
            "---\n"
            f"{image_contexts_str}\n"
            "---"
        )
        llm_messages = [
             SystemMessage(content=zebra_system_prompt),
             HumanMessage(content=human_message_content)
         ]

        # --- Invoke LLM and return --- #
        print(f"Sending {len(text_chunks)} text chunks and {len(image_contexts_for_prompt)} unique image contexts to LLM.")
        final_image_path_list = sorted(list(image_paths_for_state))
        try:
            response = self.llm.invoke(llm_messages)
            return {
                "messages": [AIMessage(content=response.content)],
                "relevant_image_paths": final_image_path_list,
                "next": END
            }
        except Exception as e:
            print(f"PDF Questions LLM call failed: {e}")
            error_message = "Sorry, I encountered an error while generating the response using documentation and image context."
            return {"messages": [AIMessage(content=error_message)], "relevant_image_paths": [], "next": END}

    # --- Graph Compilation --- #
    def _compile_graph(self) -> None:
        """Compiles the LangGraph state machine."""
        print("Compiling agent graph...")
        graph = StateGraph(CustomState)

        # Add nodes
        graph.add_node("router", self._router_node)
        graph.add_node("general_response", self._general_response_node)
        graph.add_node("pdf_questions", self._pdf_questions_node)
        graph.add_node("vector_store", self._vector_store_node)

        # Set entry point
        graph.set_entry_point("router")

        # Add conditional edges from router
        graph.add_conditional_edges(
            "router",
            lambda x: x["next"],
            {
                "general_response": "general_response",
                "vector_store": "vector_store",
            }
        )

        # Add normal edges
        graph.add_edge("vector_store", "pdf_questions")
        graph.add_edge("pdf_questions", END)
        graph.add_edge("general_response", END)

        # Add persistence
        self.memory = MemorySaver()

        # Compile the graph
        self.compiled_graph = graph.compile(checkpointer=self.memory)
        print("Agent graph compiled successfully.")

    # --- Agent Invocation --- #
    def process_request(self, user_input: str, pdf_id: str, thread_id: str = "default_thread") -> tuple[str, list[str]]:
        """Processes a user query through the compiled agent graph. Returns text response and list of relevant image paths."""
        # Check if PDF environment is loaded
        if pdf_id not in self.vector_stores or pdf_id not in self.context_data:
             error_msg = f"Error: Environment for PDF ID '{pdf_id}' not loaded. Please process the PDF first."
             print(error_msg)
             return (error_msg, [])

        config = {"configurable": {"thread_id": thread_id}}
        # Initial state needs the active_pdf_id
        initial_state = {"messages": [HumanMessage(content=user_input)], "active_pdf_id": pdf_id}

        print(f"\n--- Processing Request (PDF ID: {pdf_id}, Thread: {thread_id}) ---")
        print(f"User Input: {user_input}")
        print("--------------------------------------------")

        final_message_content = "Error: No response generated."
        final_image_paths = [] # <<< ADDED: Initialize image paths list
        final_state_snapshot = {} # Capture the last state

        try:
            stream = self.compiled_graph.stream(
                initial_state,
                config=config,
                stream_mode="values"
            )

            final_ai_message = None
            print("--- Graph Execution --- ")
            for step_output in stream:
                last_event = list(step_output.keys())[-1]
                print(f"Node '{last_event}' executed.")
                final_state_snapshot = step_output # Keep track of the latest state
                if "messages" in step_output and step_output["messages"]:
                    current_message = step_output["messages"][-1]
                    if isinstance(current_message, AIMessage):
                        final_ai_message = current_message
            print("--- Graph Execution Finished ---")

            # Extract response and image paths from the final state
            if final_ai_message:
                final_message_content = final_ai_message.content
                # Retrieve image paths from the final state if available
                final_image_paths = final_state_snapshot.get("relevant_image_paths", [])
                print("\n--- Agent Final Response ---")
                print(final_message_content)
                print(f"-- Relevant Image Paths ({len(final_image_paths)}): {final_image_paths} --")
                print("---------------------------")
            else:
                # Handle cases where execution might end without an AIMessage in the last step
                if "messages" in final_state_snapshot and final_state_snapshot["messages"]:
                     last_msg_content = final_state_snapshot["messages"][-1].content
                     final_message_content = f"Processing ended unexpectedly. Last message: {last_msg_content}"
                else:
                      final_message_content = "Processing finished without generating a standard AI response."
                print(f"Warning: {final_message_content}")

        except Exception as e:
            print(f"Error during graph execution: {e}")
            import traceback
            traceback.print_exc()
            final_message_content = f"An error occurred during processing: {e}"
            final_image_paths = [] # Ensure empty list on error

        return final_message_content, final_image_paths # <<< CHANGED return

    def _get_current_resources(self, state: CustomState) -> Tuple[Optional[Chroma], Optional[LoadedContextsType]]:
        """Helper to safely get resources for the active PDF ID from agent state."""
        pdf_id = state.get("active_pdf_id")
        if not pdf_id:
            print(f"Error: active_pdf_id missing in state.")
            return None, None

        vs = self.vector_stores.get(pdf_id)
        if not vs:
            print(f"Error: Vector store for {pdf_id} not found.")
            return None, None

        ctx_data = self.context_data.get(pdf_id)
        if not ctx_data:
            print(f"Error: Context data for {pdf_id} not found.")
            return vs, None

        return vs, ctx_data

    # --- PDF Processing Pipeline --- #

    def get_pdf_id(self, pdf_file_path: Path) -> str:
         """Generates a unique ID based on PDF content hash."""
         hasher = hashlib.sha256()
         try:
             with open(pdf_file_path, 'rb') as f:
                 while chunk := f.read(8192):
                     hasher.update(chunk)
             # Use first 16 chars of hash for directory names
             return hasher.hexdigest()[:16]
         except OSError as e:
             print(f"Error reading PDF file for hashing: {e}")
             raise ValueError(f"Could not read PDF file {pdf_file_path} to generate ID.") from e

    def setup_pdf_environment(self, pdf_file_path: str) -> str:
        """Processes an uploaded PDF: extracts images/context, builds vector store. Returns pdf_id."""
        pdf_path = Path(pdf_file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        pdf_id = self.get_pdf_id(pdf_path)
        pdf_filename = pdf_path.name
        print(f"Processing PDF: {pdf_filename} (ID: {pdf_id})")

        # Define dynamic paths relative to the agent's execution directory
        script_dir = Path(__file__).parent
        pdf_data_dir = script_dir / DATA_DIR / pdf_id
        vs_path = pdf_data_dir / "vectorstore"
        img_dir = pdf_data_dir / "images"
        context_file_path = pdf_data_dir / "image_contexts.json"

        # Check if already processed and loaded in memory
        if pdf_id in self.vector_stores:
            print(f"PDF {pdf_id} already processed and loaded in memory.")
            # Quick check if context data is also loaded
            if pdf_id not in self.context_data:
                 print(f"Warning: VS for {pdf_id} loaded, but context data missing. Attempting reload.")
                 try:
                     # We need pdf_filename for the structure {pdf_id:{pdf_filename:{page:[Info]}}}
                     loaded_page_contexts = self._load_image_contexts_from_file(context_file_path)
                     self.context_data[pdf_id] = {pdf_filename: loaded_page_contexts} # Store correctly
                 except Exception as load_err:
                      print(f"Error reloading context data for {pdf_id}: {load_err}. Proceeding without image context.")
                      self.context_data[pdf_id] = {pdf_filename: {}} # Ensure key exists even if empty
            return pdf_id

        # Check if processed on disk but not loaded in memory
        if vs_path.exists() and context_file_path.exists() and any(vs_path.iterdir()):
             print(f"Found existing processed data for PDF {pdf_id} on disk. Loading...")
             try:
                 # Load vector store
                 self.vector_stores[pdf_id] = Chroma(persist_directory=str(vs_path), embedding_function=self.embedding_function)
                 # Load contexts directly into the main structure, associating with this pdf_id
                 loaded_page_contexts = self._load_image_contexts_from_file(context_file_path)
                 # Store under pdf_id -> pdf_filename -> page_num
                 self.context_data[pdf_id] = {pdf_filename: loaded_page_contexts}
                 print(f"Successfully loaded environment for PDF {pdf_id} from disk.")
                 return pdf_id
             except Exception as e:
                 print(f"Error loading existing data for {pdf_id} from disk: {e}. Reprocessing...")
                 if pdf_data_dir.exists(): shutil.rmtree(pdf_data_dir) # Clean up before reprocessing
        else:
             print(f"No complete pre-processed data found for PDF {pdf_id}. Starting full pipeline...")


        # --- Perform Full Processing Pipeline --- #
        # Ensure directories exist
        pdf_data_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(exist_ok=True)
        print(f"Starting new processing pipeline for PDF {pdf_id}...")
        try:
            # 1. Extract Images and Generate Context
            # This saves the JSON and returns the structured data {pdf_filename: {page: [Info]}}
            generated_context_data = self._extract_and_contextualize_images(pdf_path, img_dir, context_file_path)
            # Store structured data under pdf_id
            self.context_data[pdf_id] = generated_context_data

            # 2. Create Vector Store with Text and Context Metadata
            # Pass the context data we just generated/loaded
            vs = self._create_vector_store(pdf_path, vs_path, generated_context_data)
            self.vector_stores[pdf_id] = vs

            print(f"Successfully processed and loaded environment for PDF {pdf_id}.")
            return pdf_id

        except Exception as e:
            print(f"FATAL ERROR during PDF processing pipeline for {pdf_id}: {e}")
            # Clean up potentially corrupted directories if processing failed
            if pdf_data_dir.exists():
                print(f"Cleaning up failed processing directory: {pdf_data_dir}")
                shutil.rmtree(pdf_data_dir)
            # Remove from memory if partially loaded
            self.vector_stores.pop(pdf_id, None)
            self.context_data.pop(pdf_id, None)
            import traceback
            traceback.print_exc() # Show full error
            raise # Re-raise the exception to signal failure to Streamlit

    def _extract_and_contextualize_images(self, pdf_path: Path, img_output_dir: Path, context_json_path: Path) -> LoadedContextsType:
        """Extracts images, generates context using LLM, saves images and context JSON. Returns structured context data."""
        print(f"Extracting images and generating context for {pdf_path.name}...")
        results_list = [] # List to store dicts for JSON saving
        contexts_for_structure: Dict[int, List[ImageContextInfo]] = {} # For immediate use structure {page_num: [Info]}
        doc = None
        pdf_filename = pdf_path.name
        try:
            doc = fitz.open(pdf_path)
            total_images_processed = 0
            total_context_generated = 0
            for page_num_fitz in tqdm(range(len(doc)), desc=f"Processing Pages for Images [{pdf_filename}]"):
                page = doc.load_page(page_num_fitz)
                page_num = page_num_fitz + 1 # 1-based index
                img_index = 0
                image_info_list = page.get_image_info(hashes=True)
                if not image_info_list: continue

                page_contexts = [] # ImageContextInfo objects for this specific page

                for info in image_info_list:
                    try:
                        rect = fitz.Rect(info["bbox"])
                        if rect.is_empty or rect.width <= 1 or rect.height <= 1:
                            # print(f"Skipping small/empty image bbox on page {page_num}")
                            continue

                        # --- Image Saving --- #
                        pix = page.get_pixmap(clip=rect, dpi=150) # Moderate DPI
                        ref_id = info.get('xref', f'idx{img_index}')
                        img_filename = f"p{page_num:03d}_ref{ref_id}.png"
                        img_filepath = img_output_dir / img_filename
                        try:
                            img_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure image dir exists
                            pix.save(img_filepath)
                        except Exception as save_err:
                            print(f"Warning: Failed to save image {img_filename} on page {page_num}: {save_err}")
                            pix = None
                            continue # Skip if cannot save image
                        finally:
                            pix = None # Release memory

                        # --- Context Extraction --- #
                        context_margin = 30 # Pixels around bbox
                        page_rect = page.rect
                        context_rect = fitz.Rect(max(0, rect.x0 - context_margin), max(0, rect.y0 - context_margin),
                                                 min(page_rect.width, rect.x1 + context_margin), min(page_rect.height, rect.y1 + context_margin))
                        context_rect.normalize()
                        context_text = ""
                        if not context_rect.is_empty:
                             # Sort text blocks by vertical position for better context flow
                             context_text = page.get_text("text", clip=context_rect, sort=True).strip()

                        # --- LLM Context Generation --- #
                        image_context = self.image_context_llm.get_image_context(context_text, page_num, img_index)
                        total_images_processed += 1

                        # --- Store Results --- #
                        # Store relative path for JSON and internal structure
                        relative_img_path = img_filepath.relative_to(img_output_dir).as_posix()
                        result_item = {
                            "image_path": relative_img_path,
                            "page": page_num,
                            "bbox": [round(c, 2) for c in rect.irect], # Integer bbox
                            "surrounding_text_snippet": context_text[:200] + ("..." if len(context_text) > 200 else ""),
                            "llm_context": image_context
                        }
                        results_list.append(result_item)

                        # Store in memory structure if context is valid
                        if image_context and image_context != "Context unclear from surrounding text.":
                            page_contexts.append(ImageContextInfo(context=image_context, path=relative_img_path))
                            total_context_generated += 1

                    except Exception as img_err:
                        print(f"Warning: Error processing image {img_index} on page {page_num} (xref: {info.get('xref')}): {img_err}")
                    finally:
                        img_index += 1

                if page_contexts:
                    contexts_for_structure[page_num] = page_contexts

            # --- Save Context JSON --- #
            print(f"Saving image context JSON to {context_json_path} ({len(results_list)} entries)")
            context_json_path.parent.mkdir(parents=True, exist_ok=True) # Ensure containing dir exists
            with open(context_json_path, 'w', encoding='utf-8') as f:
                json.dump(results_list, f, indent=2, ensure_ascii=False)

            print(f"Finished image processing for {pdf_filename}. Found {total_images_processed} images, generated context for {total_context_generated}.")
            # Return dict keyed by the original PDF filename containing the page->[Info] structure
            return {pdf_filename: contexts_for_structure}

        except Exception as e:
            print(f"Error during image extraction/context generation for {pdf_filename}: {e}")
            import traceback
            traceback.print_exc()
            return {pdf_filename: {}} # Return empty on error
        finally:
            if doc: doc.close()

    def _load_image_contexts_from_file(self, context_json_path: Path) -> Dict[int, List[ImageContextInfo]]: # Returns pages dict
        """Loads image context data AND paths from a specific JSON file for ONE PDF."""
        contexts_and_paths_by_page: Dict[int, List[ImageContextInfo]] = {}
        try:
            if not context_json_path.exists():
                 print(f"Warning: Context file {context_json_path} not found during load attempt.")
                 return {} # Return empty structure

            print(f"Loading image contexts from {context_json_path}")
            with open(context_json_path, 'r', encoding='utf-8') as f:
                image_data = json.load(f)

            loaded_context_count = 0
            for item in image_data:
                try:
                    page_num = item['page']
                    context = item.get('llm_context', '')
                    img_path = item.get('image_path')
                    if img_path is None: continue # Skip if path is missing

                    if page_num not in contexts_and_paths_by_page:
                        contexts_and_paths_by_page[page_num] = []

                    # Store if context is meaningful
                    if context and context != "Context unclear from surrounding text.":
                        contexts_and_paths_by_page[page_num].append(ImageContextInfo(context=context, path=img_path))
                        loaded_context_count += 1
                except (KeyError, TypeError) as e:
                    print(f"Warning: Skipping context entry during load due to parsing error: {item}. Error: {e}")

            loaded_page_count = len(contexts_and_paths_by_page)
            print(f"Successfully loaded {loaded_context_count} contexts/paths across {loaded_page_count} pages from file.")
            # Return only the page-level dict for the single PDF being loaded
            return contexts_and_paths_by_page

        except Exception as e:
            print(f"Error loading image context file {context_json_path}: {e}")
            return {} # Return empty dict on error

    def _process_pdf_text(self, pdf_path: Path, loaded_page_contexts: Dict[int, List[ImageContextInfo]]) -> List[Document]:
        """Reads text page by page using pypdf, adding image context metadata from the provided structure."""
        docs = []
        pdf_filename = pdf_path.name
        print(f"Processing PDF text for: {pdf_filename}")
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    page_num = i + 1 # 1-based index
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Get corresponding ImageContextInfo objects using the pre-loaded page dictionary
                            page_context_infos = loaded_page_contexts.get(page_num, [])
                            # Combine only the context strings for metadata storage
                            combined_contexts = "\n".join(f"- {info.context}" for info in page_context_infos)
                            metadata = {
                                "source": pdf_filename,
                                "page": page_num,
                                "image_contexts_on_page": combined_contexts or "No relevant image contexts found on this page."
                            }
                            docs.append(Document(page_content=page_text, metadata=metadata))
                        # else: Skip pages with no text
                    except Exception as page_extract_err:
                         print(f"Warning: Error extracting text from page {page_num} of {pdf_filename}: {page_extract_err}")
                         # Optionally create a document with empty content but metadata?
                         # metadata = {"source": pdf_filename, "page": page_num, "error": "text extraction failed"}
                         # docs.append(Document(page_content="", metadata=metadata))
                         continue # Skip page if text extraction fails
        except Exception as e:
            print(f"Error processing text for {pdf_filename} with pypdf: {e}")
            import traceback
            traceback.print_exc()
            # Return potentially partial list of docs

        print(f"Created {len(docs)} page-level documents with text and metadata.")
        return docs

    def _create_vector_store(self, pdf_path: Path, vs_persist_path: Path, loaded_full_context_data: LoadedContextsType) -> Chroma:
        """Processes PDF text, adds metadata, chunks, embeds, and creates Chroma store."""
        pdf_filename = pdf_path.name
        # Ensure the target directory exists and is clean before creating
        if vs_persist_path.exists():
             print(f"Note: Removing existing directory {vs_persist_path} before creating new store.")
             shutil.rmtree(vs_persist_path)
        vs_persist_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent exists

        # Extract the page-level context dict for *this* PDF
        page_level_contexts = loaded_full_context_data.get(pdf_filename, {})
        if not page_level_contexts:
             print(f"Warning: No pre-loaded image context data found for {pdf_filename} during VS creation.")

        # 1. Process text and add metadata using the extracted page_level_contexts
        docs = self._process_pdf_text(pdf_path, page_level_contexts)
        if not docs:
            raise RuntimeError(f"Failed to process PDF text for vector store creation: {pdf_filename}")

        # 2. Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_splits = text_splitter.split_documents(docs)
        if not docs_splits:
             raise RuntimeError(f"Text splitting resulted in zero chunks for {pdf_filename}. Check PDF content and splitter settings.")
        print(f"Split {len(docs)} page documents into {len(docs_splits)} chunks.")

        # 3. Create and persist vector store
        try:
            print(f"Creating Chroma vector store with OpenAI embeddings at {vs_persist_path}...")
            vector_store = Chroma.from_documents(
                documents=docs_splits, # These splits inherit page metadata including image_contexts_on_page
                embedding=self.embedding_function,
                persist_directory=str(vs_persist_path) # Chroma expects string path
            )
            print(f"Vector store created and persisted at {vs_persist_path}")
            if vector_store is None:
                raise RuntimeError("Chroma.from_documents returned None.")
            return vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create vector store: {e}")

# --- Main Execution Block --- #
if __name__ == "__main__":
    # Define paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Use abspath for reliability
    pdf_folder = os.path.join(script_dir, "books")
    persist_dir = os.path.join(script_dir, ZebraAgent.DEFAULT_PERSIST_DIR)
    image_context_file = os.path.join(script_dir, ZebraAgent.DEFAULT_IMAGE_CONTEXT_FILE)

    print(f"Script Directory: {script_dir}")
    print(f"PDF Folder Path: {pdf_folder}")
    print(f"Persist Directory: {persist_dir}")
    print(f"Image Context File: {image_context_file}")

    if not os.path.isdir(pdf_folder):
        print(f"Error: PDF folder not found at '{pdf_folder}'")
    else:
        try:
            agent = ZebraAgent()
            # ... (rest of main block for testing queries) ...
            print("\n--- Running Example Queries ---")
            test_query = "What does the main grid mixer look like?"
            print(f"\n--- Sending Query 1: '{test_query}' ---")
            response, image_paths = agent.process_request(test_query, pdf_id="example_pdf_id")
            print("\n--- Example Queries Finished --- ")

        except (ValueError, FileNotFoundError) as init_err:
             print(f"Initialization Error: {init_err}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
