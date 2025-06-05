import streamlit as st
import os
from pathlib import Path
import sys
import tempfile

# Ensure the directory containing agent.py is in the Python path
# Assumes agent.py is in the same directory as streamlit_app.py
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Try importing the agent class
try:
    from agent import ZebraAgent, DATA_DIR # Import DATA_DIR constant
except ImportError as e:
    st.error(f"Fatal Error: Failed to import ZebraAgent from agent.py. Ensure it's in the same directory and has no import errors itself: {e}")
    st.stop()
except Exception as e:
    st.error(f"Fatal Error: An unexpected error occurred importing agent.py: {e}")
    st.stop()

# --- Configuration --- #
# Base directory where agent stores processed PDF data (relative to agent.py)
PROCESSED_DATA_BASE_DIR = current_dir / DATA_DIR

# --- Helper Functions --- #

@st.cache_resource(show_spinner="Initializing Agent Core...")
def load_agent_core():
    """Initializes the core ZebraAgent instance (without PDF data)."""
    try:
        agent = ZebraAgent()
        return agent
    except ValueError as ve:
        st.error(f"Agent Initialization Error: {ve}. Missing required API Key in .env?" )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during agent core initialization: {e}")
        import traceback
        st.exception(traceback.format_exc())
        return None

# --- Streamlit App UI --- #
st.set_page_config(page_title="Dynamic PDF Assistant", layout="wide")
st.title("📄 Dynamic PDF Assistant")
st.caption("Upload a PDF and ask questions about its content, including related images.")

# Load the core agent (cached)
agent = load_agent_core()

# --- Session State Initialization --- #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_pdf_id" not in st.session_state:
    st.session_state.active_pdf_id = None
if "active_pdf_name" not in st.session_state:
    st.session_state.active_pdf_name = None
if "thread_id" not in st.session_state:
    # Initialize thread ID only once
    st.session_state.thread_id = f"st-session-{os.urandom(8).hex()}"

# --- PDF Upload and Processing --- #

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Check if this file is different from the last successfully processed one
    new_file_identifier = (uploaded_file.name, uploaded_file.size)
    last_processed_identifier = st.session_state.get("last_processed_file_identifier")

    # Process if it's a new file or if no file was previously processed successfully
    if last_processed_identifier != new_file_identifier:
        if agent:
            st.info(f"Processing '{uploaded_file.name}'... This may take a while, especially for the first time.")
            # Save uploaded file to a temporary location for the agent to process
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            processing_success = False
            try:
                # Process the PDF using the agent's setup method
                with st.spinner("Analyzing PDF, extracting images, generating context, building knowledge base... Please wait."):
                    pdf_id = agent.setup_pdf_environment(tmp_file_path)

                # Update session state ONLY if processing was successful
                st.session_state.active_pdf_id = pdf_id
                st.session_state.active_pdf_name = uploaded_file.name
                st.session_state.messages = [] # Clear chat history for new PDF
                # Store identifier of the successfully processed file
                st.session_state.last_processed_file_identifier = new_file_identifier
                st.success(f"Successfully processed '{uploaded_file.name}' (ID: {pdf_id}). Ready for questions!")
                processing_success = True

            except FileNotFoundError as fnf_err:
                 st.error(f"Processing Error: {fnf_err}")
                 # Reset state if processing failed
                 st.session_state.active_pdf_id = None
                 st.session_state.active_pdf_name = None
                 st.session_state.last_processed_file_identifier = None # Allow re-upload attempt
            except Exception as e:
                st.error(f"An unexpected error occurred during PDF processing: {e}")
                import traceback
                st.exception(traceback.format_exc())
                # Reset state if processing failed
                st.session_state.active_pdf_id = None
                st.session_state.active_pdf_name = None
                st.session_state.last_processed_file_identifier = None # Allow re-upload attempt
            finally:
                # Clean up temporary file
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
        else:
            st.error("Agent core is not loaded. Cannot process PDF.")
    # else: # Optional: Message indicating the same file is already loaded
       # st.info(f"PDF '{st.session_state.active_pdf_name}' is already loaded and processed.")

elif st.session_state.active_pdf_id:
     st.info(f"Currently querying: **{st.session_state.active_pdf_name}** (ID: {st.session_state.active_pdf_id})", icon="📄")


# --- Chat Interface --- #

if not agent:
    st.error("Agent Core failed to initialize. Cannot start chat.")
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display MAX ONE image associated with previous assistant messages
            if message["role"] == "assistant" and "image_paths" in message and message["image_paths"] and st.session_state.active_pdf_id:
                first_image_rel_path = message["image_paths"][0]
                # Construct full path using the ACTIVE pdf_id and base data dir
                full_image_path = PROCESSED_DATA_BASE_DIR / st.session_state.active_pdf_id / "images" / first_image_rel_path
                if full_image_path.exists():
                    st.image(str(full_image_path), caption=Path(first_image_rel_path).name, use_container_width=True)
                else:
                    st.warning(f"Image not found at expected path: {full_image_path}")

    # Chat input - Enable only if a PDF is processed
    prompt = st.chat_input(f"Ask about '{st.session_state.active_pdf_name}'..." if st.session_state.active_pdf_id else "Upload a PDF to start chatting...", disabled=not st.session_state.active_pdf_id)

    if prompt and st.session_state.active_pdf_id:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_text, image_paths = agent.process_request(
                        user_input=prompt,
                        pdf_id=st.session_state.active_pdf_id, # Pass the active PDF ID
                        thread_id=st.session_state.thread_id
                    )
                    st.markdown(response_text)

                    # Display MAX ONE image if any were returned
                    if image_paths:
                        st.markdown("--- Relevant Image ---")
                        first_image_rel_path = image_paths[0]
                        # Construct full path dynamically
                        full_image_path = PROCESSED_DATA_BASE_DIR / st.session_state.active_pdf_id / "images" / first_image_rel_path
                        if full_image_path.exists():
                             st.image(str(full_image_path), caption=Path(first_image_rel_path).name, use_container_width=True)
                        else:
                             st.warning(f"Image not found at expected path: {full_image_path}")
                    # else: # Optional: Message if no images are relevant
                        # st.write("_(No specific images identified as relevant to this response.)_")

                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "image_paths": image_paths
                    })

                except Exception as e:
                    error_msg = f"An error occurred processing your request: {e}"
                    st.error(error_msg)
                    import traceback
                    st.exception(traceback.format_exc())
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"}) 