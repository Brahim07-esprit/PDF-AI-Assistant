import os
import json
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import shutil
import hashlib

import pypdf
from dotenv import load_dotenv

import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END

from .llm_context_generation import LLMContextGenerator, MockLLMContextGenerator, GeminiContextGenerator
from .pdf_processing_utils import PDFProcessor, ImageContextInfo, LoadedContextsType, DATA_DIR

load_dotenv()


class CustomState(MessagesState):
    retrieved_chunks: Optional[List[Document]] = None
    user_query: Optional[str] = None
    relevant_image_paths: Optional[List[str]] = None
    active_pdf_id: Optional[str] = None


class ZebraAgent:
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    DEFAULT_LLM_MODEL = "gemini-1.5-flash-latest"
    DEFAULT_IMG_CONTEXT_LLM_MODEL = "gemini-1.5-flash-latest"

    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.google_api_key or not self.openai_api_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY or OPENAI_API_KEY in environment.")

        self.embedding_function = OpenAIEmbeddings(
            model=self.DEFAULT_EMBEDDING_MODEL)
        self.llm = ChatGoogleGenerativeAI(
            model=self.DEFAULT_LLM_MODEL,
            temperature=0,
            google_api_key=self.google_api_key
        )
        try:
            self.image_context_llm: LLMContextGenerator = GeminiContextGenerator(
                api_key=self.google_api_key,
                model_name=self.DEFAULT_IMG_CONTEXT_LLM_MODEL,
                temperature=0.0)
        except Exception as e:
            print(
                f"WARNING: Failed to initialize Gemini for image context ({e}). Using MockLLMContextGenerator.")
            self.image_context_llm = MockLLMContextGenerator()

        self.pdf_processor = PDFProcessor(
            self.image_context_llm, self.embedding_function)

        self.vector_stores: Dict[str, Chroma] = {}
        self.context_data: Dict[str, LoadedContextsType] = {}
        self._compile_graph()

    def _router_node(self, state: CustomState) -> dict:
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            print("ERROR: Router: Last message is not a HumanMessage")
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
            classification = self.llm.invoke(
                [HumanMessage(content=router_prompt)])
            classification_text = classification.content.strip()
            if "ZEBRA_QUERY" in classification_text:
                return {
                    "next": "vector_store",
                    "user_query": last_message.content}
            else:
                return {"next": "general_response"}
        except Exception as e:
            print(f"ERROR: Router LLM call failed: {e}")
            return {"next": "general_response"}

    def _general_response_node(self, state: CustomState) -> dict:
        messages = state["messages"]
        general_system_prompt = """
You are a helpful AI assistant. Answer the user's query concisely.
If the user asks specifically about the Zebra U-HE synthesizer here, politely state that another part of the system handles those questions.
        """
        llm_messages = [SystemMessage(content=general_system_prompt)]
        if messages and isinstance(messages[-1], HumanMessage):
            llm_messages.append(messages[-1])
        else:
            return {
                "messages": [
                    AIMessage(
                        content="I seem to have lost the context. Could you please repeat your question?")],
                "next": END}
        try:
            response = self.llm.invoke(llm_messages)
            return {
                "messages": [
                    AIMessage(
                        content=response.content)],
                "next": END}
        except Exception as e:
            print(f"ERROR: General Response LLM call failed: {e}")
            return {
                "messages": [
                    AIMessage(
                        content="Sorry, I encountered an error trying to respond.")],
                "next": END}

    def _vector_store_node(self, state: CustomState) -> dict:
        vector_store, _ = self._get_current_resources(state)
        user_query = state.get("user_query")

        if not vector_store:
            error_msg = "Vector store not available for the active PDF."
            print(f"ERROR: Vector Store Node: {error_msg}")
            return {
                "messages": [
                    AIMessage(
                        content=f"Error: {error_msg}")],
                "relevant_image_paths": [],
                "next": END}
        if not user_query:
            error_msg = "User query missing from state."
            print(f"ERROR: Vector Store Node: {error_msg}")
            return {
                "messages": [
                    AIMessage(
                        content=f"Error: {error_msg}")],
                "relevant_image_paths": [],
                "next": END}

        retrieved_docs = []
        try:
            results_with_scores = vector_store.similarity_search_with_relevance_scores(
                query=user_query, k=5)
            retrieved_docs = [doc for doc,
                              score in results_with_scores if score > 0.7]
            if not retrieved_docs and results_with_scores:
                retrieved_docs = [results_with_scores[0][0]]
        except Exception as e:
            print(f"ERROR: Vector store search failed: {e}")
            return {
                "messages": [
                    AIMessage(
                        content="Error searching document database.")],
                "relevant_image_paths": [],
                "next": END}
        return {"retrieved_chunks": retrieved_docs, "next": "pdf_questions"}

    def _pdf_questions_node(self, state: CustomState) -> dict:
        vector_store, full_context_data_for_pdf = self._get_current_resources(
            state)
        retrieved_docs: List[Document] = state.get("retrieved_chunks", [])
        user_query = state.get("user_query", "Unknown query")
        active_pdf_id = state.get("active_pdf_id")

        if not retrieved_docs:
            no_context_response = "I couldn't find specific information related to your query in the processed PDF documentation."
            return {
                "messages": [
                    AIMessage(
                        content=no_context_response)],
                "relevant_image_paths": [],
                "next": END}

        if not full_context_data_for_pdf or not active_pdf_id:
            print(
                f"ERROR: PDF Questions Node: Image context data or active_pdf_id missing for PDF ID: {active_pdf_id}.")
            text_chunks_for_fallback = [
                doc.page_content for doc in retrieved_docs]
            fallback_prompt = f"""
USER_QUERY: {user_query}

RELEVANT DOCUMENTATION SNIPPETS (image context unavailable):
---
{"\n\n---\n\n".join(text_chunks_for_fallback)}
---"""
            try:
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content="Answer based on the provided snippets."), HumanMessage(
                            content=fallback_prompt)])
                return {
                    "messages": [
                        AIMessage(
                            content=response.content)],
                    "relevant_image_paths": [],
                    "next": END}
            except Exception as e:
                print(
                    f"ERROR: Fallback LLM call failed in _pdf_questions_node: {e}")
                return {
                    "messages": [
                        AIMessage(
                            content="Error: Image context data is missing, and fallback response generation failed.")],
                    "relevant_image_paths": [],
                    "next": END}

        text_chunks = []
        image_contexts_for_prompt = set()
        image_paths_for_state = set()

        for doc in retrieved_docs:
            text_chunks.append(doc.page_content)
            source_pdf_filename = doc.metadata.get("source")
            page_num = doc.metadata.get("page")
            if source_pdf_filename and page_num:
                pdf_specific_contexts = full_context_data_for_pdf.get(
                    source_pdf_filename)
                if pdf_specific_contexts:
                    page_specific_image_infos: List[ImageContextInfo] = pdf_specific_contexts.get(
                        page_num, [])
                    for img_info in page_specific_image_infos:
                        image_contexts_for_prompt.add(img_info.context)
                        image_paths_for_state.add(img_info.path)

        chunks_str = "\n\n---\n\n".join(text_chunks)
        image_contexts_str = "\n".join(f"- {ctx}" for ctx in sorted(list(image_contexts_for_prompt))) \
            if image_contexts_for_prompt else "No relevant image descriptions found for the retrieved text snippets."

        zebra_system_prompt = """
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
        final_image_path_list = sorted(list(image_paths_for_state))
        try:
            response = self.llm.invoke(llm_messages)
            return {
                "messages": [AIMessage(content=response.content)],
                "relevant_image_paths": final_image_path_list,
                "next": END
            }
        except Exception as e:
            print(f"ERROR: PDF Questions LLM call failed: {e}")
            error_message = "Sorry, I encountered an error while generating the response using documentation and image context."
            return {
                "messages": [
                    AIMessage(
                        content=error_message)],
                "relevant_image_paths": [],
                "next": END}

    def _compile_graph(self) -> None:
        graph = StateGraph(CustomState)
        graph.add_node("router", self._router_node)
        graph.add_node("general_response", self._general_response_node)
        graph.add_node("pdf_questions", self._pdf_questions_node)
        graph.add_node("vector_store", self._vector_store_node)
        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router", lambda x: x["next"], {
                "general_response": "general_response", "vector_store": "vector_store"})
        graph.add_edge("vector_store", "pdf_questions")
        graph.add_edge("pdf_questions", END)
        graph.add_edge("general_response", END)
        self.memory = MemorySaver()
        self.compiled_graph = graph.compile(checkpointer=self.memory)

    def process_request(self,
                        user_input: str,
                        pdf_id: str,
                        thread_id: str = "default_thread") -> tuple[str,
                                                                    list[str]]:
        if pdf_id not in self.vector_stores or pdf_id not in self.context_data:
            error_msg = f"ERROR: Environment for PDF ID '{pdf_id}' not loaded. Please process the PDF first."
            print(error_msg)
            return (error_msg, [])

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [
                HumanMessage(
                    content=user_input)],
            "active_pdf_id": pdf_id,
            "user_query": user_input}

        final_message_content = "Error: No response generated."
        final_image_paths = []
        final_state_snapshot = None

        try:
            stream = self.compiled_graph.stream(
                initial_state, config=config, stream_mode="values")
            for value in stream:
                final_state_snapshot = value
            if final_state_snapshot:
                final_messages = final_state_snapshot.get("messages", [])
                if final_messages and isinstance(
                        final_messages[-1], AIMessage):
                    final_message_content = final_messages[-1].content
                final_image_paths = final_state_snapshot.get(
                    "relevant_image_paths", [])
                if final_image_paths is None:
                    final_image_paths = []
        except Exception as e:
            error_msg = f"Fatal error during agent graph execution: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            final_message_content = f"Sorry, an unexpected error occurred: {e}"
            final_image_paths = []

        return final_message_content, final_image_paths

    def _get_current_resources(self,
                               state: CustomState) -> Tuple[Optional[Chroma],
                                                            Optional[LoadedContextsType]]:
        active_pdf_id = state.get("active_pdf_id")
        if not active_pdf_id:
            print("ERROR: _get_current_resources: active_pdf_id missing from state.")
            return None, None
        vector_store = self.vector_stores.get(active_pdf_id)
        context_data = self.context_data.get(active_pdf_id)
        if not vector_store:
            print(
                f"ERROR: _get_current_resources: No vector store found for PDF ID '{active_pdf_id}'.")
        if not context_data:
            print(
                f"ERROR: _get_current_resources: No context data found for PDF ID '{active_pdf_id}'.")
        return vector_store, context_data

    def setup_pdf_environment(self, pdf_file_path: str) -> str:
        pdf_path = Path(pdf_file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        pdf_id = self.pdf_processor.get_pdf_id(pdf_path)
        pdf_filename = pdf_path.name

        current_script_dir = Path(__file__).parent
        pdf_specific_data_dir = current_script_dir / DATA_DIR / pdf_id
        vs_path = pdf_specific_data_dir / "vectorstore"
        img_dir = pdf_specific_data_dir / "images"
        context_file_path = pdf_specific_data_dir / "image_contexts.json"

        if pdf_id in self.vector_stores:
            if pdf_id not in self.context_data:
                print(
                    f"WARNING: VS for {pdf_id} loaded, but context data missing. Attempting reload from {context_file_path}.")
                try:
                    loaded_page_contexts = self.pdf_processor._load_image_contexts_from_file(
                        context_file_path)
                    self.context_data[pdf_id] = {
                        pdf_filename: loaded_page_contexts}
                except Exception as load_err:
                    print(
                        f"ERROR: Failed to reload context data for {pdf_id}: {load_err}. Proceeding with potentially impaired image context.")
                    self.context_data[pdf_id] = {pdf_filename: {}}
            return pdf_id

        if vs_path.exists() and context_file_path.exists() and any(vs_path.iterdir()):
            try:
                self.vector_stores[pdf_id] = Chroma(
                    persist_directory=str(vs_path),
                    embedding_function=self.embedding_function)
                loaded_page_contexts = self.pdf_processor._load_image_contexts_from_file(
                    context_file_path)
                self.context_data[pdf_id] = {
                    pdf_filename: loaded_page_contexts}
                return pdf_id
            except Exception as e:
                print(
                    f"ERROR: Error loading existing data for {pdf_id} from disk: {e}. Reprocessing...")
                if pdf_specific_data_dir.exists():
                    shutil.rmtree(pdf_specific_data_dir)

        pdf_specific_data_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        try:
            generated_context_data = self.pdf_processor._extract_and_contextualize_images(
                pdf_path, img_dir, context_file_path)
            self.context_data[pdf_id] = generated_context_data
            vs = self.pdf_processor._create_vector_store(
                pdf_path, vs_path, generated_context_data)
            self.vector_stores[pdf_id] = vs
            return pdf_id
        except Exception as e:
            print(
                f"FATAL ERROR during PDF processing pipeline for {pdf_id}: {e}")
            if pdf_specific_data_dir.exists():
                shutil.rmtree(pdf_specific_data_dir)
            self.vector_stores.pop(pdf_id, None)
            self.context_data.pop(pdf_id, None)
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(
        __file__))
    pdf_folder = os.path.join(script_dir, "books")

    if not os.path.isdir(pdf_folder):
        print(f"Error: PDF folder not found at '{pdf_folder}'")
    else:
        try:
            agent = ZebraAgent()
            test_query = "What does the main grid mixer look like?"
            response, image_paths = agent.process_request(
                test_query, pdf_id="example_pdf_id")
            print("\n--- Example Queries Finished --- ")

        except (ValueError, FileNotFoundError) as init_err:
            print(f"Initialization Error: {init_err}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
