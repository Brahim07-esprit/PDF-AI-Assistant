import abc
from langchain_core.messages import HumanMessage


class LLMContextGenerator(abc.ABC):
    """Abstract base class for LLM context generators."""
    @abc.abstractmethod
    def get_image_context(
            self,
            text_around_image: str,
            page_num: int,
            img_idx: int) -> str:
        pass


class MockLLMContextGenerator(LLMContextGenerator):
    """Mock generator for testing without API calls."""

    def __init__(self):
        print("INFO: Using Mock LLM Context Generator.")

    def get_image_context(
            self,
            text_around_image: str,
            page_num: int,
            img_idx: int) -> str:
        if text_around_image.strip():
            return f"[Mock Context for image {img_idx} on page {page_num}]"
        else:
            return "Context unclear from surrounding text."


class GeminiContextGenerator(LLMContextGenerator):
    """Generates context using Google Gemini via Langchain."""

    def __init__(
            self,
            api_key: str,
            model_name: str,
            temperature: float = 0.0):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature
            )
            print("INFO: Image Context Gemini client initialized successfully.")
        except ImportError:
            raise ImportError(
                "langchain-google-genai is required for GeminiContextGenerator. Please install it.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Image Context Gemini client: {e}")

    def get_image_context(
            self,
            text_around_image: str,
            page_num: int,
            img_idx: int) -> str:
        prompt_text = f"""
Analyze text near an image (index {img_idx}) on PDF page {page_num}. Describe the image\\'s likely purpose/context concisely. If unclear, state \\'Context unclear from surrounding text.\\'.

Text:
---
{text_around_image}
---

Context Description:"""
        try:
            response = self.model.invoke([HumanMessage(content=prompt_text)])
            return response.content.strip() if isinstance(
                response.content, str) else "Context generation failed."
        except Exception as e:
            print(
                f"ERROR: Gemini API call failed for image {img_idx}, page {page_num}: {e}")
            return "Error generating context via API."
