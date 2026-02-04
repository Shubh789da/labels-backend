"""DeepSeek-OCR-2 model wrapper for document OCR."""
import os
import logging
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoModel, AutoTokenizer

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DeepSeekOCR:
    """Wrapper for DeepSeek-OCR-2 model.

    DeepSeek-OCR-2 uses a causal flow vision encoder (DeepEncoder V2)
    that processes documents more like human reading - flexibly adjusting
    reading order based on content semantics.

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = settings.DEVICE
        self._loaded = False

    def load_model(self):
        """Load the DeepSeek-OCR-2 model and tokenizer."""
        if self._loaded:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading DeepSeek-OCR-2 model: {settings.MODEL_NAME}")

        try:
            # Set HuggingFace token if provided
            if settings.HF_TOKEN:
                os.environ["HF_TOKEN"] = settings.HF_TOKEN

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True,
            )

            # Load model with flash attention for efficiency
            self.model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_safetensors=True,
            )

            # Set to eval mode and move to GPU with bfloat16
            self.model = self.model.eval().cuda().to(torch.bfloat16)

            self._loaded = True
            logger.info("DeepSeek-OCR-2 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def process_image(
        self,
        image_path: Path,
        output_dir: Path,
        convert_to_markdown: bool = True,
    ) -> str:
        """Process a single image with OCR.

        Args:
            image_path: Path to the image file
            output_dir: Directory to save intermediate results
            convert_to_markdown: If True, convert to markdown format

        Returns:
            Extracted text (markdown formatted if requested)
        """
        if not self._loaded:
            self.load_model()

        # Choose prompt based on output format
        if convert_to_markdown:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        else:
            prompt = "<image>\nFree OCR."

        try:
            # Run inference using model's built-in infer method
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_path),
                output_path=str(output_dir),
                base_size=settings.BASE_SIZE,
                image_size=settings.IMAGE_SIZE,
                crop_mode=settings.CROP_MODE,
                save_results=False,  # We'll handle saving ourselves
            )

            # Extract text from result
            if isinstance(result, dict):
                text = result.get("text", "") or result.get("content", "") or str(result)
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)

            logger.debug(f"Processed {image_path.name}: {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return f"[Error processing page: {e}]"

    def process_multiple_images(
        self,
        image_paths: list[Path],
        output_dir: Path,
    ) -> str:
        """Process multiple images and combine into single markdown.

        Args:
            image_paths: List of image file paths (ordered by page number)
            output_dir: Directory for intermediate results

        Returns:
            Combined markdown text from all pages
        """
        if not self._loaded:
            self.load_model()

        markdown_parts = []
        total_pages = len(image_paths)

        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing page {i}/{total_pages}: {image_path.name}")

            # Add page separator
            markdown_parts.append(f"\n\n---\n## Page {i}\n\n")

            # Process the image
            page_text = self.process_image(
                image_path,
                output_dir,
                convert_to_markdown=True,
            )
            markdown_parts.append(page_text)

        # Combine all parts
        combined_markdown = "".join(markdown_parts)

        # Add header
        header = f"# OCR Document\n\n**Total Pages:** {total_pages}\n\n---\n"
        combined_markdown = header + combined_markdown

        return combined_markdown

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


# Global model instance (singleton)
_ocr_model: Optional[DeepSeekOCR] = None


def get_ocr_model() -> DeepSeekOCR:
    """Get or create the global OCR model instance."""
    global _ocr_model
    if _ocr_model is None:
        _ocr_model = DeepSeekOCR()
    return _ocr_model
