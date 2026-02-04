"""
DeepSeek LLM Service - Text formatting using DeepSeek LLM API.

This service uses the DeepSeek LLM API (OpenAI-compatible) to format
extracted "INDICATIONS AND USAGE" text into concise bullet points
for frontend display.
"""

import logging
from typing import Optional

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


class DeepSeekLLMService:
    """Service for formatting text using DeepSeek LLM API."""
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_API_URL
        
        # Initialize OpenAI-compatible client with DeepSeek endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # Model to use - DeepSeek-V3 (latest)
        self.model = "deepseek-chat"
        
        # System prompt for formatting indications
        self.system_prompt = """You are a medical information assistant that formats FDA drug label indications.

Your task is to convert raw "INDICATIONS AND USAGE" text from FDA pharmaceutical labels into clear, concise bullet points.

Guidelines:
1. Extract each distinct indication/use case as a separate bullet point
2. Keep medical terminology accurate but make it readable
3. Include important qualifiers (patient population, conditions, limitations)
4. Remove redundant disclaimers and legal boilerplate
5. Use bullet format: "• Indication text"
6. If there are subsections (e.g., "1.1 Cancer Treatment"), preserve the hierarchy
8. If accelerated approval or limitations exist, note them clearly

Output format:
• First indication
• Second indication
  - Sub-indication if applicable
• Third indication

Do NOT include:
- Full prescribing information references
- Lengthy clinical trial descriptions
- Regulatory submission details"""
    
    async def count_indications(self, raw_text: str) -> int:
        """
        Count the number of distinct approved indications in the text.
        
        Args:
            raw_text: Indication text to analyze
            
        Returns:
            Integer count of indications
        """
        if not raw_text or len(raw_text.strip()) < 20:
            return 0
            
        try:
            prompt = """Analyze the following "INDICATIONS AND USAGE" text and count the number of distinct approved indications or conditions treated.
            
            Return ONLY a JSON object in this format:
            {"count": <integer>}
            
            Do not include any other text."""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": raw_text[:5000]} # Limit context
                ],
                max_tokens=100,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            import json
            data = json.loads(content)
            return int(data.get("count", 0))
            
        except Exception as e:
            logger.error(f"[DeepSeekLLM] Failed to count indications: {e}")
            return 0

    async def format_indications_as_bullets(self, raw_text: str) -> Optional[str]:
        """
        Format raw INDICATIONS AND USAGE text into bullet points.
        
        Args:
            raw_text: Raw extracted text from FDA label
            
        Returns:
            Formatted bullet points or None if failed
        """
        if not raw_text or len(raw_text.strip()) < 20:
            logger.warning("[DeepSeekLLM] Input text too short to format")
            return None
        
        if not self.api_key:
            logger.error("[DeepSeekLLM] API key not configured")
            return None
        
        try:
            logger.info(f"[DeepSeekLLM] Formatting {len(raw_text)} chars of indication text")
            
            # Truncate if too long (API has token limits)
            max_input_chars = 8000
            if len(raw_text) > max_input_chars:
                raw_text = raw_text[:max_input_chars] + "\n...[truncated]"
                logger.info("[DeepSeekLLM] Truncated input to 8000 chars")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Please format the following INDICATIONS AND USAGE section into concise bullet points:\n\n{raw_text}"}
                ],
                max_tokens=1500,
                temperature=0.3,  # Lower temperature for more consistent formatting
            )
            
            formatted_text = response.choices[0].message.content
            
            if formatted_text:
                logger.info(f"[DeepSeekLLM] Formatted output: {len(formatted_text)} chars")
                return formatted_text.strip()
            else:
                logger.warning("[DeepSeekLLM] Empty response from API")
                return None
                
        except Exception as e:
            logger.error(f"[DeepSeekLLM] API call failed: {e}")
            return None

    async def process_indications(self, raw_text: str) -> dict:
        """
        Format indications AND count them in a single pass.
        
        Args:
            raw_text: Raw extracted text
            
        Returns:
            Dict with 'formatted_text' (str) and 'indication_count' (int)
        """
        default_result = {"formatted_text": None, "indication_count": 0}
        
        if not raw_text or len(raw_text.strip()) < 20:
            return default_result
            
        try:
            prompt = """You are a medical information assistant.
            
            Task 1: Format the "INDICATIONS AND USAGE" text into concise bullet points.
            Task 2: Count the number of distinct approved indications/conditions.
            
            Guidelines:
            - Use bullet format: "• Indication text"
            - Keep medical terminology accurate but readable
            - Remove disclaimers
            
            Return ONLY a JSON object with these exact keys:
            {
                "formatted_text": "<string with bullet points>",
                "indication_count": <integer>
            }"""
            
            # Truncate if needed
            if len(raw_text) > 8000:
                raw_text = raw_text[:8000] + "\n...[truncated]"
                
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Process this text:\n\n{raw_text}"}
                ],
                max_tokens=2000,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            import json
            data = json.loads(content)
            
            return {
                "formatted_text": data.get("formatted_text"),
                "indication_count": int(data.get("indication_count", 0))
            }
            
        except Exception as e:
            logger.error(f"[DeepSeekLLM] Processing failed: {e}")
            return default_result
    
    async def close(self):
        """Close the HTTP client - noop for singleton safety."""
        # Don't actually close the client to avoid breaking the singleton
        # The openai client manages its own connection pool
        pass


def get_deepseek_llm_service() -> DeepSeekLLMService:
    """Get a DeepSeek LLM service instance.
    
    Creates a new instance each time to avoid closed client issues.
    The openai client manages its own connection pool efficiently.
    """
    return DeepSeekLLMService()
