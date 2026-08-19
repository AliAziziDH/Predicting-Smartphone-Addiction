"""
Google AI Studio Engine Bridge (High-Capacity Gemini 3.1 Pro).
Serves as the high-throughput, token-free compute bridge for:
1. Deep Tabular Feature Engineering & Non-linear Interaction Synthesis.
2. Model Residual Analysis and Decision Intelligence Auditing.
3. Evolutionary Hyperparameter & Ensembling Formulation.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Union
from google import genai
from google.genai import types


class StudioEngine:
    """
    Dedicated Bridge to Google AI Studio with flagship Gemini 3.1 Pro.
    Operates with 1M+ context window and advanced mathematical reasoning.
    """

    PRIMARY_MODEL = "gemini-3.1-pro-preview"
    FALLBACK_MODEL = "gemini-3.7-flash"

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set. "
                "Please configure your Google AI Studio API Key."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.default_model = default_model or self.PRIMARY_MODEL
        self.total_prompt_tokens = 0
        self.total_candidate_tokens = 0
        self.total_calls = 0

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> str:
        """Generates plain text response with exponential backoff retry."""
        target_model = model or self.default_model

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )

                # Record token metrics
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self.total_prompt_tokens += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    self.total_candidate_tokens += getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                self.total_calls += 1

                return response.text.strip() if response.text else ""

            except Exception as e:
                last_err = e
                print(f"⚠️ [StudioEngine] Attempt {attempt}/{max_retries} failed for {target_model}: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        # Fallback to secondary model if primary fails
        if target_model != self.FALLBACK_MODEL:
            print(f"🔄 [StudioEngine] Falling back to {self.FALLBACK_MODEL}...")
            return self.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                model=self.FALLBACK_MODEL,
                temperature=temperature,
                max_retries=1,
            )

        raise RuntimeError(f"StudioEngine failed to generate content after retries: {last_err}")

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> Union[Dict[str, Any], List[Any]]:
        """Generates strictly parsed JSON output from Google AI Studio."""
        system_prompt = (system_instruction or "") + "\nOutput MUST be valid, parseable JSON only. Do not include markdown code blocks or preamble."
        raw_text = self.generate_text(
            prompt=prompt,
            system_instruction=system_prompt.strip(),
            model=model,
            temperature=temperature,
        )

        clean_text = raw_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]

        try:
            return json.loads(clean_text.strip())
        except json.JSONDecodeError as err:
            # Fallback parse attempt
            print(f"⚠️ [StudioEngine] JSON decode error: {err}. Raw text:\n{raw_text[:200]}")
            raise

    def get_usage_summary(self) -> str:
        """Returns cumulative token utilization summary on Google AI Studio."""
        return (
            f"⚡ [Google AI Studio Quota Tracker]\n"
            f"   Calls: {self.total_calls}\n"
            f"   Prompt Tokens: {self.total_prompt_tokens:,}\n"
            f"   Candidate Tokens: {self.total_candidate_tokens:,}\n"
            f"   Total Studio Tokens: {(self.total_prompt_tokens + self.total_candidate_tokens):,}"
        )


# Global singleton instance for easy import
_global_engine: Optional[StudioEngine] = None

def get_studio_engine(model: Optional[str] = None) -> StudioEngine:
    global _global_engine
    if _global_engine is None or (model and _global_engine.default_model != model):
        _global_engine = StudioEngine(default_model=model)
    return _global_engine
