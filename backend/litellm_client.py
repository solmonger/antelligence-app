"""
LiteLLM client for Antelligence simulation.

This module replaces direct OpenAI/Google/Mistral API calls with LiteLLM
to route through the local LiteLLM proxy at http://host.orb.internal:4000
"""

import os
import requests
import json
from typing import Dict, List, Optional, Any


class LiteLLMClient:
    """Client for LiteLLM proxy API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LiteLLM client.
        
        Args:
            api_key: API key for LiteLLM (defaults to environment variable)
            base_url: Base URL for LiteLLM proxy (defaults to http://host.orb.internal:4000)
        """
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-litellm-openclaw-local-2026")
        self.base_url = base_url or os.getenv("LITELLM_URL", "http://host.orb.internal:4000")
        
    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Create a chat completion using LiteLLM.
        
        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus", "gemini-pro")
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            
        Returns:
            Dict with completion response
            
        Raises:
            Exception: If API call fails
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"LiteLLM API request failed: {e}")
            
    def simple_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Simple wrapper for single-prompt completions.
        
        Args:
            model: Model name
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response["choices"][0]["message"]["content"].strip()


# Global client instance
_client = None

def get_client() -> LiteLLMClient:
    """Get or create global LiteLLM client instance."""
    global _client
    if _client is None:
        _client = LiteLLMClient()
    return _client