import httpx
import logging
from typing import List

logger = logging.getLogger(__name__)

# Fallbacks ensure the UI never breaks if an API key is missing or invalid
FALLBACK_MODELS = {
    "OpenAI": ["gpt-5.2-pro", "gpt-5.2-thinking", "gpt-5.2-instant", "o3-mini"],
    "Anthropic": ["claude-4.6-opus", "claude-4.5-sonnet", "claude-4.5-haiku"],
    "Google": ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
    "Ollama": ["deepseek-r1", "llama-3.3", "mistral", "qwen2.5"]
}


async def fetch_available_models(provider: str, api_key: str = "", base_url: str = "") -> List[str]:
    """Dynamically fetches available models from the specified provider's API."""
    timeout = httpx.Timeout(10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if provider == "OpenAI":
                if not api_key:
                    return FALLBACK_MODELS["OpenAI"]
                headers = {"Authorization": f"Bearer {api_key}"}
                response = await client.get("https://api.openai.com/v1/models", headers=headers)
                response.raise_for_status()
                data = response.json()
                # Filter for core chat/reasoning models
                models = [m["id"] for m in data.get("data", []) if m["id"].startswith(("gpt", "o1", "o3"))]
                models.sort(reverse=True)
                return models

            elif provider == "Anthropic":
                if not api_key:
                    return FALLBACK_MODELS["Anthropic"]
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
                response = await client.get("https://api.anthropic.com/v1/models", headers=headers)
                response.raise_for_status()
                data = response.json()
                models = [m["id"] for m in data.get("data", []) if m["id"].startswith("claude")]
                return models

            elif provider == "Google":
                if not api_key:
                    return FALLBACK_MODELS["Google"]
                response = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
                response.raise_for_status()
                data = response.json()
                # Gemini returns names as 'models/gemini-3.1-pro', strip the prefix
                models = [m["name"].replace("models/", "") for m in data.get("models", []) if
                          "gemini" in m["name"].lower()]
                models.sort(reverse=True)
                return models

            elif provider == "Ollama":
                url = base_url.rstrip("/") if base_url else "http://localhost:11434"
                response = await client.get(f"{url}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return models

    except Exception as e:
        logger.warning(f"Failed to fetch {provider} models dynamically: {e}. Defaulting to fallbacks.")

    return FALLBACK_MODELS.get(provider, [])