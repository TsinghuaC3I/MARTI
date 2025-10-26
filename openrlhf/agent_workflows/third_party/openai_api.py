import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_TIMEOUT = 60
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2

logger = logging.getLogger(__name__)


def call_openai_api(
    api_base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call OpenAI compatible API with retry logic.
    
    Args:
        api_base_url: API base URL
        api_key: API key
        model: Model name (e.g., "gpt-4", "o1-preview")
        messages: Message list
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        timeout: Request timeout in seconds
        **kwargs: Additional API parameters
        
    Returns:
        Tuple of (response_json, error_message)
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[API-{request_id[:8]}] "
    
    endpoint = f"{api_base_url.rstrip('/')}/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
    }
    
    # Add parameters based on model type
    if not model.startswith(("o1", "o3", "o4")):  # Chat models
        payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
    else:  # Reasoning models
        if max_tokens:
            payload["max_completion_tokens"] = max_tokens
            
    payload.update(kwargs)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}: Calling {model}")
            
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            
            # Retry on server errors and rate limits
            if response.status_code in [429, 500, 502, 503, 504]:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.warning(f"{log_prefix}Server error {response.status_code}, retrying in {delay}s")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                continue
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"{log_prefix}Success on attempt {attempt + 1}")
            return result, None
            
        except requests.exceptions.Timeout:
            logger.warning(f"{log_prefix}Timeout on attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"{log_prefix}Connection error on attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
            
        except Exception as e:
            return None, f"API call failed: {e}"
    
    return None, f"API call failed after {MAX_RETRIES} retries"


def extract_content(response: Dict[str, Any]) -> Optional[str]:
    """Extract content from API response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def get_usage_info(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract usage information from API response."""
    usage = response.get("usage", {})
    
    # Handle both chat and reasoning models
    result = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    
    # Reasoning models have additional reasoning_tokens
    completion_details = usage.get("completion_tokens_details", {})
    if "reasoning_tokens" in completion_details:
        result["reasoning_tokens"] = completion_details["reasoning_tokens"]
        
    return result


def perform_inference(
    api_base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Perform inference with OpenAI compatible API.
    
    Args:
        api_base_url: API base URL
        api_key: API key  
        model: Model name
        prompt: User prompt
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Max tokens
        timeout: Timeout in seconds
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (response_text, metadata)
    """
    start_time = time.time()
    
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Call API
    response, error = call_openai_api(
        api_base_url=api_base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs
    )
    
    elapsed = time.time() - start_time
    
    # Build metadata
    metadata = {
        "model": model,
        "elapsed_time": elapsed,
        "status": "unknown",
        "usage": {},
        "error": error,
    }
    
    if error:
        metadata["status"] = "error"
        return None, metadata
    
    if response:
        content = extract_content(response)
        usage = get_usage_info(response)
        
        metadata.update({
            "status": "success" if content else "empty_response",
            "usage": usage,
        })
        
        logger.info(f"Inference completed: {usage.get('total_tokens', 0)} total tokens in {elapsed:.2f}s")
        return content, metadata
    
    metadata["status"] = "unknown_error"
    return None, metadata


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Chat model example
    response, metadata = perform_inference(
        api_base_url="https://api.openai.com/v1",
        api_key="your-api-key",
        model="gpt-4",
        prompt="Explain reinforcement learning briefly",
        temperature=0.7,
        max_tokens=500
    )
    
    if response:
        print("Response:", response[:100] + "...")
        print("Usage:", metadata["usage"])
    
    # Reasoning model example  
    response, metadata = perform_inference(
        api_base_url="https://api.openai.com/v1",
        api_key="your-api-key", 
        model="o1-preview",
        prompt="Solve this complex math problem step by step",
        max_tokens=1000
    )
    
    if response:
        print("Reasoning Response:", response[:100] + "...")
        print("Usage:", metadata["usage"])
        if "reasoning_tokens" in metadata["usage"]:
            print(f"Reasoning tokens used: {metadata['usage']['reasoning_tokens']}")