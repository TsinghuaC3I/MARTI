from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import threading
from typing import Dict, Optional

import ray

from marti.utils.logging_utils import init_logger
logger = init_logger(__name__)


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketRateLimiter:
    """Token bucket rate limiter (from VERL)."""
    
    def __init__(self, rate_limit: int, name: str):
        self.rate_limit = rate_limit
        self.name = name
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)
    
    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1
    
    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1
    
    def get_status(self):
        """Get current status."""
        return {
            "name": self.name,
            "rate_limit": self.rate_limit,
            "current_count": self.current_count,
            "available": self.rate_limit - self.current_count
        }


@ray.remote
class RateLimiterPool:
    """Manages rate limiters for different tools."""
    
    def __init__(self):
        self.limiters = {}
    
    def get_or_create_limiter(self, tool_name: str, rate_limit: int) -> TokenBucketRateLimiter:
        """Get existing or create new rate limiter for a tool."""
        if tool_name not in self.limiters:
            limiter_name = f"rate-limiter-{tool_name}"
            self.limiters[tool_name] = TokenBucketRateLimiter.options(
                name=limiter_name,
                get_if_exists=True
            ).remote(rate_limit, tool_name)
        return self.limiters[tool_name]
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all rate limiters."""
        status = {}
        for tool_name, limiter in self.limiters.items():
            status[tool_name] = ray.get(limiter.get_status.remote())
        return status

class BaseToolExecutor(ABC):
    """Base class for all tool executors."""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the tool with given parameters.
        
        Returns:
            Tuple of (response_text, metadata)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get tool name."""
        pass
