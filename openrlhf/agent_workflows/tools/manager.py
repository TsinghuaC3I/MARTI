import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from collections import defaultdict
from contextlib import asynccontextmanager

import ray

from openrlhf.agent_workflows.tools.base import RateLimiterPool
from openrlhf.agent_workflows.tools.base import BaseToolExecutor
from openrlhf.agent_workflows.tools.parser import ToolParser
from openrlhf.utils.logging_utils import init_logger
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

@ray.remote
class ToolMetricsCollector:
    """Centralized metrics collector for all tools."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0, 
            "failed_calls": 0,
            "total_latency": 0.0,
            "rate_limited": 0,
            "errors": defaultdict(int)  # Error type -> count
        })
        self.start_time = time.time()
    
    def record_call(self, tool_name: str, success: bool, latency: float,
                   rate_limited: bool = False, error: Optional[str] = None):
        """Record a tool call."""
        metrics = self.metrics[tool_name]
        metrics["total_calls"] += 1
        
        if success:
            metrics["successful_calls"] += 1
        else:
            metrics["failed_calls"] += 1
            if error:
                # Count by error type
                error_type = error.split(":")[0]
                metrics["errors"][error_type] += 1
        
        metrics["total_latency"] += latency
        
        if rate_limited:
            metrics["rate_limited"] += 1
        
        # Calculate average latency
        if metrics["total_calls"] > 0:
            metrics["avg_latency"] = metrics["total_latency"] / metrics["total_calls"]
    
    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for specific tool or all tools."""
        if tool_name:
            return dict(self.metrics.get(tool_name, {}))
        
        all_metrics = {}
        for name, metrics in self.metrics.items():
            all_metrics[name] = dict(metrics)
            all_metrics[name]["errors"] = dict(metrics["errors"])
        
        # Add summary
        summary = {
            "total_calls": sum(m["total_calls"] for m in all_metrics.values()),
            "successful_calls": sum(m["successful_calls"] for m in all_metrics.values()),
            "failed_calls": sum(m["failed_calls"] for m in all_metrics.values()),
            "uptime": time.time() - self.start_time,
            "tools_count": len(all_metrics)
        }
        
        return {
            "tools": all_metrics,
            "summary": summary
        }


class ToolManager:
    """Generic tool manager for any tool type."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tool_parser = ToolParser()
        self.tool_executors: Dict[str, BaseToolExecutor] = {}
        self.available_tools = self.tool_executors.keys()
        
        # Initialize rate limiter pool
        self.rate_limiter_pool = None
        if config.get("enable_rate_limiting", True):
            self.rate_limiter_pool = RateLimiterPool.options(
                name="global-rate-limiter-pool",
                get_if_exists=True
            ).remote()
        
        # Initialize metrics collector
        self.metrics_collector = None
        if config.get("enable_metrics", True):
            self.metrics_collector = ToolMetricsCollector.options(
                name="global-tool-metrics",
                get_if_exists=True
            ).remote()
    
    def set_tools(self, tools):
        self.tools = tools

    def get_max_turns(self):
        return self.config.get("max_turns", 3)
    
    def get_num_workers(self):
        return self.config.get("num_workers", 128)
    
    def register_tool(self, tool_name: str, executor: BaseToolExecutor):
        """Register a tool executor."""
        self.tool_executors[tool_name] = executor
        logger.info(f"Registered tool: {tool_name}")
    
    def register_tool_class(self, tool_name: str, executor_class: Type[BaseToolExecutor], 
                           executor_config: Dict[str, Any]):
        """Register a tool by class and config."""
        executor = executor_class(**executor_config)
        self.register_tool(tool_name, executor)
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        return self.config.get("tools", {}).get(tool_name, {})
    
    @asynccontextmanager
    async def _rate_limit_context(self, tool_name: str):
        """Context manager for rate limiting."""
        tool_config = self.get_tool_config(tool_name)
        
        if self.rate_limiter_pool and tool_config.get("enable_rate_limit", True):
            rate_limit = tool_config.get("rate_limit", 100)
            limiter = await self.rate_limiter_pool.get_or_create_limiter.remote(
                tool_name, rate_limit
            )
            
            # Acquire token
            acquired = False
            try:
                await limiter.acquire.remote()
                acquired = True
                yield True
            finally:
                if acquired:
                    limiter.release.remote()
        else:
            yield False

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Execute a tool with rate limiting and metrics."""
        if tool_name not in self.tool_executors:
            error_msg = f"Unknown tool: {tool_name}"
            if self.metrics_collector:
                await self.metrics_collector.record_call.remote(
                    tool_name, False, 0.0, error=error_msg
                )
            raise ValueError(error_msg)
        
        executor = self.tool_executors[tool_name]
        start_time = time.time()
        rate_limited = False
        
        try:
            async with self._rate_limit_context(tool_name) as was_rate_limited:
                rate_limited = was_rate_limited
                
                # Execute tool
                response, metadata = await executor.execute(parameters, **kwargs)
                
                # Record success
                latency = time.time() - start_time
                if self.metrics_collector:
                    await self.metrics_collector.record_call.remote(
                        tool_name, True, latency, rate_limited=rate_limited
                    )
                
                return response, metadata
                
        except Exception as e:
            # Record failure
            latency = time.time() - start_time
            if self.metrics_collector:
                await self.metrics_collector.record_call.remote(
                    tool_name, False, latency, 
                    rate_limited=rate_limited, error=str(e)
                )
            raise

    async def batch_execute_tool(self, requests: List[Tuple[str, Dict[str, Any]]], kwargs) -> List[Any]:
        """
        Executes multiple tool calls concurrently, respecting concurrency limits.

        Args:
            requests: A list of tuples, where each tuple is (tool_name, parameters).

        Returns:
            A list of results. Each result is either the successful output or an exception.
        """
        max_concurrent_calls = self.config.get("max_concurrent_calls", 20)
        semaphore = asyncio.Semaphore(max_concurrent_calls)

        async def _call_with_semaphore(tool_name: str, parameters: Dict[str, Any]):
            async with semaphore:
                return await self.execute_tool(tool_name, parameters)

        tasks = [_call_with_semaphore(name, params) for name, params in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


    async def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for tools."""
        if self.metrics_collector:
            return await self.metrics_collector.get_metrics.remote(tool_name)
        return {}
    
    async def get_rate_limiter_status(self) -> Dict[str, Dict]:
        """Get status of all rate limiters."""
        if self.rate_limiter_pool:
            return await self.rate_limiter_pool.get_all_status.remote()
        return {}
