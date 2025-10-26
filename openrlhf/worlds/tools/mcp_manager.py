import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from contextlib import asynccontextmanager

import ray
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ListToolsResult, Tool as McpTool, CallToolResult, TextContent

from marti.worlds.tools.utils import convert_mcp_to_openai_tools
from marti.worlds.tools.parser import ToolParser

# --- Setup Logging ---
# Configure logging for clear output
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))

# --- Actor Definition for Rate Limiting ---
# This implements the pattern suggested by your original code.

@ray.remote
class RateLimiter:
    """
    A Ray actor that acts as a token bucket rate limiter for a single tool.
    It refills tokens at a steady rate to allow for bursts of requests.
    """
    def __init__(self, calls_per_second: int):
        self.semaphore = asyncio.Semaphore(calls_per_second)
        self.calls_per_second = calls_per_second
        # The background task will automatically be managed by the Ray actor lifecycle.
        self.refill_task = asyncio.create_task(self._refill())

    async def _refill(self):
        """
        Background task to continuously add tokens back to the bucket
        at a rate of one per (1 / calls_per_second).
        """
        while True:
            await asyncio.sleep(1.0 / self.calls_per_second)
            if self.semaphore._value < self.calls_per_second:
                self.semaphore.release()
    
    async def acquire(self):
        """Acquires a token, blocking if the rate limit has been reached."""
        await self.semaphore.acquire()

    async def get_status(self) -> dict:
        """Returns the current status of the limiter."""
        return {
            "limit_per_second": self.calls_per_second,
            "available_tokens": self.semaphore._value
        }

@ray.remote
class RateLimiterPool:
    """
    A Ray actor that serves as a factory and registry for RateLimiter actors.
    It ensures that there is one RateLimiter actor per tool.
    """
    def __init__(self):
        self.limiters: Dict[str, ray.actor.ActorHandle] = {}

    async def get_or_create_limiter(self, name: str, calls_per_second: int) -> ray.actor.ActorHandle:
        """
        Returns the handle to an existing RateLimiter actor for the given name,
        or creates a new one if it doesn't exist.
        """
        if name not in self.limiters:
            logger.info(f"Creating new RateLimiter actor for '{name}' with limit {calls_per_second}/s.")
            self.limiters[name] = RateLimiter.remote(calls_per_second)
        return self.limiters[name]

    async def get_all_status(self) -> Dict[str, Any]:
        """Gathers and returns the status of all managed RateLimiter actors."""
        status_tasks = [limiter.get_status.remote() for limiter in self.limiters.values()]
        statuses = await asyncio.gather(*status_tasks)
        return {name: status for name, status in zip(self.limiters.keys(), statuses)}


# --- Actor Definition for Metrics Collection ---
# Reusing your provided ToolMetricsCollector directly.

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
        
    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for specific tool or all tools."""
        # Calculate average latency before returning
        for name in self.metrics:
            if self.metrics[name]["total_calls"] > 0:
                self.metrics[name]["avg_latency"] = self.metrics[name]["total_latency"] / self.metrics[name]["total_calls"]
                
        if tool_name:
            return dict(self.metrics.get(tool_name, {}))
        
        all_metrics = {name: dict(data) for name, data in self.metrics.items()}
        for name in all_metrics:
            all_metrics[name]['errors'] = dict(self.metrics[name]['errors'])
        
        summary = {
            "total_calls": sum(m["total_calls"] for m in all_metrics.values()),
            "successful_calls": sum(m["successful_calls"] for m in all_metrics.values()),
            "failed_calls": sum(m["failed_calls"] for m in all_metrics.values()),
            "uptime": time.time() - self.start_time,
            "tools_count": len(all_metrics)
        }
        
        return {"summary": summary, "tools": all_metrics}


# --- Final MCPManager ---
# Integrates MCP connection logic with Ray-based services.

class MCPManager:
    """
    Manages interaction with an MCP server, integrating with Ray for
    distributed rate limiting and metrics collection.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MCPManager.

        Args:
            config (Dict[str, Any]): Configuration dictionary. Expected keys include
                                     'enable_rate_limiting', 'enable_metrics', and 'tools'
                                     for tool-specific settings.
        """
        self.mcp_url = config["mcp_url"]
        self.config = config
        self.tool_parser = ToolParser()

        self.openai_tools: List[Dict] = []
        self.available_tools: Dict[str, Dict] = {}
        self._is_initialized = False

        # Initialize handles to the centralized Ray actors.
        # `get_if_exists=True` allows multiple MCPManager instances to share the same actors.
        self.rate_limiter_pool = None
        if self.config.get("enable_rate_limiting", True):
            self.rate_limiter_pool = RateLimiterPool.options(
                name="global-mcp-rate-limiter-pool", get_if_exists=True
            ).remote()

        self.metrics_collector = None
        if self.config.get("enable_metrics", True):
            self.metrics_collector = ToolMetricsCollector.options(
                name="global-mcp-tool-metrics", get_if_exists=True
            ).remote()
    
    def set_tools(self, tools):
        self.tools = tools

    def get_max_turns(self):
        return self.config.get("max_turns", 3)
    
    def get_num_workers(self):
        return self.config.get("num_workers", 128)

    def get_timeout(self):
        return self.config.get("timeout", 300)

    async def initialize(self):
        """
        Initializes the manager by discovering available tools from the MCP server.
        This must be called before making any tool calls.
        """
        if self._is_initialized:
            return
        logger.info("Initializing MCPManager and discovering tools...")
        try:
            async with self._get_session() as session:
                mcp_tools = await session.list_tools()
                self.openai_tools = convert_mcp_to_openai_tools(mcp_tools)
                for tool in self.openai_tools:
                    self.available_tools[tool["function"]["name"]] = tool

            self._is_initialized = True
            logger.info(f"Initialization complete. Discovered {len(self.available_tools)} tools.")
        except Exception as e:
            logger.error(f"Failed to discover tools from MCP server: {e}")
            self._is_initialized = False
            raise

    @asynccontextmanager
    async def _get_session(self):
        """
        An async context manager that provides a new, initialized MCP session.
        This encapsulates the correct connection and session lifecycle management.
        """
        try:
            async with streamablehttp_client(self.mcp_url, timeout=300) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
        except Exception as e:
            logger.error(f"Failed to create MCP session: {e}")
            raise

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Retrieves the specific configuration for a given tool."""
        return self.config.get("tools", {}).get(tool_name, {})

    @asynccontextmanager
    async def _rate_limit_context(self, tool_name: str):
        """
        An async context manager to handle rate limiting for a tool call.
        It interacts with the RateLimiterPool actor.
        """
        tool_config = self.get_tool_config(tool_name)
        if self.rate_limiter_pool and tool_config.get("enable_rate_limit", True):
            rate_limit = tool_config.get("rate_limit", 10) # Default rate limit
            limiter_actor = await self.rate_limiter_pool.get_or_create_limiter.remote(tool_name, rate_limit)
            
            await limiter_actor.acquire.remote()
            yield True
        else:
            yield False

    @staticmethod
    def _extract_content(obj: Union[CallToolResult, Dict, List, TextContent]) -> Any:
        if isinstance(obj, CallToolResult):
            return MCPManager._extract_content(obj.content)
        if isinstance(obj, list):
            return "\n".join(MCPManager._extract_content(o) for o in obj)
        if isinstance(obj, TextContent):
            return obj.text
        if isinstance(obj, dict):
            if 'text' in obj:
                return obj['text']
            if 'content' in obj:
                return MCPManager._extract_content(obj['content'])
            return {k: MCPManager._extract_content(v) for k, v in obj.items()}
        return str(obj)

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Executes a single tool call against the MCP server, with rate limiting and metrics.

        Args:
            tool_name (str): The name of the tool to call.
            parameters (Dict[str, Any]): The arguments for the tool.

        Returns:
            A tuple containing the string result and metadata about the call.
        """
        if not self._is_initialized:
            raise RuntimeError("MCPManager is not initialized. Please call `await manager.initialize()` first.")
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' is not available.")

        start_time = time.time()
        was_rate_limited = False
        error_str = None

        # --- IMPROVEMENT: Get timeout configuration ---
        # Get tool-specific timeout, or fall back to global default (e.g., 60s)
        timeout = self.config.get("timeout", 300.0)

        try:
            async with self._rate_limit_context(tool_name) as was_rate_limited_context:
                was_rate_limited = was_rate_limited_context
                
                # The core logic that performs the network request
                async def timed_section():
                    async with self._get_session() as session:
                        return await session.call_tool(tool_name, arguments=parameters)

                # Wrap the core logic with the configured timeout
                result = await asyncio.wait_for(timed_section(), timeout=timeout)

                if result.isError:
                    error_str = result.content[0].text if result.content else f"Unknown error from tool '{tool_name}'"
                    return error_str, {"tool_name": tool_name, "success": False, "error": error_str}
                
                response_content = self._extract_content(result)
                return response_content, {"tool_name": tool_name, "success": True}

        except asyncio.TimeoutError:
            error_str = f"Tool '{tool_name}' call timed out after {timeout}s"
            logger.warning(error_str)
            # Raise a generic runtime error that agent loops can easily catch
            return error_str, {"tool_name": tool_name, "success": False, "error": error_str}

        except Exception as e:
            # Capture any other exception, including the RuntimeError from isError
            error_str = error_str or str(e)
            return error_str, {"tool_name": tool_name, "success": False, "error": error_str}

        finally:
            latency = time.time() - start_time
            if self.metrics_collector:
                success = error_str is None
                await self.metrics_collector.record_call.remote(
                    tool_name, success, latency, was_rate_limited, error_str
                )

        # start_time = time.time()
        # was_rate_limited = False
        # error_str = None
        
        # try:
        #     async with self._rate_limit_context(tool_name) as was_rate_limited:
        #         async with self._get_session() as session:
        #             result = await session.call_tool(tool_name, arguments=parameters)
                
        #         if result.isError:
        #             error_str = result.content[0].text if result.content else "Unknown tool error"
        #             raise RuntimeError(error_str)
                
        #         response_content = self._extract_content(result)
        #         return response_content, {"tool_name": tool_name, "success": True}

        # except Exception as e:
        #     error_str = error_str or str(e)
        #     # logger.error(f"Error calling tool '{tool_name}': {error_str}")
        #     # Re-raise the exception after recording metrics
        #     raise
        # finally:
        #     latency = time.time() - start_time
        #     if self.metrics_collector:
        #         success = error_str is None
        #         await self.metrics_collector.record_call.remote(
        #             tool_name, success, latency, was_rate_limited, error_str
        #         )

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
        """Retrieves metrics from the ToolMetricsCollector actor."""
        if self.metrics_collector:
            return await self.metrics_collector.get_metrics.remote(tool_name)
        return {"status": "Metrics are disabled."}
    
    async def get_rate_limiter_status(self) -> Dict[str, Dict]:
        """Retrieves the status of all rate limiters from the RateLimiterPool actor."""
        if self.rate_limiter_pool:
            return await self.rate_limiter_pool.get_all_status.remote()
        return {"status": "Rate limiting is disabled."}

# --- Main execution block for demonstration ---
async def main():
    """Demonstrates the usage of the MCPManager with Ray."""
    # Start or connect to a Ray cluster.
    # For local testing, ray.init() is sufficient.
    # In a cluster environment, you might use ray.init(address='auto').
    ray.init()

    try:
        # Define configuration for the manager
        mcp_config = {
            "mcp_url": "http://101.6.64.254:15555/mcp",
            "max_concurrent_calls": 50,
            "enable_rate_limiting": True,
            "enable_metrics": True,
            "tools": {
                "web_search": {"rate_limit": 5}, # Limit web_search to 5 calls/second
                "wiki_lookup": {"rate_limit": 10}, # Limit wiki_lookup to 10 calls/second
            }
        }

        # Instantiate the manager
        manager = MCPManager(config=mcp_config)

        # 1. Initialize the manager
        await manager.initialize()

        # 2. Perform a batch of concurrent calls to test rate limiting and metrics
        logger.info("--- Starting batch tool calls ---")
        requests = [
            ("web_search", {"query": f"Ray framework topic {i}"}) for i in range(7)
        ] + [
            ("wiki_lookup", {"entity": f"Entity {i}"}) for i in range(3)
        ] + [
            ("non_existent_tool", {"query": "this will fail"}) # Intentionally failing call
        ]

        start_batch_time = time.time()
        results = await manager.batch_call_tools(requests)
        batch_duration = time.time() - start_batch_time
        logger.info(f"--- Batch of {len(requests)} calls completed in {batch_duration:.2f}s ---")

        # Log results for inspection
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning(f"Request {i+1} ({requests[i][0]}) failed: {type(res).__name__}")
            else:
                logger.info(f"Request {i+1} ({requests[i][0]}): {res}")
        
        # Add a small delay to allow metrics to be fully processed
        await asyncio.sleep(0.5)

        # 3. Retrieve and print status and metrics
        logger.info("\n--- Retrieving final status and metrics ---")

        limiter_status = await manager.get_rate_limiter_status()
        print("\n[Rate Limiter Status]")
        print(json.dumps(limiter_status, indent=2))

        metrics = await manager.get_metrics()
        print("\n[Tool Call Metrics]")
        print(json.dumps(metrics, indent=2, default=str))

    finally:
        # Shutdown the Ray connection
        ray.shutdown()
        logger.info("Ray has been shut down.")


if __name__ == "__main__":
    asyncio.run(main())