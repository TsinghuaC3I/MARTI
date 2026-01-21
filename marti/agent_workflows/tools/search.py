# worlds.tools.search: 
# Support local search engine in search-r1
# TODO: Support google search with api-key
import json
from typing import Any, Dict, List, Tuple

from marti.agent_workflows.tools.base import BaseToolExecutor
from marti.agent_workflows.third_party.search_r1 import perform_single_search_batch
from marti.utils.logging_utils import init_logger

logger = init_logger(__name__)

class SearchToolExecutor(BaseToolExecutor):
    """Search tool executor using VERL's implementation."""
    
    def __init__(self, base_url: str, topk: int = 3, timeout: int = 15, **kwargs):
        self.base_url = base_url
        self.topk = topk
        self.timeout = timeout
        self.config = kwargs

    def get_name(self) -> str:
        return "search"

    async def execute(self, parameters: Dict[str, Any], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Execute search using VERL's perform_single_search_batch."""
        query_list = parameters.get("query_list", [])
        
        if not query_list:
            return "No queries provided", {"error": "empty_query_list"}
        
        # Use asyncio.run_in_executor for sync function
        import asyncio
        result_text, metadata = await asyncio.get_event_loop().run_in_executor(
            None,
            perform_single_search_batch,
            self.base_url,
            query_list,
            self.topk,
            None,  # concurrent_semaphore handled by ToolManager
            self.timeout
        )
        
        # Parse result for individual queries
        try:
            result_json = json.loads(result_text)
            result_str = result_json.get("result", "No results found")
            
            # Split by query separator if multiple queries
            # if "---" in result_str:
            #     results = result_str.split("\n---\n")
            # else:
            #     results = [result_str]
            
            # # Format response
            # response = "\n------\n".join(results)
            
            return result_str, metadata
            
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            return result_text, metadata
