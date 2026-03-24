"""
WebSearchTool - Web search using Tavily and Serper APIs.

Returns 5 results from each API, combined for comprehensive search results.
"""

import os
import asyncio
from typing import Dict, Any, Optional
from httpx import AsyncClient, HTTPError
from core.BaseTools import BaseTool
from models.tool import ToolResult, ToolConfig


class WebSearchTool(BaseTool):
    """
    Web search tool using Tavily and Serper APIs.

    Fetches 5 results from each API and combines them for comprehensive coverage.
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize WebSearchTool with API keys from environment."""
        super().__init__(config)

        # Get API keys from environment
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")

        # API endpoints
        self.tavily_url = "https://api.tavily.com/search"
        self.serper_url = "https://google.serper.dev/search"

        # Validate API keys
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY not found in environment")

    async def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """
        Execute web search using both Tavily and Serper APIs.

        Args:
            payload: Must contain 'query' field
                - query: Search query string
                - max_results: Optional, max results per API (default: 5)

        Returns:
            ToolResult with combined search results from both APIs
        """
        # Validate payload
        is_valid, error_msg = self.validate_payload(payload, ["query"])
        if not is_valid:
            return ToolResult.failed(error_msg)

        query = payload.get("query")
        max_results = payload.get("max_results", 5)

        # Run both searches in parallel
        async with AsyncClient(timeout=30.0) as client:
            tavily_task = self._search_tavily(client, query, max_results)
            serper_task = self._search_serper(client, query, max_results)

            tavily_results, serper_results = await asyncio.gather(
                tavily_task,
                serper_task,
                return_exceptions=True
            )

        # Handle exceptions
        if isinstance(tavily_results, Exception):
            tavily_results = {"error": str(tavily_results), "results": []}
        if isinstance(serper_results, Exception):
            serper_results = {"error": str(serper_results), "results": []}

        # Combine results
        combined_results = self._combine_results(
            tavily_results if not isinstance(tavily_results, Exception) else {"results": []},
            serper_results if not isinstance(serper_results, Exception) else {"results": []}
        )

        return ToolResult.success(
            data={
                "query": query,
                "tavily_results": tavily_results.get("results", []),
                "serper_results": serper_results.get("results", []),
                "combined_results": combined_results,
                "total_results": len(combined_results)
            },
            metadata={
                "tavily_error": tavily_results.get("error") if isinstance(tavily_results, dict) and "error" in tavily_results else None,
                "serper_error": serper_results.get("error") if isinstance(serper_results, dict) and "error" in serper_results else None
            }
        )

    async def _search_tavily(self, client: AsyncClient, query: str, max_results: int) -> Dict[str, Any]:
        """
        Search using Tavily API.

        Args:
            client: HTTPX async client
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dict with search results
        """
        try:
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False
            }

            response = await client.post(self.tavily_url, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract results
            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "source": "tavily"
                })

            return {"results": results}

        except HTTPError as e:
            return {"error": f"Tavily API error: {str(e)}", "results": []}
        except Exception as e:
            return {"error": f"Tavily error: {str(e)}", "results": []}

    async def _search_serper(self, client: AsyncClient, query: str, max_results: int) -> Dict[str, Any]:
        """
        Search using Serper API.

        Args:
            client: HTTPX async client
            query: Search query
            max_results: Maximum number of results

        Returns:
            Dict with search results
        """
        try:
            payload = {
                "q": query,
                "num": max_results
            }

            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }

            response = await client.post(self.serper_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Extract results
            results = []
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "score": 0.0,  # Serper doesn't provide scores
                    "source": "serper"
                })

            return {"results": results}

        except HTTPError as e:
            return {"error": f"Serper API error: {str(e)}", "results": []}
        except Exception as e:
            return {"error": f"Serper error: {str(e)}", "results": []}

    def _combine_results(self, tavily_data: Dict[str, Any], serper_data: Dict[str, Any]) -> list:
        """
        Combine results from both APIs, removing duplicates.

        Args:
            tavily_data: Tavily search results
            serper_data: Serper search results

        Returns:
            Combined and deduplicated results
        """
        tavily_results = tavily_data.get("results", [])
        serper_results = serper_data.get("results", [])

        # Track URLs to avoid duplicates
        seen_urls = set()
        combined = []

        # Add Tavily results first (they have scores)
        for result in tavily_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)

        # Add Serper results that aren't duplicates
        for result in serper_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)

        return combined
