"""Web Search Tool for external knowledge retrieval.

Supports:
- Tavily API (optimized for LLM applications)
- Serper API (Google Search API)
"""

import os
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool for external knowledge retrieval.

    Supported providers:
    - Tavily: https://tavily.com (optimized for AI/LLM applications)
    - Serper: https://serper.dev (Google Search API)
    """

    def __init__(
        self,
        provider: str = "tavily",
        max_results: int = 10,
        search_depth: str = "basic",
        tavily_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None
    ):
        """Initialize WebSearchTool.

        Args:
            provider: Search provider (tavily, serper)
            max_results: Maximum number of results to return
            search_depth: Search depth - basic or advanced (Tavily only)
            tavily_api_key: Tavily API key (defaults to env var)
            serper_api_key: Serper API key (defaults to env var)
        """
        self.provider = provider
        self.max_results = max_results
        self.search_depth = search_depth

        # API keys (from params or environment)
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")

        if not self.tavily_api_key and not self.serper_api_key:
            logger.warning("No web search API keys found. Set TAVILY_API_KEY or SERPER_API_KEY.")

    def search(
        self,
        query: str,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform web search.

        Args:
            query: Search query
            max_results: Override max results

        Returns:
            Dict with results
        """
        if not query:
            return {
                "results": [],
                "total_results": 0,
                "provider": self.provider,
                "query": query,
                "error": "Empty query"
            }

        max_results = max_results or self.max_results

        # Perform search
        results = []
        if self.provider == "tavily" and self.tavily_api_key:
            results = self._search_tavily(query, max_results)
        elif self.provider == "serper" and self.serper_api_key:
            results = self._search_serper(query, max_results)

        return {
            "results": results,
            "total_results": len(results),
            "provider": self.provider,
            "query": query,
            "error": None if results else "No results found or API error"
        }

    def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            import requests

            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": self.search_depth,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                    "published_date": item.get("published_date")
                })

            logger.info(f"Tavily search returned {len(results)} results for: {query}")
            return results

        except ImportError:
            logger.error("requests library not installed")
            return []
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    def _search_serper(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Serper API (Google Search)."""
        try:
            import requests

            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": max_results
            }

            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("organic", []):
                content = item.get("snippet") or item.get("description") or ""
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "content": content,
                    "score": 0.8,  # Serper doesn't provide scores
                    "published_date": None
                })

            logger.info(f"Serper search returned {len(results)} results for: {query}")
            return results

        except ImportError:
            logger.error("requests library not installed")
            return []
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for LangChain/LangGraph integration."""
        return {
            "name": "web_search",
            "description": "Search the web for current information. Use when internal knowledge is insufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }


# Convenience function for quick usage
def search_web(
    query: str,
    provider: str = "tavily",
    max_results: int = 10
) -> Dict[str, Any]:
    """Quick web search function.

    Args:
        query: Search query
        provider: Search provider (tavily, serper)
        max_results: Maximum results

    Returns:
        Dict with search results
    """
    tool = WebSearchTool(provider=provider, max_results=max_results)
    return tool.search(query)
