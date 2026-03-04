"""Web Search Agent for external knowledge retrieval.

Supports:
- Tavily API (optimized for LLM applications)
- Serper API (Google Search API)
- Multi-tenant configuration
- Fallback between providers
"""

import time
import os
from typing import Dict, Any, List, Optional
from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch
from models.sections import WebSearchSchema, WebSearchResult
import logging

logger = logging.getLogger(__name__)


class WebSearchAgent(BaseAgent):
    """
    Performs web search when internal knowledge is insufficient.

    Reads:
    - understanding.input.raw_text: User query for search
    - knowledge.has_relevant_content: Whether RAG found relevant content
    - context.tenant_id: For tenant-specific configuration

    Writes:
    - web_search.query: Original search query
    - web_search.results: Web search results
    - web_search.total_results: Number of results
    - web_search.search_provider: Provider used (tavily, serper)
    - web_search.has_results: Whether results were found
    - web_search.search_latency_ms: Total search time
    """

    agent_name = "WebSearchAgent"
    allowed_section = "web_search"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")

        # Web search configuration
        self.web_search_config = config.get("web_search_config", {})
        self.primary_provider = self.web_search_config.get("primary_provider", "tavily")
        self.max_results = self.web_search_config.get("max_results", 10)
        self.search_depth = self.web_search_config.get("search_depth", "basic")

        # API keys
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.serper_api_key = os.getenv("SERPER_API_KEY")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Execute web search."""
        start_time = time.time()

        # -------------------------------------------------
        # 1️⃣ Check if web search is enabled
        # -------------------------------------------------
        if not self._is_web_search_enabled():
            return self._empty_result("Web search disabled", context)

        # -------------------------------------------------
        # 2️⃣ Extract query from state
        # -------------------------------------------------
        understanding = state.get("understanding", {})
        query = understanding.get("input", {}).get("raw_text", "")

        if not query:
            return self._empty_result("No query found", context)

        # -------------------------------------------------
        # 3️⃣ Optimize query for web search
        # -------------------------------------------------
        optimized_query = self._optimize_query(query, understanding)
        logger.info(f"Web search query: {optimized_query}")

        # -------------------------------------------------
        # 4️⃣ Perform web search
        # -------------------------------------------------
        results = []
        provider_used = None

        if self.primary_provider == "tavily" and self.tavily_api_key:
            results = self._search_tavily(optimized_query)
            provider_used = "tavily"
        elif self.primary_provider == "serper" and self.serper_api_key:
            results = self._search_serper(optimized_query)
            provider_used = "serper"

        # -------------------------------------------------
        # 5️⃣ Build final result
        # -------------------------------------------------
        search_latency_ms = int((time.time() - start_time) * 1000)
        has_results = len(results) > 0
        confidence = sum(r.score for r in results) / len(results) if results else 0.0

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0 if has_results else 0.0,
            changes={
                "query": optimized_query,
                "results": [self._result_to_dict(r) for r in results],
                "total_results": len(results),
                "search_provider": provider_used or "none",
                "has_results": has_results,
                "search_latency_ms": search_latency_ms
            }
        )

    def _search_tavily(self, query: str) -> List[WebSearchResult]:
        """Search using Tavily API (optimized for LLMs)."""
        try:
            import requests

            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": self.search_depth,
                "max_results": self.max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0),
                    source="tavily",
                    published_date=item.get("published_date")
                ))

            logger.info(f"Tavily search returned {len(results)} results")
            return results

        except ImportError:
            logger.error("requests library not installed")
            return []
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

    def _search_serper(self, query: str) -> List[WebSearchResult]:
        """Search using Serper API (Google Search)."""
        try:
            import requests

            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": self.max_results
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
                # Extract snippet from snippet or description field
                content = item.get("snippet") or item.get("description") or ""

                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    content=content,
                    score=0.8,  # Serper doesn't provide scores, use default
                    source="serper",
                    published_date=None
                ))

            logger.info(f"Serper search returned {len(results)} results")
            return results

        except ImportError:
            logger.error("requests library not installed")
            return []
        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []

    def _optimize_query(self, query: str, understanding: dict) -> str:
        """Optimize query for web search.

        Can be enhanced with:
        - Entity extraction
        - Query rewriting
        - Adding context from intent
        """
        # Basic optimization: strip unnecessary words
        query = query.strip()

        # Add context from intent if available
        intent = understanding.get("intent", {})
        intent_name = intent.get("name", "")

        # For FAQ queries, add "FAQ" or "help" to search
        if intent_name == "FAQ_QUERY":
            if "faq" not in query.lower() and "help" not in query.lower():
                query = f"{query} FAQ help"

        return query

    def _is_web_search_enabled(self) -> bool:
        """Check if web search is enabled in config."""
        if self.config_loader:
            return self.config_loader.is_web_search_enabled()
        return False

    def _empty_result(
        self,
        reason: str,
        context: AgentExecutionContext,
        query: str = ""
    ) -> Patch:
        """Return empty result patch."""
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "query": query,
                "results": [],
                "total_results": 0,
                "search_provider": "none",
                "has_results": False,
                "search_latency_ms": 0
            }
        )

    def _result_to_dict(self, result: WebSearchResult) -> Dict[str, Any]:
        """Convert WebSearchResult to dict for state storage."""
        return {
            "title": result.title,
            "url": result.url,
            "content": result.content,
            "score": result.score,
            "source": result.source,
            "published_date": result.published_date
        }
