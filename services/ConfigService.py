"""
ConfigService - Handles all configuration loading from database and files.

Uses DBManager for LLM credentials and FileConfigLoader for intents/sentiments/policies.
Prompts are loaded from file system using FilePromptLoader.
"""

from typing import List, Dict, Any, Optional
from utils.DBManager import DBManager
from utils.FilePromptLoader import get_prompt_loader
from utils.FileConfigLoader import get_config_loader
from schemas.intent import IntentLabel


class ConfigService:
    """
    Service for loading configuration from database and files.

    - LLM credentials: from database (via DBManager)
    - Intents, sentiments, policies: from file system (config/ directory)
    - Prompts: from file system (prompts/ directory)
    """

    def __init__(self, db_manager: DBManager, prompts_dir: str = "prompts", config_dir: str = "config"):
        """
        Initialize ConfigService.

        Args:
            db_manager: DBManager instance
            prompts_dir: Directory containing prompt files (default: "prompts")
            config_dir: Directory containing configuration files (default: "config")
        """
        self.db = db_manager
        self._cache: Dict[str, Any] = {}
        self.prompt_loader = get_prompt_loader(prompts_dir)
        self.config_loader = get_config_loader(config_dir)

    # ============ INTENTS ============

    async def get_intents(
        self,
        tenant_id: str = "default",
        use_cache: bool = True
    ) -> List[IntentLabel]:
        """
        Get intent labels for a tenant.

        Loads intents from file system (config/intents.json) with database fallback.

        Args:
            tenant_id: Tenant identifier
            use_cache: Whether to use cached intents

        Returns:
            List of IntentLabel objects

        Raises:
            ValueError: If no intents configured for tenant
        """
        cache_key = f"intents:{tenant_id}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Try loading from file system first
            intents_data = self.config_loader.get_intents(tenant_id)
        except FileNotFoundError:
            # Fallback to database
            results = await self.db.fetch(
                table="intent_labels",
                filters={"is_active": True}
            )

            # Filter by tenant (global or tenant-specific)
            intents_data = [
                r for r in results
                if r.get("tenant_id") is None or r.get("tenant_id") == tenant_id
            ]

            if not intents_data:
                raise ValueError(
                    f"No intents configured for tenant '{tenant_id}'. "
                    "Please configure intents in config/intents.json or in the database."
                )

        # Convert to IntentLabel objects
        intents = [
            IntentLabel(
                label=row.get("intent_label", row.get("label", row.get("name"))),
                description=row.get("description"),
                confidence_threshold=row.get("confidence_threshold", 0.7),
                tool_mapping=row.get("tool_mapping"),
                examples=row.get("examples")
            )
            for row in intents_data
        ]

        if use_cache:
            self._cache[cache_key] = intents

        return intents

    # ============ PROMPTS ============

    async def get_prompt(
        self,
        agent_name: str,
        version: str = "v1",
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Get prompt content for an agent.

        Loads prompts from file system (prompts/{agent_name}/{version}.txt).

        Args:
            agent_name: Name of the agent
            version: Prompt version
            tenant_id: Optional tenant identifier for tenant-specific prompts

        Returns:
            Prompt content string

        Raises:
            FileNotFoundError: If prompt file not found
        """
        try:
            return self.prompt_loader.get_prompt(agent_name, version, tenant_id)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{str(e)}\n"
                f"Expected prompt file: prompts/{agent_name}/{version}.txt\n"
                f"Please create the prompts directory and add prompt files."
            )

    # ============ POLICIES ============

    async def get_policy(
        self,
        tenant_id: str,
        intent_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get policy for tenant and intent.

        Args:
            tenant_id: Tenant identifier
            intent_type: Intent type

        Returns:
            Policy dictionary or None if not found
        """
        result = await self.db.fetch_one(
            table="policies",
            filters={
                "tenant_id": tenant_id,
                "intent_type": intent_type,
                "is_active": True
            }
        )
        return result

    # ============ SENTIMENTS ============

    async def get_sentiment_config(
        self,
        tenant_id: str = "default",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get sentiment configuration for tenant.

        Loads sentiments from file system (config/sentiments.json) with database fallback.

        Args:
            tenant_id: Tenant identifier
            use_cache: Whether to use cached sentiment config

        Returns:
            Sentiment configuration dictionary with:
                - sentiments: List of sentiment configs
                - escalation_rules: Escalation rules dict

        Raises:
            ValueError: If no sentiment configuration found
        """
        cache_key = f"sentiments:{tenant_id}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Try loading from file system first
            sentiment_config = self.config_loader.get_sentiment_config(tenant_id)
        except FileNotFoundError:
            # Fallback to database
            results = await self.db.fetch(
                table="sentiment_configs",
                filters={"is_active": True}
            )

            # Filter by tenant (global or tenant-specific)
            sentiments_data = [
                r for r in results
                if r.get("tenant_id") is None or r.get("tenant_id") == tenant_id
            ]

            if not sentiments_data:
                # Use default sentiment config
                sentiment_config = self.config_loader._get_default_sentiments()
            else:
                # Convert to expected format
                sentiment_config = {
                    "sentiments": sentiments_data,
                    "escalation_rules": {}
                }

        if use_cache:
            self._cache[cache_key] = sentiment_config

        return sentiment_config

    async def get_sentiments(
        self,
        tenant_id: str = "default",
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get list of sentiment configurations for a tenant.

        Args:
            tenant_id: Tenant identifier
            use_cache: Whether to use cached sentiment config

        Returns:
            List of sentiment configurations
        """
        config = await self.get_sentiment_config(tenant_id, use_cache)
        return config.get("sentiments", [])

    # ============ TOOLS ============

    async def get_tools(
        self,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available tools.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of tool configurations
        """
        filters = {"enabled": True}
        results = await self.db.fetch("tools", filters)

        # Filter by tenant if specified
        if tenant_id:
            return [r for r in results if r.get("tenant_id") is None or r.get("tenant_id") == tenant_id]

        return results

    # ============ AGENTS ============

    async def get_agents(
        self,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available agents.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of agent configurations
        """
        filters = {"enabled": True}
        results = await self.db.fetch("agents", filters)

        # Filter by tenant if specified
        if tenant_id:
            return [r for r in results if r.get("tenant_id") is None or r.get("tenant_id") == tenant_id]

        return results

    # ============ TENANT ============

    async def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tenant configuration.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant configuration or None if not found
        """
        return await self.db.fetch_one(
            table="tenants",
            filters={"tenant_id": tenant_id, "is_active": True}
        )

    # ============ CACHE MANAGEMENT ============

    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Clear configuration cache.

        Args:
            key: Specific cache key to clear, or None to clear all
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()
