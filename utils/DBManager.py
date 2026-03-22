"""
DBManager - Database connection manager for Supabase.

Handles database connections and provides a unified interface for database operations.
"""

import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()


class DBManager:
    """
    Database manager for Supabase.

    Provides methods to connect to Supabase and execute queries.
    """

    _instance: Optional['DBManager'] = None
    _client = None

    def __new__(cls):
        """Singleton pattern to ensure only one DBManager instance."""
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize DBManager (singleton)."""
        if self._initialized:
            return

        self.supabase_url = os.getenv("SUPABASE_URL")
        # Use service role key for backend operations (bypasses RLS)
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
        self._connection = None
        self._initialized = False

    async def initialize(self):
        """
        Initialize Supabase connection.

        Raises:
            ValueError: If Supabase credentials are not configured
        """
        if self._initialized:
            return

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase credentials not configured. "
                "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your environment."
            )

        try:
            from supabase import create_client, Client
            self._connection: Client = create_client(
                self.supabase_url,
                self.supabase_key
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "Supabase client not installed. "
                "Install with: pip install supabase"
            )

    def get_connection(self):
        """
        Get the Supabase client connection.

        Returns:
            Supabase client

        Raises:
            RuntimeError: If connection is not initialized
        """
        if not self._initialized or not self._connection:
            raise RuntimeError(
                "Database connection not initialized. "
                "Call await DBManager().initialize() first."
            )
        return self._connection

    async def fetch(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        select: str = "*"
    ) -> List[Dict[str, Any]]:
        """
        Fetch records from a table.

        Args:
            table: Table name
            filters: Optional dictionary of field=value filters
            select: Columns to select (default: "*")

        Returns:
            List of dictionaries representing rows

        Raises:
            RuntimeError: If connection is not initialized
        """
        client = self.get_connection()

        query = client.table(table).select(select)

        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)

        response = query.execute()

        return response.data if response.data else []

    async def fetch_one(
        self,
        table: str,
        filters: Dict[str, Any],
        select: str = "*"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch one record from a table.

        Args:
            table: Table name
            filters: Dictionary of field=value filters
            select: Columns to select (default: "*")

        Returns:
            Dictionary representing the row, or None if not found
        """
        results = await self.fetch(table, filters, select)
        print(f"fetch_one results for table '{table}' with filters {filters}: {results}")
        return results[0] if results else None

    async def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Insert a record into a table.

        Args:
            table: Table name
            data: Dictionary of field=value pairs

        Returns:
            Inserted record as dictionary

        Raises:
            RuntimeError: If connection is not initialized
        """
        client = self.get_connection()

        response = client.table(table).insert(data).execute()

        return response.data[0] if response.data else {}

    async def update(
        self,
        table: str,
        filters: Dict[str, Any],
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Update records in a table.

        Args:
            table: Table name
            filters: Dictionary of field=value filters for WHERE clause
            data: Dictionary of field=value pairs to update

        Returns:
            List of updated records

        Raises:
            RuntimeError: If connection is not initialized
        """
        client = self.get_connection()

        query = client.table(table).update(data)

        for key, value in filters.items():
            query = query.eq(key, value)

        response = query.execute()

        return response.data if response.data else []

    async def delete(
        self,
        table: str,
        filters: Dict[str, Any]
    ) -> None:
        """
        Delete records from a table.

        Args:
            table: Table name
            filters: Dictionary of field=value filters for WHERE clause

        Raises:
            RuntimeError: If connection is not initialized
        """
        client = self.get_connection()

        query = client.table(table).delete()

        for key, value in filters.items():
            query = query.eq(key, value)

        query.execute()

    async def execute_rpc(
        self,
        function_name: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a Remote Procedure Call (PostgreSQL function).

        Args:
            function_name: Name of the PostgreSQL function
            params: Optional parameters for the function

        Returns:
            Function result

        Raises:
            RuntimeError: If connection is not initialized
        """
        client = self.get_connection()

        response = client.rpc(function_name, params or {}).execute()

        return response.data if response.data else None

    async def table_exists(self, table: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table: Table name

        Returns:
            True if table exists, False otherwise
        """
        try:
            result = await self.fetch(table, limit=1)
            return True
        except Exception:
            return False

    def is_initialized(self) -> bool:
        """Check if database connection is initialized."""
        return self._initialized

    async def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection = None
            self._initialized = False


# Global instance
_db_manager: Optional[DBManager] = None


async def get_db_manager() -> DBManager:
    """
    Get the global DBManager instance.

    Returns:
        DBManager instance

    Raises:
        RuntimeError: If database is not initialized
    """
    global _db_manager

    if _db_manager is None:
        _db_manager = DBManager()
        await _db_manager.initialize()

    return _db_manager
